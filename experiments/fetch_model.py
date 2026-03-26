#!/usr/bin/env python3
"""Download a HuggingFace model and split its transformer layers into
individual files that RAIS LayerStreamer can read.

Usage:
    python3 experiments/fetch_model.py [model_id] [output_dir]

Defaults:
    model_id  = HuggingFaceTB/SmolLM2-135M
    output_dir = experiments/models/smollm2-135m

Output structure:
    output_dir/
        layer_0000    # all tensors for transformer layer 0, concatenated
        layer_0001
        ...
        manifest.tsv  # layer_idx  size_bytes  num_tensors
"""

import json
import os
import struct
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files


def find_safetensor_files(model_id: str) -> list[str]:
    """List all .safetensors files in the repo."""
    files = list_repo_files(model_id)
    return sorted(f for f in files if f.endswith(".safetensors"))


def download_safetensors(model_id: str, filenames: list[str], cache_dir: str) -> list[str]:
    """Download safetensor files and return local paths."""
    paths = []
    for fname in filenames:
        p = hf_hub_download(model_id, fname, cache_dir=cache_dir)
        paths.append(p)
        print(f"  downloaded {fname} -> {p}")
    return paths


def parse_safetensors_header(path: str):
    """Parse safetensors header to get tensor metadata and data offset."""
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header_json = f.read(header_len)
    header = json.loads(header_json)
    data_offset = 8 + header_len
    # Remove __metadata__ key if present
    header.pop("__metadata__", None)
    return header, data_offset


def read_tensor_raw(path: str, data_offset: int, offsets: list[int]) -> bytes:
    """Read raw tensor bytes from a safetensors file."""
    begin, end = offsets
    with open(path, "rb") as f:
        f.seek(data_offset + begin)
        return f.read(end - begin)


def extract_layers(st_paths: list[str], output_dir: Path):
    """Group tensors by transformer layer and write each layer to a file.

    Naming convention assumed by most HF models:
        model.layers.{N}.{...}     (LLaMA, Mistral, SmolLM, etc.)
    Non-layer tensors (embed_tokens, lm_head, norm) are written to layer_embed
    and layer_head respectively.
    """
    # Collect tensors per layer index
    layer_tensors: dict[int, list[tuple[str, int]]] = {}  # key -> (key, size)
    non_layer: list[tuple[str, int]] = []

    # Build a list of (st_path, data_offset, key, begin, end, layer_idx_or_None)
    all_tensors = []

    for st_path in st_paths:
        header, data_offset = parse_safetensors_header(st_path)
        for key, meta in header.items():
            offsets = meta["data_offsets"]
            size = offsets[1] - offsets[0]

            # Try to extract layer index from key
            parts = key.split(".")
            layer_idx = None
            for i, part in enumerate(parts):
                if part == "layers" and i + 1 < len(parts):
                    try:
                        layer_idx = int(parts[i + 1])
                        break
                    except ValueError:
                        pass

            all_tensors.append((st_path, data_offset, key, offsets, size, layer_idx))

            if layer_idx is not None:
                layer_tensors.setdefault(layer_idx, []).append(len(all_tensors) - 1)
            else:
                non_layer.append(len(all_tensors) - 1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Write non-layer tensors (embeddings, final norm, lm_head) as layer_0000
    # and shift transformer layers to start at layer_0001
    manifest = []
    file_idx = 0

    if non_layer:
        path = output_dir / f"layer_{file_idx:04d}"
        total = 0
        with open(path, "wb") as f:
            for tidx in non_layer:
                st_path, data_offset, key, offsets, size, _ = all_tensors[tidx]
                raw = read_tensor_raw(st_path, data_offset, offsets)
                f.write(raw)
                total += size
        manifest.append((file_idx, total, len(non_layer)))
        print(f"  layer_{file_idx:04d}: {total / 1024 / 1024:.1f} MB  ({len(non_layer)} tensors: embed/norm/head)")
        file_idx += 1

    # Write transformer layers in order
    for layer_idx in sorted(layer_tensors.keys()):
        tensor_indices = layer_tensors[layer_idx]
        path = output_dir / f"layer_{file_idx:04d}"
        total = 0
        with open(path, "wb") as f:
            for tidx in tensor_indices:
                st_path, data_offset, key, offsets, size, _ = all_tensors[tidx]
                raw = read_tensor_raw(st_path, data_offset, offsets)
                f.write(raw)
                total += size
        manifest.append((file_idx, total, len(tensor_indices)))
        print(f"  layer_{file_idx:04d}: {total / 1024 / 1024:.1f} MB  (transformer layer {layer_idx}, {len(tensor_indices)} tensors)")
        file_idx += 1

    # Write manifest
    manifest_path = output_dir / "manifest.tsv"
    with open(manifest_path, "w") as f:
        f.write("layer_idx\tsize_bytes\tnum_tensors\n")
        for idx, size, count in manifest:
            f.write(f"{idx}\t{size}\t{count}\n")

    print(f"\n  {file_idx} layer files written to {output_dir}")
    print(f"  manifest: {manifest_path}")
    return file_idx, manifest


def main():
    model_id = sys.argv[1] if len(sys.argv) > 1 else "HuggingFaceTB/SmolLM2-135M"
    default_dir = "experiments/models/" + model_id.split("/")[-1].lower()
    output_dir = Path(sys.argv[2] if len(sys.argv) > 2 else default_dir)
    cache_dir = str(Path("experiments/.hf_cache").resolve())

    print(f"Model:  {model_id}")
    print(f"Output: {output_dir}\n")

    print("Finding safetensor files...")
    st_files = find_safetensor_files(model_id)
    if not st_files:
        print("No .safetensors files found in repo. Aborting.")
        sys.exit(1)
    print(f"  found: {st_files}\n")

    print("Downloading...")
    local_paths = download_safetensors(model_id, st_files, cache_dir)

    print("\nExtracting layers...")
    num_layers, manifest = extract_layers(local_paths, output_dir)

    # Print summary for the C++ benchmark
    sizes = [m[1] for m in manifest]
    max_size = max(sizes)
    total_size = sum(sizes)
    print(f"\n  Total model size: {total_size / 1024 / 1024:.1f} MB")
    print(f"  Max layer size:   {max_size / 1024 / 1024:.1f} MB")
    print(f"  Num layers:       {num_layers}")
    print(f"\nRun benchmark:")
    print(f"  ./build/bench_inference_llm {output_dir}")


if __name__ == "__main__":
    main()
