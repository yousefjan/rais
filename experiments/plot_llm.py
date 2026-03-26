#!/usr/bin/env python3
"""Plot TTFT and tok/s from bench_inference_llm TSV output.

Usage:
    ./build/bench_inference_llm models/smollm2-135m models/tinyllama > results.tsv
    python3 experiments/plot_llm.py experiments/llm_results.tsv
"""

import sys
import csv
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "experiments/llm_results.tsv"
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)

    # Group by model
    models = []
    seen = set()
    for r in rows:
        if r["model"] not in seen:
            models.append(r["model"])
            seen.add(r["model"])

    naive_ttft = []
    rais_ttft  = []
    naive_toks = []
    rais_toks  = []

    for m in models:
        for r in rows:
            if r["model"] == m and r["approach"] == "naive":
                naive_ttft.append(float(r["ttft_us"]))
                naive_toks.append(float(r["tok_s"]))
            elif r["model"] == m and r["approach"] == "rais":
                rais_ttft.append(float(r["ttft_us"]))
                rais_toks.append(float(r["tok_s"]))

    x = np.arange(len(models))
    w = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # --- TTFT ---
    ax1.bar(x - w/2, naive_ttft, w, label="Naive (sequential)", color="#d45d5d")
    ax1.bar(x + w/2, rais_ttft,  w, label="RAIS (pipelined)",   color="#4a90d9")
    ax1.set_xlabel("Model")
    ax1.set_ylabel("TTFT (us)")
    ax1.set_title("Time to First Token")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=15, ha="right")
    ax1.legend()
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))

    for i in range(len(models)):
        if rais_ttft[i] > 0:
            speedup = naive_ttft[i] / rais_ttft[i]
            higher = max(naive_ttft[i], rais_ttft[i])
            ax1.annotate(f"{speedup:.2f}x",
                         xy=(x[i], higher),
                         xytext=(0, 8), textcoords="offset points",
                         ha="center", fontsize=9, fontweight="bold", color="#2a5d8a")

    # --- Tok/s ---
    ax2.bar(x - w/2, naive_toks, w, label="Naive (sequential)", color="#d45d5d")
    ax2.bar(x + w/2, rais_toks,  w, label="RAIS (pipelined)",   color="#4a90d9")
    ax2.set_xlabel("Model")
    ax2.set_ylabel("Tokens / sec")
    ax2.set_title("Throughput (tok/s)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=15, ha="right")
    ax2.legend()
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))

    for i in range(len(models)):
        if naive_toks[i] > 0:
            speedup = rais_toks[i] / naive_toks[i]
            higher = max(naive_toks[i], rais_toks[i])
            ax2.annotate(f"{speedup:.2f}x",
                         xy=(x[i], higher),
                         xytext=(0, 8), textcoords="offset points",
                         ha="center", fontsize=9, fontweight="bold", color="#2a5d8a")

    fig.suptitle("Naive Sequential vs RAIS Pipelined — Real LLM Weights", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    out = path.rsplit(".", 1)[0] + ".png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
