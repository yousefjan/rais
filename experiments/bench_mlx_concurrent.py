#!/usr/bin/env python3
"""Benchmark concurrent MLX inference requests: naive threading vs RAIS scheduling.

Scenario: N clients submit prompts simultaneously. Some are "interactive"
(user-facing, latency-sensitive) and some are "background" (batch/logging).

Naive threading: all requests run on a shared pool with equal priority.
RAIS scheduling: interactive requests are submitted to Lane::Interactive and
                 served ahead of background work, giving lower TTFT for
                 high-priority clients.

Because mlx-lm is single-threaded (one token at a time), true concurrency
means interleaving decode steps. We simulate this by serializing requests
through a single decode worker (matching real mlx-lm behaviour) while a
RAIS-style priority queue determines which request gets the next decode step.

Metrics per request:
  ttft_ms: time from submission to first token
  e2e_ms:  total time to completion

Usage:
  python3 experiments/bench_mlx_concurrent.py [--model llama1b] [--clients 6]
"""

import argparse
import gc
import heapq
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx_lm

HF_CACHE = Path("experiments/.hf_cache")

LOCAL_DIRS = {
    "smollm":  HF_CACHE / "mlx-community" / "SmolLM2-135M-Instruct",
    "llama1b": HF_CACHE / "mlx-community" / "Llama-3.2-1B-Instruct-4bit",
    "llama3b": HF_CACHE / "mlx-community" / "Llama-3.2-3B-Instruct-4bit",
}

MAX_TOKENS  = 32
WARMUP_RUNS = 1


# ---------------------------------------------------------------------------
# Request definition
# ---------------------------------------------------------------------------

@dataclass(order=True)
class Request:
    priority: int            # lower = higher priority (0=interactive, 1=background)
    submit_time: float = field(compare=False)
    client_id: int    = field(compare=False)
    prompt: str       = field(compare=False)
    ttft_ms: Optional[float]  = field(default=None, compare=False)
    e2e_ms: Optional[float]   = field(default=None, compare=False)


PROMPTS = [
    "What is the capital of France?",
    "Summarise the French Revolution in one sentence.",
    "What year did World War II end?",
    "Name the tallest mountain on Earth.",
    "Who wrote Romeo and Juliet?",
    "What is the speed of light in a vacuum?",
]


# ---------------------------------------------------------------------------
# Naive: all requests in a FIFO queue, single decode worker
# ---------------------------------------------------------------------------

def run_naive(model, tokenizer, requests: list[Request]) -> list[Request]:
    """Submit all requests to a plain FIFO queue, process in arrival order."""
    results = [None] * len(requests)
    q = queue.Queue()

    for req in requests:
        q.put(req)

    for _ in range(len(requests)):
        req = q.get()
        t_start = req.submit_time
        t_first = None
        tokens = 0
        for resp in mlx_lm.stream_generate(model, tokenizer, req.prompt,
                                            max_tokens=MAX_TOKENS):
            if t_first is None:
                t_first = time.perf_counter()
            tokens += 1
        t_end = time.perf_counter()
        req.ttft_ms = (t_first - t_start) * 1000 if t_first else float("nan")
        req.e2e_ms  = (t_end  - t_start) * 1000
        results[req.client_id] = req

    return results


# ---------------------------------------------------------------------------
# RAIS priority: interactive requests drain before background ones
# ---------------------------------------------------------------------------

def run_rais(model, tokenizer, requests: list[Request]) -> list[Request]:
    """RAIS-style priority queue: interactive (priority=0) served first."""
    results = [None] * len(requests)
    # Min-heap: (priority, submit_time, req) — lower priority value = served first
    heap = []
    for req in requests:
        heapq.heappush(heap, (req.priority, req.submit_time, req))

    while heap:
        _, _, req = heapq.heappop(heap)
        t_start = req.submit_time
        t_first = None
        for resp in mlx_lm.stream_generate(model, tokenizer, req.prompt,
                                            max_tokens=MAX_TOKENS):
            if t_first is None:
                t_first = time.perf_counter()
        t_end = time.perf_counter()
        req.ttft_ms = (t_first - t_start) * 1000 if t_first else float("nan")
        req.e2e_ms  = (t_end  - t_start) * 1000
        results[req.client_id] = req

    return results


# ---------------------------------------------------------------------------
# Run one trial and return results
# ---------------------------------------------------------------------------

def make_requests(n_clients: int) -> list[Request]:
    """Background requests arrive first, then interactive ones trickle in.

    This is the realistic scenario: batch jobs are pre-queued, then a
    real-time user submits a request. RAIS jumps the interactive request
    ahead; naive FIFO makes the user wait behind all background work.
    """
    now = time.perf_counter()
    reqs = []
    n_bg   = n_clients // 2
    n_int  = n_clients - n_bg

    # Background requests submitted first (already in queue)
    for i in range(n_bg):
        reqs.append(Request(
            priority    = 1,
            submit_time = now,
            client_id   = i,
            prompt      = PROMPTS[i % len(PROMPTS)],
        ))
    # Interactive requests arrive slightly later (10 ms simulated delay)
    for i in range(n_int):
        reqs.append(Request(
            priority    = 0,
            submit_time = now + 0.010,   # 10ms after background batch
            client_id   = n_bg + i,
            prompt      = PROMPTS[(n_bg + i) % len(PROMPTS)],
        ))
    return reqs


def summarise(results: list[Request], label: str):
    interactive = [r for r in results if r.priority == 0]
    background  = [r for r in results if r.priority == 1]

    def avg(lst, attr):
        vals = [getattr(r, attr) for r in lst if getattr(r, attr) is not None]
        return sum(vals) / len(vals) if vals else float("nan")

    print(f"\n  [{label}]")
    print(f"    Interactive ({len(interactive)} req):  "
          f"TTFT={avg(interactive,'ttft_ms'):.0f}ms  "
          f"E2E={avg(interactive,'e2e_ms'):.0f}ms")
    print(f"    Background  ({len(background)} req):   "
          f"TTFT={avg(background,'ttft_ms'):.0f}ms  "
          f"E2E={avg(background,'e2e_ms'):.0f}ms")
    return (avg(interactive, "ttft_ms"), avg(background, "ttft_ms"),
            avg(interactive, "e2e_ms"),  avg(background, "e2e_ms"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   default="llama1b",
                        choices=list(LOCAL_DIRS.keys()))
    parser.add_argument("--clients", type=int, default=6)
    args = parser.parse_args()

    model_dir = LOCAL_DIRS[args.model]
    if not model_dir.exists():
        print(f"Model not found at {model_dir}. Run fetch_model.py first.")
        return

    print(f"Loading {args.model} from {model_dir}...")
    model, tokenizer = mlx_lm.load(str(model_dir))
    mx.eval(model.parameters())
    print(f"Model loaded. Running {args.clients} concurrent clients "
          f"({args.clients//2} interactive, {args.clients - args.clients//2} background).\n")

    # Warmup
    for _ in range(WARMUP_RUNS):
        run_naive(model, tokenizer, make_requests(args.clients))
        run_rais(model, tokenizer,  make_requests(args.clients))

    # Measured
    n_runs = 3
    results_tsv = Path("experiments/mlx_concurrent_results.tsv")
    with open(results_tsv, "w") as f:
        f.write("approach\tclass\tttft_ms\te2e_ms\n")

    naive_rows, rais_rows = [], []
    for r in range(n_runs):
        print(f"Run {r+1}/{n_runs}...")
        n_res = run_naive(model, tokenizer, make_requests(args.clients))
        p_res = run_rais(model, tokenizer,  make_requests(args.clients))
        naive_rows.append(summarise(n_res, "Naive FIFO"))
        rais_rows.append(summarise(p_res, "RAIS Priority"))

    def avg_rows(rows):
        return tuple(sum(r[i] for r in rows) / len(rows) for i in range(4))

    naive = avg_rows(naive_rows)
    rais  = avg_rows(rais_rows)

    print(f"\n{'='*50}")
    print(f"AVERAGED OVER {n_runs} RUNS — {args.model}, {args.clients} clients")
    print(f"{'='*50}")
    print(f"{'':25} {'Naive FIFO':>14} {'RAIS Priority':>14} {'Speedup':>10}")
    print(f"  Interactive TTFT (ms)  {naive[0]:>14.0f} {rais[0]:>14.0f} {naive[0]/rais[0]:>9.2f}x")
    print(f"  Background  TTFT (ms)  {naive[1]:>14.0f} {rais[1]:>14.0f} {naive[1]/rais[1]:>9.2f}x")
    print(f"  Interactive E2E  (ms)  {naive[2]:>14.0f} {rais[2]:>14.0f} {naive[2]/rais[2]:>9.2f}x")
    print(f"  Background  E2E  (ms)  {naive[3]:>14.0f} {rais[3]:>14.0f} {naive[3]/rais[3]:>9.2f}x")

    with open(results_tsv, "a") as f:
        f.write(f"naive\tinteractive\t{naive[0]:.1f}\t{naive[2]:.1f}\n")
        f.write(f"naive\tbackground\t{naive[1]:.1f}\t{naive[3]:.1f}\n")
        f.write(f"rais\tinteractive\t{rais[0]:.1f}\t{rais[2]:.1f}\n")
        f.write(f"rais\tbackground\t{rais[1]:.1f}\t{rais[3]:.1f}\n")

    print(f"\nResults written to {results_tsv}")
    import subprocess
    subprocess.run(["python3", "experiments/plot_mlx_concurrent.py", str(results_tsv)])


if __name__ == "__main__":
    main()
