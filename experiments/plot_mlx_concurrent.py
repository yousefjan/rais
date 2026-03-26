#!/usr/bin/env python3
"""Plot concurrent inference benchmark: RAIS priority vs naive FIFO."""

import sys, csv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "experiments/mlx_concurrent_results.tsv"
    with open(path) as f:
        rows = {(r["approach"], r["class"]): r
                for r in csv.DictReader(f, delimiter="\t")}

    labels   = ["Interactive", "Background"]
    ttft_naive = [float(rows[("naive", c.lower())]["ttft_ms"]) for c in labels]
    ttft_rais  = [float(rows[("rais",  c.lower())]["ttft_ms"]) for c in labels]
    e2e_naive  = [float(rows[("naive", c.lower())]["e2e_ms"])  for c in labels]
    e2e_rais   = [float(rows[("rais",  c.lower())]["e2e_ms"])  for c in labels]

    x, w = np.arange(len(labels)), 0.35
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))

    def plot_pair(ax, naive, rais, ylabel, title):
        b1 = ax.bar(x - w/2, naive, w, label="Naive FIFO",     color="#d45d5d")
        b2 = ax.bar(x + w/2, rais,  w, label="RAIS Priority",  color="#4a90d9")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        for i in range(len(labels)):
            speedup = naive[i] / rais[i] if rais[i] > 0 else 1
            color = "#1a7a1a" if speedup > 1 else "#b03030"
            ax.annotate(f"{speedup:.2f}x",
                        xy=(x[i] + w/2, rais[i]),
                        xytext=(0, 6), textcoords="offset points",
                        ha="center", fontsize=9, fontweight="bold", color=color)

    plot_pair(ax1, ttft_naive, ttft_rais, "TTFT (ms)", "Time to First Token")
    plot_pair(ax2, e2e_naive,  e2e_rais,  "E2E (ms)",  "End-to-End Latency")

    fig.suptitle("Naive FIFO vs RAIS Priority Scheduling — Concurrent MLX Inference",
                 fontsize=13, fontweight="bold")
    note = ("Interactive = latency-sensitive (priority 0)  |  "
            "Background = batch (priority 1)  |  "
            "RAIS drains interactive queue before background")
    fig.text(0.5, 0.01, note, ha="center", fontsize=8, color="#555")
    fig.tight_layout(rect=[0, 0.04, 1, 0.93])

    out = path.rsplit(".", 1)[0] + ".png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")

if __name__ == "__main__":
    main()
