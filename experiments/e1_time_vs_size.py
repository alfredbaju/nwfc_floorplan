"""E1 -- Generation time vs. plan size: standard WFC vs. N-WFC.

Replicates the base paper's Section III-C5 / Fig. 4 methodology: for each
grid size, run multiple trials and record mean/variance of wall-clock time
to reach an accepted solution.
"""

import random
import time
import csv

from common import DATA_PATH, result_path
from tileset import Tileset
from wfc_core import WFCGrid, solve as solve_baseline
from nwfc import solve_nwfc

N_TRIALS = 5
# (rows, cols) for baseline WFC, and the equivalent (A, B, C) for N-WFC
# such that A*(C-1)+1 == rows and B*(C-1)+1 == cols.
SIZE_CONFIGS = [
    {"label": "5x5", "rows": 5, "cols": 5, "A": 2, "B": 2, "C": 3},
    {"label": "7x7", "rows": 7, "cols": 7, "A": 2, "B": 2, "C": 4},
    {"label": "10x10", "rows": 10, "cols": 10, "A": 3, "B": 3, "C": 4},
    {"label": "13x13", "rows": 13, "cols": 13, "A": 4, "B": 4, "C": 4},
    {"label": "16x16", "rows": 16, "cols": 16, "A": 5, "B": 5, "C": 4},
    {"label": "19x19", "rows": 19, "cols": 19, "A": 6, "B": 6, "C": 4},
    {"label": "25x25", "rows": 25, "cols": 25, "A": 8, "B": 8, "C": 4},
    {"label": "31x31", "rows": 31, "cols": 31, "A": 10, "B": 10, "C": 4},
    {"label": "37x37", "rows": 37, "cols": 37, "A": 12, "B": 12, "C": 4},
    {"label": "46x46", "rows": 46, "cols": 46, "A": 15, "B": 15, "C": 4},
]


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def stdev(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def run():
    tileset = Tileset.load(DATA_PATH)
    rows_out = []

    for cfg in SIZE_CONFIGS:
        baseline_times = []
        for trial in range(N_TRIALS):
            random.seed(1000 + trial)
            grid = WFCGrid(rows=cfg["rows"], cols=cfg["cols"], tileset=tileset)
            t0 = time.perf_counter()
            solve_baseline(grid)
            baseline_times.append(time.perf_counter() - t0)

        nwfc_times = []
        for trial in range(N_TRIALS):
            random.seed(1000 + trial)
            t0 = time.perf_counter()
            solve_nwfc(tileset, A=cfg["A"], B=cfg["B"], C=cfg["C"])
            nwfc_times.append(time.perf_counter() - t0)

        row = {
            "grid_size": cfg["label"],
            "cells": cfg["rows"] * cfg["cols"],
            "baseline_mean_s": mean(baseline_times),
            "baseline_std_s": stdev(baseline_times),
            "nwfc_mean_s": mean(nwfc_times),
            "nwfc_std_s": stdev(nwfc_times),
        }
        rows_out.append(row)
        print(f"{cfg['label']:>8} ({row['cells']:>3} cells) | "
              f"baseline: {row['baseline_mean_s']*1000:8.2f}ms +/- {row['baseline_std_s']*1000:6.2f}ms | "
              f"N-WFC: {row['nwfc_mean_s']*1000:8.2f}ms +/- {row['nwfc_std_s']*1000:6.2f}ms")

    csv_path = result_path("e1_time_vs_size.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"\nSaved: {csv_path}")

    try:
        import matplotlib.pyplot as plt
        cells = [r["cells"] for r in rows_out]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.errorbar(cells, [r["baseline_mean_s"] * 1000 for r in rows_out],
                    yerr=[r["baseline_std_s"] * 1000 for r in rows_out],
                    marker="o", label="Standard WFC", color="#C0392B")
        ax.errorbar(cells, [r["nwfc_mean_s"] * 1000 for r in rows_out],
                    yerr=[r["nwfc_std_s"] * 1000 for r in rows_out],
                    marker="o", label="N-WFC", color="#1E2761")
        ax.set_xlabel("Grid size (cells)")
        ax.set_ylabel("Generation time (ms)")
        ax.set_title("E1: Generation Time vs. Plan Size")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        png_path = result_path("e1_time_vs_size.png")
        fig.savefig(png_path, dpi=150)
        print(f"Saved: {png_path}")
    except ImportError:
        print("matplotlib not available, skipping chart")

    return rows_out


if __name__ == "__main__":
    run()