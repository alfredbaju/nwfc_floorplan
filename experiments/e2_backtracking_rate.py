"""E2 -- Contradiction / backtracking rate: standard WFC vs. N-WFC.

For standard WFC, "backtracking" means a full-grid restart (see
wfc_core.solve). For N-WFC, we separate two distinct kinds, matching the
base paper's own distinction (Section III-C6): I-WFC-internal backtracking
(expected, and fine per the paper for a subcomplete tileset) vs. exterior
backtracking (redoing an already-accepted subgrid -- the thing subcomplete
tilesets are specifically supposed to make rare/unnecessary).
"""

import random
import csv

from common import DATA_PATH, result_path
from tileset import Tileset
from wfc_core import WFCGrid, solve as solve_baseline
from nwfc import solve_nwfc

N_TRIALS = 15
SIZE_CONFIGS = [
    {"label": "7x7", "rows": 7, "cols": 7, "A": 2, "B": 2, "C": 4},
    {"label": "10x10", "rows": 10, "cols": 10, "A": 3, "B": 3, "C": 4},
    {"label": "16x16", "rows": 16, "cols": 16, "A": 5, "B": 5, "C": 4},
    {"label": "25x25", "rows": 25, "cols": 25, "A": 8, "B": 8, "C": 4},
    {"label": "37x37", "rows": 37, "cols": 37, "A": 12, "B": 12, "C": 4},
]


def run():
    tileset = Tileset.load(DATA_PATH)
    rows_out = []

    for cfg in SIZE_CONFIGS:
        baseline_restarts = []
        for trial in range(N_TRIALS):
            random.seed(2000 + trial)
            grid = WFCGrid(rows=cfg["rows"], cols=cfg["cols"], tileset=tileset)
            _, restarts = solve_baseline(grid)
            baseline_restarts.append(restarts)

        nwfc_internal = []
        nwfc_exterior = []
        for trial in range(N_TRIALS):
            random.seed(2000 + trial)
            result = solve_nwfc(tileset, A=cfg["A"], B=cfg["B"], C=cfg["C"])
            nwfc_internal.append(result.total_restarts)
            nwfc_exterior.append(result.exterior_backtracks)

        row = {
            "grid_size": cfg["label"],
            "cells": cfg["rows"] * cfg["cols"],
            "baseline_restarts_total": sum(baseline_restarts),
            "baseline_restarts_mean": sum(baseline_restarts) / N_TRIALS,
            "nwfc_internal_backtracks_total": sum(nwfc_internal),
            "nwfc_internal_backtracks_mean": sum(nwfc_internal) / N_TRIALS,
            "nwfc_exterior_backtracks_total": sum(nwfc_exterior),
            "nwfc_exterior_backtracks_mean": sum(nwfc_exterior) / N_TRIALS,
        }
        rows_out.append(row)
        print(f"{cfg['label']:>8} | baseline restarts: {row['baseline_restarts_mean']:5.2f}/run | "
              f"N-WFC internal: {row['nwfc_internal_backtracks_mean']:6.2f}/run | "
              f"N-WFC EXTERIOR: {row['nwfc_exterior_backtracks_mean']:5.2f}/run")

    csv_path = result_path("e2_backtracking_rate.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"\nSaved: {csv_path}")

    total_exterior = sum(r["nwfc_exterior_backtracks_total"] for r in rows_out)
    total_nwfc_runs = len(SIZE_CONFIGS) * N_TRIALS
    print(f"\nExterior backtracking across all {total_nwfc_runs} N-WFC runs: "
          f"{total_exterior} total ({total_exterior/total_nwfc_runs:.3f} per run)")

    try:
        import matplotlib.pyplot as plt
        cells = [r["cells"] for r in rows_out]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(cells, [r["baseline_restarts_mean"] for r in rows_out], marker="o", label="Standard WFC (full restarts)", color="#C0392B")
        ax.plot(cells, [r["nwfc_internal_backtracks_mean"] for r in rows_out], marker="o", label="N-WFC (I-WFC internal backtracks)", color="#1E2761")
        ax.plot(cells, [r["nwfc_exterior_backtracks_mean"] for r in rows_out], marker="s", label="N-WFC (exterior backtracks)", color="#E8871E")
        ax.set_xlabel("Grid size (cells)")
        ax.set_ylabel("Mean backtracks per run")
        ax.set_title("E2: Backtracking Rate vs. Plan Size")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        png_path = result_path("e2_backtracking_rate.png")
        fig.savefig(png_path, dpi=150)
        print(f"Saved: {png_path}")
    except ImportError:
        print("matplotlib not available, skipping chart")

    return rows_out


if __name__ == "__main__":
    run()