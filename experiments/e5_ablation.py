"""E5 -- Ablation study: isolating which part of N-WFC actually helps.

A 2x2 design crossing two independent factors:
  - Macro/micro split:  standard single-grid WFC  vs.  N-WFC (I-WFC + diagonal exterior)
  - Tileset:            reduced, non-subcomplete (16 tiles)  vs.  subcomplete (90 tiles)

All four conditions target the same effective grid size (10x10) and use
Layer 2 enforcement throughout, so the only things that vary are the two
factors under test. This directly answers the proposal's E5 question:
is it the macro/micro split, the subcomplete tileset, or both together
that drives solvability?
"""

import random
import csv

from common import DATA_PATH, result_path
from tileset import Tileset
from wfc_core import WFCGrid, solve as solve_baseline
from nwfc import solve_nwfc
from validity import validate

N_TRIALS = 20
GRID_SIZE = 10  # baseline WFC: 10x10; N-WFC: A=3,B=3,C=4 -> also 10x10
A, B, C = 3, 3, 4

REDUCED_PATH = DATA_PATH.replace("tileset_core.json", "tileset_reduced_nonsubcomplete.json")


def run_baseline_condition(tileset_path, label):
    tileset = Tileset.load(tileset_path)
    successes, failures = 0, 0
    solvable_count = 0
    satisfaction_rates = []

    for trial in range(N_TRIALS):
        random.seed(4000 + trial)
        try:
            grid = WFCGrid(rows=GRID_SIZE, cols=GRID_SIZE, tileset=tileset, enforce_layer2=True)
            tile_ids, _ = solve_baseline(grid, max_restarts=50)
            successes += 1
            report = validate(tile_ids, tileset)
            solvable_count += report.functional_solvable
            satisfaction_rates.append(report.constraint_satisfaction_rate)
        except RuntimeError:
            failures += 1

    return _summarize(label, successes, failures, solvable_count, satisfaction_rates)


def run_nwfc_condition(tileset_path, label):
    tileset = Tileset.load(tileset_path)
    successes, failures = 0, 0
    solvable_count = 0
    satisfaction_rates = []

    for trial in range(N_TRIALS):
        random.seed(4000 + trial)
        try:
            result = solve_nwfc(tileset, A=A, B=B, C=C, max_exterior_backtracks=50, enforce_layer2=True)
            successes += 1
            report = validate(result.global_grid, tileset)
            solvable_count += report.functional_solvable
            satisfaction_rates.append(report.constraint_satisfaction_rate)
        except RuntimeError:
            failures += 1

    return _summarize(label, successes, failures, solvable_count, satisfaction_rates)


def _summarize(label, successes, failures, solvable_count, satisfaction_rates):
    trials = successes + failures
    mean_sat = sum(satisfaction_rates) / len(satisfaction_rates) if satisfaction_rates else float("nan")
    return {
        "condition": label,
        "generation_success_rate": successes / trials if trials else float("nan"),
        "functional_solvability_rate": solvable_count / successes if successes else float("nan"),
        "mean_constraint_satisfaction": mean_sat,
    }


def run():
    print(f"Running 2x2 ablation, {N_TRIALS} trials per cell...\n")

    conditions = [
        ("A: Standard WFC, reduced tileset", run_baseline_condition, REDUCED_PATH),
        ("B: Standard WFC, subcomplete tileset", run_baseline_condition, DATA_PATH),
        ("C: N-WFC, reduced tileset", run_nwfc_condition, REDUCED_PATH),
        ("D: N-WFC, subcomplete tileset", run_nwfc_condition, DATA_PATH),
    ]

    rows_out = []
    for label, fn, path in conditions:
        row = fn(path, label)
        rows_out.append(row)
        print(f"{label}")
        print(f"  Generation success rate:      {row['generation_success_rate']:.1%}")
        print(f"  Functional solvability rate:  {row['functional_solvability_rate']:.1%}")
        print(f"  Mean constraint satisfaction: {row['mean_constraint_satisfaction']:.1%}")
        print()

    csv_path = result_path("e5_ablation.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"Saved: {csv_path}")

    try:
        import matplotlib.pyplot as plt
        import numpy as np
        labels = ["A: Standard WFC\nreduced tileset", "B: Standard WFC\nsubcomplete tileset",
                  "C: N-WFC\nreduced tileset", "D: N-WFC\nsubcomplete tileset"]
        gen_rates = [r["generation_success_rate"] for r in rows_out]
        sat_rates = [r["mean_constraint_satisfaction"] for r in rows_out]

        x = np.arange(len(labels))
        width = 0.35
        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.bar(x - width / 2, gen_rates, width, label="Generation success rate", color="#C0392B")
        ax.bar(x + width / 2, sat_rates, width, label="Constraint satisfaction rate", color="#1E2761")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("Rate")
        ax.set_ylim(0, 1.05)
        ax.set_title("E5: Ablation \u2014 Macro/Micro Split x Subcomplete Tileset")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        png_path = result_path("e5_ablation.png")
        fig.savefig(png_path, dpi=150)
        print(f"Saved: {png_path}")
    except ImportError:
        print("matplotlib not available, skipping chart")

    return rows_out


if __name__ == "__main__":
    run()