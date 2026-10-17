"""E3 -- Structural/functional solvability: N-WFC with vs. without a
subcomplete boundary tileset.

"With" = the current 90-tile tileset (verified 6/6 subcompleteness
conditions). "Without" = the original 16 curated tiles only, which was
this project's tileset before the subcompleteness gap was discovered and
fixed (see project history) -- it satisfies only 2 of the 4 base-paper
conditions and neither of the two bounded-grid corner extensions.
"""

import random
import csv

from common import DATA_PATH, result_path
from tileset import Tileset
from nwfc import solve_nwfc
from validity import validate

N_TRIALS = 20
A, B, C = 3, 3, 4

REDUCED_PATH = DATA_PATH.replace("tileset_core.json", "tileset_reduced_nonsubcomplete.json")


def run_condition(tileset_path, label, enforce_layer2):
    tileset = Tileset.load(tileset_path)
    successes = 0
    failures = 0
    exterior_backtracks_total = 0
    solvable_count = 0
    satisfaction_rates = []

    for trial in range(N_TRIALS):
        random.seed(3000 + trial)
        try:
            result = solve_nwfc(tileset, A=A, B=B, C=C, max_exterior_backtracks=50, enforce_layer2=enforce_layer2)
            successes += 1
            exterior_backtracks_total += result.exterior_backtracks
            report = validate(result.global_grid, tileset)
            solvable_count += report.functional_solvable
            satisfaction_rates.append(report.constraint_satisfaction_rate)
        except RuntimeError:
            failures += 1

    mean_satisfaction = sum(satisfaction_rates) / len(satisfaction_rates) if satisfaction_rates else float("nan")
    row = {
        "condition": label,
        "trials": N_TRIALS,
        "generation_successes": successes,
        "generation_failures": failures,
        "generation_success_rate": successes / N_TRIALS,
        "mean_exterior_backtracks": exterior_backtracks_total / N_TRIALS,
        "functional_solvability_rate": solvable_count / successes if successes else float("nan"),
        "mean_constraint_satisfaction": mean_satisfaction,
    }
    return row


def run():
    print(f"Generating {N_TRIALS} plans per condition at A={A},B={B},C={C}...\n")

    without_subcomplete = run_condition(REDUCED_PATH, "1. Without subcomplete tileset, no Layer 2 (16 tiles)", enforce_layer2=False)
    with_subcomplete_only = run_condition(DATA_PATH, "2. With subcomplete tileset, no Layer 2 (90 tiles)", enforce_layer2=False)
    with_both = run_condition(DATA_PATH, "3. With subcomplete tileset + Layer 2 (90 tiles)", enforce_layer2=True)

    rows_out = [without_subcomplete, with_subcomplete_only, with_both]
    for row in rows_out:
        print(f"{row['condition']}")
        print(f"  Generation success rate:      {row['generation_success_rate']:.1%} ({row['generation_successes']}/{row['trials']})")
        print(f"  Mean exterior backtracks/run: {row['mean_exterior_backtracks']:.2f}")
        print(f"  Functional solvability rate:  {row['functional_solvability_rate']:.1%}")
        print(f"  Mean constraint satisfaction: {row['mean_constraint_satisfaction']:.1%}")
        print()

    csv_path = result_path("e3_solvability.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"Saved: {csv_path}")

    try:
        import matplotlib.pyplot as plt
        import numpy as np
        labels = ["1. No subcomplete\nno Layer 2", "2. Subcomplete\nno Layer 2", "3. Subcomplete\n+ Layer 2"]
        metrics = {
            "Generation success rate": [r["generation_success_rate"] for r in rows_out],
            "Functional solvability": [r["functional_solvability_rate"] for r in rows_out],
            "Constraint satisfaction": [r["mean_constraint_satisfaction"] for r in rows_out],
        }
        x = np.arange(len(labels))
        width = 0.25
        fig, ax = plt.subplots(figsize=(9, 5.5))
        colors = ["#C0392B", "#E8871E", "#1E2761"]
        for i, (name, vals) in enumerate(metrics.items()):
            ax.bar(x + (i - 1) * width, vals, width, label=name, color=colors[i])
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Rate")
        ax.set_ylim(0, 1.05)
        ax.set_title("E3: Effect of Subcomplete Tileset and Layer 2 on Solvability")
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        png_path = result_path("e3_solvability.png")
        fig.savefig(png_path, dpi=150)
        print(f"Saved: {png_path}")
    except ImportError:
        print("matplotlib not available, skipping chart")

    return rows_out


if __name__ == "__main__":
    run()