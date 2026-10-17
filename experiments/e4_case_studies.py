"""E4 -- Qualitative case studies: generate several plans, render them, and
report each one's validity summary alongside the image. This is the direct
evidence base for visual/manual inspection, matching the base paper's own
approach of showing generated Mario/Carcassonne examples rather than only
reporting aggregate numbers.
"""

import random
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from common import DATA_PATH, RESULTS_DIR
from tileset import Tileset
from nwfc import solve_nwfc
from validity import validate
from visualize import render_floor_plan

N_CASES = 5
A, B, C = 3, 3, 4


def run():
    tileset = Tileset.load(DATA_PATH)
    case_dir = os.path.join(RESULTS_DIR, "e4_case_studies")
    os.makedirs(case_dir, exist_ok=True)

    report_lines = ["# E4 -- Qualitative Case Studies\n"]
    report_lines.append(f"{N_CASES} plans generated at A={A}, B={B}, C={C} "
                         f"(with subcomplete tileset + Layer 2 enforcement).\n")

    found = 0
    attempted = 0
    seed = 4000
    max_attempts = N_CASES * 4  # generous budget given E3 showed ~90% success rate

    while found < N_CASES and attempted < max_attempts:
        attempted += 1
        random.seed(seed)
        try:
            result = solve_nwfc(tileset, A=A, B=B, C=C, enforce_layer2=True)
        except RuntimeError as exc:
            report_lines.append(f"## Attempt (seed={seed}) -- GENERATION FAILED\n")
            report_lines.append(f"- {exc}\n")
            print(f"Seed {seed}: generation failed ({exc})")
            seed += 1
            continue

        found += 1
        report = validate(result.global_grid, tileset)

        img_path = os.path.join(case_dir, f"case_{found}_seed{seed}.png")
        render_floor_plan(
            result.global_grid, tileset, cell_size=1.0,
            title=f"Case {found} (seed={seed})", merge_rooms=True, save_path=img_path,
        )

        report_lines.append(f"## Case {found} (seed={seed})\n")
        report_lines.append(f"![Case {found}](e4_case_studies/case_{found}_seed{seed}.png)\n")
        report_lines.append(f"- Exterior backtracks: {result.exterior_backtracks}")
        report_lines.append(f"- Structural validity: {'PASS' if report.structural_valid else 'FAIL'}")
        report_lines.append(f"- Functional solvability: {'PASS' if report.functional_solvable else 'FAIL'} "
                             f"({len(report.unreachable_cells)} unreachable cells)")
        report_lines.append(f"- Constraint satisfaction: {report.constraint_satisfaction_rate:.1%}")
        report_lines.append(f"- Entrance cells found: {len(report.entrance_cells)}\n")

        print(f"Case {found}: structural={report.structural_valid}, "
              f"solvable={report.functional_solvable}, "
              f"satisfaction={report.constraint_satisfaction_rate:.1%}, "
              f"saved={img_path}")
        seed += 1

    report_lines.insert(2, f"Generation attempts: {attempted}, successful cases: {found} "
                            f"(empirical success rate this run: {found/attempted:.1%})\n")

    report_path = os.path.join(RESULTS_DIR, "e4_case_studies.md")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"\nSaved report: {report_path}")


if __name__ == "__main__":
    run()