"""Demo: run N-WFC (macro/micro split) and compare its backtracking
behavior against baseline single-grid WFC on an equivalent total size.

    python src/demo_nwfc.py
"""

import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from tileset import Tileset  # noqa: E402
from wfc_core import WFCGrid, solve as solve_baseline  # noqa: E402
from nwfc import solve_nwfc, validate_global_edge_consistency  # noqa: E402

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "tileset_core.json")


def render(tile_id_grid, tileset):
    lines = []
    for row in tile_id_grid:
        lines.append(" | ".join(f"{tileset.tiles[tid].room_type:<4}" for tid in row))
    return "\n".join(lines)


def main(seed=7, A=2, B=2, C=4):
    random.seed(seed)
    tileset = Tileset.load(DATA_PATH)

    M = A * (C - 1) + 1
    N = B * (C - 1) + 1
    print(f"N-WFC: {A}x{B} subgrids of size {C}x{C}  ->  stitched grid {M}x{N}\n")

    t0 = time.perf_counter()
    result = solve_nwfc(tileset, A=A, B=B, C=C)
    t_nwfc = time.perf_counter() - t0

    print("Stitched N-WFC grid (room types):")
    print(render(result.global_grid, tileset))
    print(f"\nPer-subgrid I-WFC restarts: {result.subgrid_restarts}")
    print(f"Total I-WFC-level restarts: {result.total_restarts}")
    print(f"Edge consistency across the full stitched grid: "
          f"{validate_global_edge_consistency(tileset, result)}")
    print(f"N-WFC wall-clock time: {t_nwfc:.4f}s")

    print(f"\n--- Baseline WFC on an equivalent {M}x{N} single grid ---")
    random.seed(seed)
    grid = WFCGrid(rows=M, cols=N, tileset=tileset)
    t0 = time.perf_counter()
    _, baseline_restarts = solve_baseline(grid)
    t_baseline = time.perf_counter() - t0
    print(f"Baseline WFC restarts: {baseline_restarts}")
    print(f"Baseline WFC wall-clock time: {t_baseline:.4f}s")


if __name__ == "__main__":
    main()