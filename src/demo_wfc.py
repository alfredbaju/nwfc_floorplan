"""Quick demo: solve a small grid with the baseline WFC engine and print it
as a readable room-type grid. Run from the project root:

    python src/demo_wfc.py
"""

import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))

from tileset import Tileset  # noqa: E402
from wfc_core import WFCGrid, solve, validate_edge_consistency  # noqa: E402

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "tileset_core.json")


def render(tile_id_grid, tileset):
    lines = []
    for row in tile_id_grid:
        lines.append(" | ".join(f"{tileset.tiles[tid].room_type:<4}" for tid in row))
    return "\n".join(lines)


def main(seed=None, rows=4, cols=4):
    if seed is not None:
        random.seed(seed)

    tileset = Tileset.load(DATA_PATH)
    grid = WFCGrid(rows=rows, cols=cols, tileset=tileset)

    tile_id_grid, restarts = solve(grid)

    print(f"Solved a {rows}x{cols} grid after {restarts} restart(s)\n")
    print("Room types:")
    print(render(tile_id_grid, tileset))
    print("\nTile ids:")
    for row in tile_id_grid:
        print(" | ".join(f"{tid:<4}" for tid in row))
    print("\nEdge consistency check passed:", validate_edge_consistency(grid))


if __name__ == "__main__":
    main(seed=7)