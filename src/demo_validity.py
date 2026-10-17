import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))

from tileset import Tileset
from wfc_core import WFCGrid, solve as solve_baseline
from nwfc import solve_nwfc
from validity import validate

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "tileset_core.json")

tileset = Tileset.load(DATA_PATH)

print("=== Baseline WFC, 6x6 ===")
random.seed(1)
grid = WFCGrid(rows=6, cols=6, tileset=tileset)
tile_ids, _ = solve_baseline(grid)
print(validate(tile_ids, tileset).summary())

print("\n=== N-WFC, 3x3 subgrids of 4x4 ===")
random.seed(1)
result = solve_nwfc(tileset, A=3, B=3, C=4)
print(validate(result.global_grid, tileset).summary())