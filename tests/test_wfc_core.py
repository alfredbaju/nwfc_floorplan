"""Regression tests for the baseline WFC engine (Layer 1 only).

Run from the project root with: pytest tests/
"""

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest  # noqa: E402
from tileset import Tileset  # noqa: E402
from wfc_core import WFCGrid, solve, validate_edge_consistency, Contradiction  # noqa: E402

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "tileset_core.json")


@pytest.fixture(scope="module")
def tileset():
    return Tileset.load(DATA_PATH)


@pytest.mark.parametrize("rows,cols", [(3, 3), (4, 4), (5, 5), (6, 6)])
def test_solve_produces_fully_collapsed_grid(tileset, rows, cols):
    grid = WFCGrid(rows=rows, cols=cols, tileset=tileset)
    tile_ids, _ = solve(grid)
    assert len(tile_ids) == rows
    assert all(len(row) == cols for row in tile_ids)
    assert grid.fully_collapsed()


@pytest.mark.parametrize("seed", range(10))
def test_solved_grid_passes_edge_consistency(tileset, seed):
    random.seed(seed)
    grid = WFCGrid(rows=5, cols=5, tileset=tileset)
    solve(grid)
    assert validate_edge_consistency(grid)


def test_boundary_cells_only_use_boundary_compatible_edges(tileset):
    grid = WFCGrid(rows=4, cols=4, tileset=tileset)
    tile_ids, _ = solve(grid)
    for c in range(grid.cols):
        top = tileset.tiles[tile_ids[0][c]]
        bottom = tileset.tiles[tile_ids[grid.rows - 1][c]]
        assert top.e_n in grid.boundary_edges
        assert bottom.e_s in grid.boundary_edges
    for r in range(grid.rows):
        left = tileset.tiles[tile_ids[r][0]]
        right = tileset.tiles[tile_ids[r][grid.cols - 1]]
        assert left.e_w in grid.boundary_edges
        assert right.e_e in grid.boundary_edges


def test_reset_restores_full_candidate_set(tileset):
    grid = WFCGrid(rows=3, cols=3, tileset=tileset)
    solve(grid)
    grid.reset()
    # after reset, no cell should be collapsed except where the boundary
    # constraint alone happens to leave a single candidate
    assert not grid.fully_collapsed()


def test_contradiction_is_raised_on_empty_candidate_set(tileset):
    grid = WFCGrid(rows=3, cols=3, tileset=tileset)
    grid.cells[1][1] = set()  # force an empty candidate set
    with pytest.raises(Contradiction):
        grid.propagate(1, 1)


def test_solve_is_deterministic_given_a_fixed_seed(tileset):
    random.seed(42)
    grid_a = WFCGrid(rows=4, cols=4, tileset=tileset)
    result_a, _ = solve(grid_a)

    random.seed(42)
    grid_b = WFCGrid(rows=4, cols=4, tileset=tileset)
    result_b, _ = solve(grid_b)

    assert result_a == result_b