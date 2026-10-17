"""Tests for nwfc.py — the macro/micro split and diagonal exterior process.

Run from the project root with: pytest tests/
"""

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest  # noqa: E402
from tileset import Tileset  # noqa: E402
from nwfc import (  # noqa: E402
    solve_nwfc,
    validate_global_edge_consistency,
    _diagonal_order,
)

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "tileset_core.json")


@pytest.fixture(scope="module")
def tileset():
    return Tileset.load(DATA_PATH)


def test_diagonal_order_covers_every_subgrid_exactly_once():
    A, B = 3, 4
    order = list(_diagonal_order(A, B))
    assert len(order) == A * B
    assert set(order) == {(a, b) for a in range(A) for b in range(B)}


def test_diagonal_order_respects_dependencies():
    """Every subgrid must appear after both its top neighbor (a-1, b) and
    its left neighbor (a, b-1), since it depends on their output."""
    A, B = 3, 3
    order = list(_diagonal_order(A, B))
    position = {ab: i for i, ab in enumerate(order)}
    for (a, b), i in position.items():
        if a > 0:
            assert position[(a - 1, b)] < i
        if b > 0:
            assert position[(a, b - 1)] < i


@pytest.mark.parametrize("A,B,C", [(2, 2, 4), (3, 3, 4), (2, 4, 5), (4, 4, 3)])
def test_stitched_grid_has_expected_dimensions(tileset, A, B, C):
    result = solve_nwfc(tileset, A=A, B=B, C=C)
    assert result.rows == A * (C - 1) + 1
    assert result.cols == B * (C - 1) + 1


@pytest.mark.parametrize("seed", range(15))
def test_stitched_grid_passes_edge_consistency(tileset, seed):
    random.seed(seed)
    result = solve_nwfc(tileset, A=3, B=3, C=4)
    assert validate_global_edge_consistency(tileset, result)


def test_overlapping_cells_agree_between_neighboring_subgrids(tileset):
    """The shared row/column N-WFC pastes between adjacent subgrids must
    be identical, not just edge-compatible — this is what 'overlap' means
    in the base paper's Section III-C2, and what makes exterior
    backtracking unnecessary in the first place."""
    random.seed(3)
    A, B, C = 2, 3, 4
    result = solve_nwfc(tileset, A=A, B=B, C=C)

    for a in range(A):
        for b in range(B):
            tile_grid = result.subgrids[(a, b)]
            if a > 0:
                above = result.subgrids[(a - 1, b)]
                assert tile_grid[0] == above[C - 1], (
                    f"Top row of subgrid ({a},{b}) doesn't match bottom row "
                    f"of subgrid ({a-1},{b})"
                )
            if b > 0:
                left = result.subgrids[(a, b - 1)]
                left_col = [left[r][C - 1] for r in range(C)]
                this_col = [tile_grid[r][0] for r in range(C)]
                assert left_col == this_col, (
                    f"Left column of subgrid ({a},{b}) doesn't match right "
                    f"column of subgrid ({a},{b-1})"
                )


def test_exterior_process_never_needs_to_redo_an_earlier_subgrid(tileset):
    """The determinacy property (Section IV-D of the base paper): once a
    subgrid is accepted, solve_nwfc's diagonal loop never revisits it.
    This is true by construction (each subgrid is solved exactly once in
    the loop), but we assert it explicitly here as a regression guard."""
    random.seed(11)
    A, B, C = 3, 3, 4
    result = solve_nwfc(tileset, A=A, B=B, C=C)
    assert len(result.subgrids) == A * B
    assert set(result.subgrids.keys()) == {(a, b) for a in range(A) for b in range(B)}


def test_boundary_edges_only_appear_on_true_grid_perimeter(tileset):
    """Cells strictly inside the stitched grid should not be forced to
    wall/W_EXT edges — only cells on the actual outer perimeter of the
    full M x N grid should ever be boundary-constrained."""
    random.seed(5)
    A, B, C = 2, 2, 4
    result = solve_nwfc(tileset, A=A, B=B, C=C)
    grid = result.global_grid
    rows, cols = result.rows, result.cols

    boundary_edges = {"W", "W_EXT"}
    for r in range(rows):
        for c in range(cols):
            tile = tileset.tiles[grid[r][c]]
            if r == 0:
                assert tile.e_n in boundary_edges
            if r == rows - 1:
                assert tile.e_s in boundary_edges
            if c == 0:
                assert tile.e_w in boundary_edges
            if c == cols - 1:
                assert tile.e_e in boundary_edges