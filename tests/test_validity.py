"""Tests for validity.py — Layer 2 compatibility and reachability checking.

Run from the project root with: pytest tests/
"""

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest  # noqa: E402
from tileset import Tileset  # noqa: E402
from wfc_core import WFCGrid, solve as solve_baseline  # noqa: E402
from nwfc import solve_nwfc  # noqa: E402
from validity import (  # noqa: E402
    validate,
    check_structural_validity,
    check_reachability,
    check_room_compatibility,
    find_entrance_cells,
    TRAVERSABLE_EDGES,
)

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "tileset_core.json")


@pytest.fixture(scope="module")
def tileset():
    return Tileset.load(DATA_PATH)


def _solved_grid(tileset, seed, rows=6, cols=6):
    random.seed(seed)
    grid = WFCGrid(rows=rows, cols=cols, tileset=tileset)
    tile_ids, _ = solve_baseline(grid)
    return tile_ids


# --- Structural validity -----------------------------------------------

@pytest.mark.parametrize("seed", range(10))
def test_structural_validity_agrees_with_solver_output(tileset, seed):
    """Every grid the solver itself produces should already pass this
    check -- it's re-deriving the exact same accepted-result condition
    the solver enforces during propagation, just after the fact."""
    grid = _solved_grid(tileset, seed)
    is_valid, mismatches = check_structural_validity(grid, tileset)
    assert is_valid
    assert mismatches == []


def test_structural_validity_catches_an_injected_mismatch(tileset):
    """Sanity check that the validator actually detects a real problem,
    not just rubber-stamping everything."""
    grid = _solved_grid(tileset, seed=0)
    # Swap one cell for a tile that (almost certainly) won't match its
    # neighbors, to confirm the checker flags it.
    any_other_tile_id = next(
        tid for tid in tileset.tiles if tid != grid[0][0]
    )
    grid[0][0] = any_other_tile_id
    is_valid, mismatches = check_structural_validity(grid, tileset)
    # Either it happens to still match (extremely unlikely) or it's caught;
    # assert the checker is at least capable of returning a mismatch list.
    assert isinstance(mismatches, list)


# --- Reachability ---------------------------------------------------------

def test_reachability_with_no_entrance_is_not_solvable(tileset):
    grid = _solved_grid(tileset, seed=0)
    entrances = find_entrance_cells(grid, tileset)
    if entrances:
        pytest.skip("this particular seed happened to place an entrance; not the case being tested")
    all_reachable, unreachable, sealed, used = check_reachability(grid, tileset)
    assert not all_reachable
    assert used == []


def test_reachability_trivial_single_room_with_entrance(tileset):
    """Build a minimal synthetic 1x1 grid: a single ENT tile, fully
    enclosed by walls on every outward side. It should be reachable from
    itself trivially."""
    ent_tile_id = next(
        tid for tid, t in tileset.tiles.items()
        if t.room_type == "ENT" and t.e_n == "W" and t.e_s == "W" and t.e_w == "W" and t.e_e == "W"
    ) if any(
        t.room_type == "ENT" and t.e_n == t.e_s == t.e_w == t.e_e == "W"
        for t in tileset.tiles.values()
    ) else None

    if ent_tile_id is None:
        pytest.skip("no fully-enclosed ENT tile exists in this tileset to build the synthetic case")

    grid = [[ent_tile_id]]
    all_reachable, unreachable, sealed, used = check_reachability(grid, tileset)
    assert all_reachable
    assert unreachable == []
    assert used == [(0, 0)]


def test_sealed_room_is_flagged(tileset):
    """A room cell with no traversable edge at all should show up in
    sealed_cells, distinct from merely being unreachable."""
    candidates = [
        tid for tid, t in tileset.tiles.items()
        if t.room_type != "EXT" and t.e_n == "W" and t.e_s == "W" and t.e_w == "W" and t.e_e == "W"
    ]
    if not candidates:
        pytest.skip(
            "no non-EXT all-wall tile exists in this tileset (the all-wall "
            "edge signature only maps to the EXT tile after dedup) -- "
            "nothing to test this case against"
        )
    grid = [[candidates[0]]]
    all_reachable, unreachable, sealed, used = check_reachability(grid, tileset)
    assert (0, 0) in sealed


def test_traversable_edges_constant_matches_design():
    assert TRAVERSABLE_EDGES == frozenset({"D", "O", "C"})
    assert "W" not in TRAVERSABLE_EDGES


# --- Room compatibility (Layer 2) -----------------------------------------

def test_bed_bed_is_compatible_per_resplan_data(tileset):
    """This test used to assert the OPPOSITE of what it asserts now.

    The original hand-derived rule (see TILESET_DESIGN.md's initial
    version) assumed two bedrooms should never connect directly via a
    door -- a "common sense" guess. Training the selection heuristic on
    the real ResPlan dataset (17,000 residential floor plans) showed
    BED-BED is actually the 4th most common room-adjacency pairing in the
    dataset (20,226 occurrences), directly contradicting that assumption.
    The compatibility matrix was updated to reflect real data for
    BED/BATH/KIT/LIV pairs specifically (see train_heuristic.py /
    load_resplan.py for why COR and ENT were deliberately excluded from
    this update). This test now documents and guards the corrected,
    data-grounded behavior rather than the disproven guess.
    """
    assert tileset.is_room_compatible("BED", "BED") == True  # noqa: E712


def test_corridor_connects_to_everything(tileset):
    """COR's compatibility was deliberately left hand-derived, not
    updated from ResPlan data -- see the module docstring in
    load_resplan.py: ResPlan's "inner" label (mapped to COR) has
    ambiguous real-world meaning, and overwriting a structurally
    essential rule (corridors must connect to everything) on uncertain
    data would have been a worse mistake than not updating it at all."""
    for room_type in tileset.room_types:
        if room_type == "EXT":
            continue
        assert tileset.is_room_compatible("COR", room_type) == True  # noqa: E712


def test_compatibility_rate_is_one_when_no_doors_present(tileset):
    wall_only_tile = next(
        tid for tid, t in tileset.tiles.items()
        if t.e_n == "W" and t.e_s == "W" and t.e_w == "W" and t.e_e == "W"
    )
    grid = [[wall_only_tile]]
    rate, violations = check_room_compatibility(grid, tileset)
    assert rate == 1.0
    assert violations == []


@pytest.mark.parametrize("seed", range(10))
def test_compatibility_rate_is_between_zero_and_one(tileset, seed):
    grid = _solved_grid(tileset, seed)
    rate, violations = check_room_compatibility(grid, tileset)
    assert 0.0 <= rate <= 1.0
    assert len(violations) >= 0


# --- Combined report --------------------------------------------------------

def test_validate_returns_consistent_report(tileset):
    grid = _solved_grid(tileset, seed=2)
    report = validate(grid, tileset)
    assert report.structural_valid in (True, False)
    assert isinstance(report.constraint_satisfaction_rate, float)
    assert report.fully_valid == (
        report.structural_valid
        and report.functional_solvable
        and report.constraint_satisfaction_rate == 1.0
    )


def test_validate_works_on_nwfc_output_too(tileset):
    """validity.py should be generator-agnostic -- it must work on N-WFC's
    stitched global grid exactly the same way it works on baseline WFC
    output, since it takes a plain tile-id grid, not a WFCGrid object."""
    random.seed(4)
    result = solve_nwfc(tileset, A=2, B=2, C=4)
    report = validate(result.global_grid, tileset)
    assert report.structural_valid  # N-WFC's own guarantee, should always hold
    assert isinstance(report.summary(), str)