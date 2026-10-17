"""Layer 2 (functional) validation: room-type compatibility checking and
exit-reachability analysis for a generated floor plan.

Deliberately works on a plain 2-D tile-id grid plus a Tileset — not on
WFCGrid or NWFCResult objects directly — so the same functions validate
output from wfc_core.py (baseline), nwfc.py (N-WFC), or any future variant,
without coupling this module to a particular generator's internals. This
also makes it the natural tool for the E5 ablation study: run it against
grids produced with and without the subcomplete tileset / Layer 2 filtering
and compare the three metrics below.

Implements the three distinct validity metrics from the project proposal
(Section 5.6 / 1.3):

  - structural_valid       — the base paper's accepted-result condition:
                              every pair of adjacent cells has matching
                              touching edges (Layer 1 only).
  - functional_solvable    — every room cell is reachable from at least
                              one entrance cell, and no room is fully
                              sealed (the floor-plan analogue of the base
                              paper's Infinite Mario traversability check).
  - constraint_satisfaction_rate — fraction of door (D) connections whose
                              two room types are actually compatible per
                              the Layer 2 matrix (1.0 = zero violations).

Layer 2 is NOT currently enforced during generation (wfc_core.py and
nwfc.py are Layer 1 only, by design — see the project roadmap). This
module is therefore both the project's solvability checker *and* the
measurement instrument for how much Layer 2 matters, which is exactly
what the E3/E5 experiments need.
"""

import sys
import os
from collections import deque

sys.path.insert(0, os.path.dirname(__file__))

from wfc_core import DIRS, OPPOSITE  # noqa: E402

# Edge types a person can physically walk through. Matches TILESET_DESIGN.md
# section 6: doors, same-room openings, and corridor connections are all
# traversable; walls are not.
TRAVERSABLE_EDGES = frozenset({"D", "O", "C"})


class ValidityReport:
    def __init__(
        self,
        structural_valid,
        edge_mismatches,
        functional_solvable,
        unreachable_cells,
        entrance_cells,
        sealed_cells,
        constraint_satisfaction_rate,
        compatibility_violations,
    ):
        self.structural_valid = structural_valid
        self.edge_mismatches = edge_mismatches  # list of ((r,c), direction) pairs
        self.functional_solvable = functional_solvable
        self.unreachable_cells = unreachable_cells  # list of (r, c)
        self.entrance_cells = entrance_cells  # list of (r, c)
        self.sealed_cells = sealed_cells  # list of (r, c) with zero traversable edges
        self.constraint_satisfaction_rate = constraint_satisfaction_rate  # float in [0, 1]
        self.compatibility_violations = compatibility_violations  # list of ((r,c),(r,c),room_a,room_b)

    @property
    def fully_valid(self):
        """True only if the plan is structurally sound, functionally
        solvable, AND has zero Layer 2 compatibility violations. Individual
        metrics are still available separately — most experiments (E3
        especially) will want those, not just this all-or-nothing flag."""
        return (
            self.structural_valid
            and self.functional_solvable
            and self.constraint_satisfaction_rate == 1.0
        )

    def summary(self):
        lines = [
            f"Structural validity:            {'PASS' if self.structural_valid else 'FAIL'}"
            + (f" ({len(self.edge_mismatches)} mismatches)" if self.edge_mismatches else ""),
            f"Functional solvability:         {'PASS' if self.functional_solvable else 'FAIL'}"
            + (f" ({len(self.unreachable_cells)} unreachable, {len(self.sealed_cells)} sealed)"
               if not self.functional_solvable else ""),
            f"Constraint satisfaction rate:    {self.constraint_satisfaction_rate:.2%}"
            + (f" ({len(self.compatibility_violations)} violations)"
               if self.compatibility_violations else ""),
            f"Entrance cell(s) found:          {len(self.entrance_cells)}",
        ]
        return "\n".join(lines)


def _in_bounds(grid, r, c):
    return 0 <= r < len(grid) and 0 <= c < len(grid[0])


def check_structural_validity(tile_id_grid, tileset):
    """Layer 1 accepted-result check (Section III-B4 of the base paper):
    every pair of adjacent cells must have matching touching edges.
    Returns (is_valid, list_of_mismatches)."""
    mismatches = []
    rows, cols = len(tile_id_grid), len(tile_id_grid[0])
    for r in range(rows):
        for c in range(cols):
            tile = tileset.tiles[tile_id_grid[r][c]]
            for direction, (dr, dc) in DIRS.items():
                nr, nc = r + dr, c + dc
                if not _in_bounds(tile_id_grid, nr, nc):
                    continue
                neighbor = tileset.tiles[tile_id_grid[nr][nc]]
                if tile.edge(direction) != neighbor.edge(OPPOSITE[direction]):
                    mismatches.append(((r, c), direction))
    return (len(mismatches) == 0), mismatches


def find_entrance_cells(tile_id_grid, tileset, entrance_room_type="ENT"):
    rows, cols = len(tile_id_grid), len(tile_id_grid[0])
    return [
        (r, c)
        for r in range(rows)
        for c in range(cols)
        if tileset.tiles[tile_id_grid[r][c]].room_type == entrance_room_type
    ]


def check_reachability(tile_id_grid, tileset, entrance_cells=None, exterior_room_type="EXT"):
    """BFS from every entrance cell over traversable (D/O/C) edges. Returns
    (all_reachable, unreachable_cells, sealed_cells, entrance_cells_used).

    Cells whose room_type is the exterior placeholder are excluded from the
    "must be reachable" requirement — they represent the void outside the
    building footprint, not an actual room, mirroring how the base paper's
    Infinite Mario connectivity check only cares about traversable level
    geometry, not decorative background tiles.
    """
    rows, cols = len(tile_id_grid), len(tile_id_grid[0])

    if entrance_cells is None:
        entrance_cells = find_entrance_cells(tile_id_grid, tileset)

    room_cells = {
        (r, c)
        for r in range(rows)
        for c in range(cols)
        if tileset.tiles[tile_id_grid[r][c]].room_type != exterior_room_type
    }

    sealed_cells = []
    for (r, c) in room_cells:
        tile = tileset.tiles[tile_id_grid[r][c]]
        edges = {tile.e_n, tile.e_s, tile.e_w, tile.e_e}
        if not edges & TRAVERSABLE_EDGES:
            sealed_cells.append((r, c))

    if not entrance_cells:
        # No designated exit exists at all -- nothing can be "reachable
        # from an exit" by definition.
        return False, sorted(room_cells), sealed_cells, []

    visited = set()
    queue = deque(entrance_cells)
    visited.update(entrance_cells)
    while queue:
        r, c = queue.popleft()
        tile = tileset.tiles[tile_id_grid[r][c]]
        for direction, (dr, dc) in DIRS.items():
            if tile.edge(direction) not in TRAVERSABLE_EDGES:
                continue
            nr, nc = r + dr, c + dc
            if not _in_bounds(tile_id_grid, nr, nc):
                continue
            if (nr, nc) in visited:
                continue
            neighbor = tileset.tiles[tile_id_grid[nr][nc]]
            if neighbor.edge(OPPOSITE[direction]) not in TRAVERSABLE_EDGES:
                continue  # geometrically shouldn't happen if structurally valid, but guard anyway
            visited.add((nr, nc))
            queue.append((nr, nc))

    unreachable = sorted(room_cells - visited)
    all_reachable = len(unreachable) == 0
    return all_reachable, unreachable, sealed_cells, entrance_cells


def check_room_compatibility(tile_id_grid, tileset):
    """Layer 2 check: for every door (D) edge between two cells, confirm
    the two room types are allowed to connect per the compatibility
    matrix. Returns (satisfaction_rate, list_of_violations).

    Only D edges are checked -- O edges are, by the current tile
    vocabulary's design (TILESET_DESIGN.md section 2), meant to occur only
    within a single multi-cell room, so no cross-type check applies to
    them yet. Note: the Floor Plan 1 pilot annotation found a real
    counterexample (Kitchen connecting to Living via an open archway) —
    if that pattern holds up across more annotated plans, this function's
    scope should be widened to also check O-edge connections, not just D.
    """
    rows, cols = len(tile_id_grid), len(tile_id_grid[0])
    violations = []
    total_door_edges = 0
    checked_pairs = set()

    for r in range(rows):
        for c in range(cols):
            tile = tileset.tiles[tile_id_grid[r][c]]
            for direction, (dr, dc) in DIRS.items():
                if tile.edge(direction) != "D":
                    continue
                nr, nc = r + dr, c + dc
                if not _in_bounds(tile_id_grid, nr, nc):
                    continue
                pair_key = frozenset({(r, c), (nr, nc)})
                if pair_key in checked_pairs:
                    continue  # each door edge is shared by two cells; count once
                checked_pairs.add(pair_key)

                neighbor = tileset.tiles[tile_id_grid[nr][nc]]
                if neighbor.edge(OPPOSITE[direction]) != "D":
                    continue  # not actually a matched door pair (shouldn't happen if structurally valid)

                total_door_edges += 1
                if not tileset.is_room_compatible(tile.room_type, neighbor.room_type):
                    violations.append(((r, c), (nr, nc), tile.room_type, neighbor.room_type))

    if total_door_edges == 0:
        return 1.0, []  # vacuously satisfied -- no doors to violate anything
    rate = 1.0 - (len(violations) / total_door_edges)
    return rate, violations


def validate(tile_id_grid, tileset) -> ValidityReport:
    """Run all three validity checks and return a combined report."""
    structural_valid, edge_mismatches = check_structural_validity(tile_id_grid, tileset)
    entrance_cells = find_entrance_cells(tile_id_grid, tileset)
    functional_solvable, unreachable, sealed, entrances_used = check_reachability(
        tile_id_grid, tileset, entrance_cells=entrance_cells
    )
    rate, violations = check_room_compatibility(tile_id_grid, tileset)

    return ValidityReport(
        structural_valid=structural_valid,
        edge_mismatches=edge_mismatches,
        functional_solvable=functional_solvable,
        unreachable_cells=unreachable,
        entrance_cells=entrances_used,
        sealed_cells=sealed,
        constraint_satisfaction_rate=rate,
        compatibility_violations=violations,
    )