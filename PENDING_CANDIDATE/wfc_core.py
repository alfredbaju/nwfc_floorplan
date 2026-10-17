"""Baseline Wave Function Collapse engine — Layer 1 (geometric) only.

This is deliberately *plain* WFC: one grid, solved as a single constraint
satisfaction problem, with no subgrids, no macro/micro split, and no
room-compatibility (Layer 2) checks. Per the project roadmap, this exists
to validate the tile vocabulary and edge-matching logic in isolation
*before* N-WFC (nwfc.py) is built on top of it.

Implements the WFC solver loop described in Section III-B of the base paper
(Nie, Zheng, Zhuang & Togelius, 2024):
  1. Initialization    — every cell holds the full candidate tileset
  2. Entropy selection  — pick the uncollapsed cell with fewest candidates
                          (minimum remaining value heuristic)
  3. Collapse           — randomly choose one tile for that cell
  4. Propagation        — remove incompatible candidates from neighbors
  5. Termination        — all cells collapsed (accepted result), or a
                          contradiction occurs and the grid restarts
                          (baseline WFC's textbook backtracking behavior —
                          Section III-B5). Avoiding this at the *exterior*
                          level is N-WFC's whole point; at this baseline
                          stage, restarts are expected and fine.
"""

import random
from collections import deque

DIRS = {
    "n": (-1, 0),
    "s": (1, 0),
    "w": (0, -1),
    "e": (0, 1),
}
OPPOSITE = {"n": "s", "s": "n", "w": "e", "e": "w"}

# Edge codes a tile may show on the outer perimeter of a grid (i.e. facing
# "outside the building"). Matches the W_EXT convention introduced when
# annotating reference floor plans.
DEFAULT_BOUNDARY_EDGES = frozenset({"W", "W_EXT"})


class Contradiction(Exception):
    """Raised when a cell's candidate set is reduced to empty during
    propagation — i.e. no tile can satisfy the accumulated constraints."""


class WFCGrid:
    def __init__(
        self,
        rows,
        cols,
        tileset,
        boundary_edges=DEFAULT_BOUNDARY_EDGES,
        boundary_sides=("n", "s", "w", "e"),
        enforce_layer2=False,
        context_scorer=None,
        candidate_tile_ids=None,
        south_neighbor_types=None,
        east_neighbor_types=None,
    ):
        """boundary_sides controls which of the grid's four outer sides get
        the wall-only perimeter constraint applied. The baseline (single
        full-grid) solver always uses all four — this only matters once
        I-WFC starts solving C x C subgrids that sit *inside* a larger
        N-WFC grid, where an edge facing a not-yet-generated neighboring
        subgrid must stay unconstrained rather than forced to a wall.

        enforce_layer2: if True, propagation also filters out door (D)
        connections whose two room types aren't compatible per the
        tileset's Layer 2 matrix, not just geometrically matching edges.
        Default False preserves the original Layer-1-only baseline
        behavior documented in this module's docstring.

        context_scorer: optional heuristic_model.ContextScorer instance.
        If given, collapse() weights each candidate not just by its
        static Tile.weight but also by how plausible the trained model
        finds that room type given the room types already placed in
        neighboring cells -- the learned selection heuristic, as opposed
        to a fixed per-tile frequency hint. None (default) preserves the
        original static-weight-only behavior.

        candidate_tile_ids: optional iterable restricting which tile ids
        this grid's (non-pinned) cells may initially choose from -- used
        by zoning.py to restrict a whole subgrid to one room type (e.g.
        "this subgrid is a kitchen") for exact room-count control. The
        full tileset is still used for all edge/room_type lookups, so
        pinned cells inherited from a neighboring subgrid of a *different*
        restricted room type still resolve correctly; only the initial
        candidate pool is narrowed. None (default) uses every tile in the
        tileset, unchanged from the original behavior."""
        self.rows = rows
        self.cols = cols
        self.tileset = tileset
        self.boundary_edges = set(boundary_edges)
        self.boundary_sides = set(boundary_sides)
        self.enforce_layer2 = enforce_layer2
        # Optional: room types planned for the not-yet-generated subgrids
        # that this grid's last row / last column will be shared with.
        # When given (with enforce_layer2), tiles in that shared row/col
        # are not allowed a DOOR facing a neighbour whose type can't
        # legally take one -- otherwise the neighbour's solve starts from
        # a contradictory pinned edge and the exterior solver has to
        # backtrack (verified to be the dominant cause of floor
        # generation failures under layer-2 rules).
        self.south_neighbor_types = frozenset(south_neighbor_types) if south_neighbor_types else None
        self.east_neighbor_types = frozenset(east_neighbor_types) if east_neighbor_types else None
        self.context_scorer = context_scorer
        if candidate_tile_ids is not None:
            self._all_ids = frozenset(candidate_tile_ids)
        else:
            self._all_ids = frozenset(tileset.tiles.keys())
        self.cells = None
        self.reset()

    def reset(self):
        """Return every cell to its full initial candidate set, then apply
        the perimeter (boundary) constraint on the configured sides."""
        self.cells = [[set(self._all_ids) for _ in range(self.cols)] for _ in range(self.rows)]
        self._apply_boundary_constraints()

    def in_bounds(self, r, c):
        return 0 <= r < self.rows and 0 <= c < self.cols

    def _apply_boundary_constraints(self):
        """Cells on a configured outer side may only use tiles whose
        outward-facing edge is a boundary-compatible code (e.g. a wall),
        since there is no neighboring cell there to satisfy a door/open/
        corridor connection. Sides not in self.boundary_sides are left
        unconstrained (used for I-WFC subgrid edges facing a neighboring
        subgrid rather than the true exterior)."""
        for r in range(self.rows):
            for c in range(self.cols):
                candidates = self.cells[r][c]
                to_remove = set()
                for tid in candidates:
                    tile = self.tileset.tiles[tid]
                    if r == 0 and "n" in self.boundary_sides and tile.e_n not in self.boundary_edges:
                        to_remove.add(tid)
                    if r == self.rows - 1 and "s" in self.boundary_sides and tile.e_s not in self.boundary_edges:
                        to_remove.add(tid)
                    if c == 0 and "w" in self.boundary_sides and tile.e_w not in self.boundary_edges:
                        to_remove.add(tid)
                    if c == self.cols - 1 and "e" in self.boundary_sides and tile.e_e not in self.boundary_edges:
                        to_remove.add(tid)
                    if self.enforce_layer2:
                        if (r == self.rows - 1 and self.south_neighbor_types and tile.e_s == "D"
                                and not self._door_ok(tile.room_type, self.south_neighbor_types)):
                            to_remove.add(tid)
                        if (c == self.cols - 1 and self.east_neighbor_types and tile.e_e == "D"
                                and not self._door_ok(tile.room_type, self.east_neighbor_types)):
                            to_remove.add(tid)
                candidates -= to_remove

    def _door_ok(self, room_type, neighbor_types):
        """A door from `room_type` is only safe if EVERY room type the
        neighbouring subgrid might use can legally share one."""
        is_ok = self.tileset.is_room_compatible
        return all(is_ok(room_type, nt) and is_ok(nt, room_type) for nt in neighbor_types)

    def pin(self, r, c, tile_id):
        """Force cell (r, c) to a specific tile id, bypassing entropy
        selection/random collapse. Used by I-WFC to inherit a fixed
        boundary row/column from an already-solved neighboring subgrid."""
        self.cells[r][c] = {tile_id}

    def snapshot(self):
        """Deep copy of all cell candidate sets, for backtracking search."""
        return [[set(cell) for cell in row] for row in self.cells]

    def restore(self, snap):
        """Restore cell candidate sets from a previous snapshot()."""
        self.cells = [[set(cell) for cell in row] for row in snap]

    def is_collapsed(self, r, c):
        return len(self.cells[r][c]) == 1

    def fully_collapsed(self):
        return all(
            self.is_collapsed(r, c) for r in range(self.rows) for c in range(self.cols)
        )

    def entropy_cell(self):
        """Return the (row, col) of the uncollapsed cell with the fewest
        remaining candidates (minimum remaining value). Ties are broken
        randomly, which is what keeps repeated runs from being identical
        (the aperiodicity property discussed in the base paper, Section
        IV-C, applied here at the single-grid level)."""
        best_n = None
        tied = []
        for r in range(self.rows):
            for c in range(self.cols):
                n = len(self.cells[r][c])
                if n <= 1:
                    continue
                if best_n is None or n < best_n:
                    best_n = n
                    tied = [(r, c)]
                elif n == best_n:
                    tied.append((r, c))
        return random.choice(tied) if tied else None

    def collapsed_neighbor_room_types(self, r, c):
        """Room types of this cell's already-collapsed (single-candidate)
        orthogonal neighbors. Used by the learned selection heuristic to
        score a candidate against its actual local context, rather than
        a fixed per-tile weight that ignores what's already been placed."""
        types = []
        for _, (dr, dc) in DIRS.items():
            nr, nc = r + dr, c + dc
            if not self.in_bounds(nr, nc):
                continue
            neighbor_candidates = self.cells[nr][nc]
            if len(neighbor_candidates) == 1:
                tid = next(iter(neighbor_candidates))
                types.append(self.tileset.tiles[tid].room_type)
        return types

    # Room types with real, trustworthy training signal from ResPlan (see
    # load_resplan.py's ROOM_TYPE_MAP docstring). COR ("inner" in ResPlan)
    # has ambiguous real-world meaning and ENT never appeared in the
    # training graphs at all -- the static compatibility matrix update
    # deliberately excluded both. The runtime scorer must apply the same
    # exclusion, or it ends up suppressing/boosting COR/ENT placements
    # based on a signal we've already decided not to trust.
    DATA_INFORMED_ROOM_TYPES = frozenset({"BED", "BATH", "KIT", "LIV"})

    def effective_weight(self, r, c, tile_id):
        """The weight collapse() actually uses for a candidate: the
        tile's static frequency-hint weight, optionally modulated by the
        learned context scorer's plausibility score given this cell's
        already-placed neighbors. When context_scorer is None, this is
        just the static weight -- unchanged from the original behavior.

        The scorer only ever modulates the weight when BOTH the candidate
        room type AND every already-placed neighbor room type are in
        DATA_INFORMED_ROOM_TYPES -- if COR, ENT, or EXT are involved
        anywhere, the static weight is used unmodified, consistent with
        excluding those types from the data-driven compatibility matrix
        update for the same reason (no trustworthy training signal)."""
        base = self.tileset.tiles[tile_id].weight
        if self.context_scorer is None:
            return base
        candidate_room_type = self.tileset.tiles[tile_id].room_type
        if candidate_room_type not in self.DATA_INFORMED_ROOM_TYPES:
            return base
        neighbor_types = self.collapsed_neighbor_room_types(r, c)
        if not neighbor_types:
            return base  # no placed neighbors yet -- nothing to score against
        if not all(nt in self.DATA_INFORMED_ROOM_TYPES for nt in neighbor_types):
            return base  # a COR/ENT/EXT neighbor is involved -- don't trust the scorer here
        score = self.context_scorer.predict(candidate_room_type, neighbor_types)
        # Additive floor keeps a low-scoring candidate selectable (small
        # nonzero probability) rather than making it impossible to ever
        # pick -- the heuristic should bias search, not hard-forbid moves
        # Layer 1/Layer 2 haven't already ruled out.
        return base * (score + 0.05)

    def collapse(self, r, c):
        """Collapse cell (r, c) to one of its remaining candidates, using
        each tile's effective weight (static weight, optionally modulated
        by the learned context scorer) as a frequency hint -- matching
        Gumin's original WFC design, where tile weights bias output
        distribution towards more common tiles rather than treating every
        candidate as equally likely. Without this, tiles with many
        geometrically-easy variants (e.g. wall-heavy filler) dominate
        output regardless of how semantically rare they should be."""
        candidates = sorted(self.cells[r][c])
        if not candidates:
            raise Contradiction(f"Cell ({r},{c}) has no candidates left to collapse")
        weights = [self.effective_weight(r, c, tid) for tid in candidates]
        chosen = random.choices(candidates, weights=weights, k=1)[0]
        self.cells[r][c] = {chosen}
        return chosen

    def propagate(self, start_r, start_c):
        """Constraint propagation via a BFS worklist: after collapsing a
        cell, remove any neighboring candidate tile whose facing edge
        wouldn't match, then continue outward from every cell that
        actually changed. Mirrors the base paper's accepted-result
        definition (Section III-B4): touching edges must be equal.

        When self.enforce_layer2 is True, a door (D) edge additionally
        requires the two tiles' room types to be compatible per the
        tileset's Layer 2 matrix -- not just geometrically matching edge
        codes. This is what turns "structurally valid" into "functionally
        sensible" *during* generation, rather than only measuring it
        afterward in validity.py.
        """
        queue = deque([(start_r, start_c)])
        while queue:
            r, c = queue.popleft()
            cell_candidates = self.cells[r][c]
            for direction, (dr, dc) in DIRS.items():
                nr, nc = r + dr, c + dc
                if not self.in_bounds(nr, nc):
                    continue
                neighbor_candidates = self.cells[nr][nc]
                if len(neighbor_candidates) <= 1:
                    continue  # already collapsed (or empty) — nothing to prune
                opp = OPPOSITE[direction]

                if not self.enforce_layer2:
                    allowed_neighbor_edges = {
                        self.tileset.tiles[tid].edge(direction) for tid in cell_candidates
                    }
                    to_remove = {
                        tid
                        for tid in neighbor_candidates
                        if self.tileset.tiles[tid].edge(opp) not in allowed_neighbor_edges
                    }
                else:
                    # Layer 2-aware: a neighbor candidate survives only if
                    # SOME candidate in this cell matches its edge code,
                    # AND (for door edges specifically) the two room types
                    # are compatible. O/W/C edges have no room-type
                    # semantics attached, so they're unaffected.
                    to_remove = set()
                    for ntid in neighbor_candidates:
                        ntile = self.tileset.tiles[ntid]
                        nedge = ntile.edge(opp)
                        compatible_found = False
                        for ctid in cell_candidates:
                            ctile = self.tileset.tiles[ctid]
                            if ctile.edge(direction) != nedge:
                                continue
                            if nedge == "D" and not self.tileset.is_room_compatible(
                                ctile.room_type, ntile.room_type
                            ):
                                continue
                            compatible_found = True
                            break
                        if not compatible_found:
                            to_remove.add(ntid)

                if to_remove:
                    neighbor_candidates -= to_remove
                    if not neighbor_candidates:
                        raise Contradiction(
                            f"Cell ({nr},{nc}) reduced to empty during propagation"
                        )
                    queue.append((nr, nc))

    def as_tile_ids(self):
        """Return the solved grid as a 2-D list of tile ids. Raises if any
        cell has not yet collapsed to exactly one candidate."""
        out = []
        for r in range(self.rows):
            row = []
            for c in range(self.cols):
                cell = self.cells[r][c]
                if len(cell) != 1:
                    raise RuntimeError(f"Cell ({r},{c}) is not collapsed yet")
                row.append(next(iter(cell)))
            out.append(row)
        return out


def solve(grid: WFCGrid, max_restarts: int = 50):
    """Run the WFC loop to an accepted result. On contradiction, the whole
    grid resets and tries again from scratch — standard baseline-WFC
    backtracking (Section III-B5 of the base paper). Returns
    (tile_id_grid, number_of_restarts_used).
    """
    for attempt in range(max_restarts):
        try:
            while True:
                cell = grid.entropy_cell()
                if cell is None:
                    break  # every cell has exactly one candidate: accepted result
                r, c = cell
                grid.collapse(r, c)
                grid.propagate(r, c)
            return grid.as_tile_ids(), attempt
        except Contradiction:
            grid.reset()
    raise RuntimeError(f"Failed to find an accepted solution after {max_restarts} restarts")


def validate_edge_consistency(grid: WFCGrid) -> bool:
    """Confirm the accepted-result condition holds: every pair of adjacent
    collapsed cells has matching touching edges. This is a sanity check on
    the solver's own output, not part of the solving process itself —
    useful as a regression test and as the basis for validity.py later."""
    tile_ids = grid.as_tile_ids()
    for r in range(grid.rows):
        for c in range(grid.cols):
            tile = grid.tileset.tiles[tile_ids[r][c]]
            for direction, (dr, dc) in DIRS.items():
                nr, nc = r + dr, c + dc
                if not grid.in_bounds(nr, nc):
                    continue
                neighbor = grid.tileset.tiles[tile_ids[nr][nc]]
                if tile.edge(direction) != neighbor.edge(OPPOSITE[direction]):
                    return False
    return True
