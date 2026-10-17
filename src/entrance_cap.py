"""Caps the number of entrance (ENT) cells in a generated grid to exactly
one -- a real house has one front door, not several.

This is a direct consequence of a fix made earlier in this project: ENT's
selection weight was boosted substantially (and given multiple
orientations) specifically because entrance cells were appearing too
*rarely* on smaller grids (see the tileset revision history). That fix
worked, but on larger grids (more cells = more chances to roll a
high-weight tile) it overshot into producing multiple entrances. Rather
than re-tuning the weight again (which would just move the problem to a
different grid size), this enforces the actual real-world constraint
directly: exactly one entrance, regardless of grid size.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from validity import find_entrance_cells  # noqa: E402


def _find_room_type_substitute(tileset, current_tile_id, preferred_types=("LIV", "COR"), strict=False):
    """Find a tile with identical edges to current_tile_id but a
    different, more common room_type -- used to convert a surplus
    entrance cell into an ordinary room without touching any of its
    already-satisfied edge constraints. Tries preferred_types in order
    first (LIV/COR are the most structurally flexible, hub-like room
    types), then falls back to any room type at all if neither has a
    matching tile -- UNLESS strict=True, in which case no fallback
    happens at all and None is returned instead.

    strict=True must be used whenever room counts are being controlled
    (see zoning.py): the unrestricted fallback can pick *any* room type
    with a matching edge pattern, including one of the very types whose
    count the caller is trying to keep exact -- silently turning a
    requested "2 bedrooms" into more than 2. Better to leave a rare
    surplus entrance cell unconverted than to corrupt that guarantee.
    """
    current = tileset.tiles[current_tile_id]
    target_edges = {d: current.edge(d) for d in ("n", "s", "w", "e")}

    for room_type in preferred_types:
        for tid, t in tileset.tiles.items():
            if t.room_type != room_type:
                continue
            if all(t.edge(d) == target_edges[d] for d in ("n", "s", "w", "e")):
                return tid

    if strict:
        return None

    for tid, t in tileset.tiles.items():
        if t.room_type in ("ENT", "EXT"):
            continue
        if all(t.edge(d) == target_edges[d] for d in ("n", "s", "w", "e")):
            return tid

    return None


def cap_entrance_count(tile_id_grid, tileset, max_entrances=1, preferred_types=("LIV", "COR"),
                        strict=False, prefer_diverse_adjacency=(), verbose=False):
    """Convert all but `max_entrances` entrance cells into an ordinary
    room type, keeping the edge pattern (and therefore structural
    validity) intact. Returns (grid, conversions_made). Never mutates the
    input grid.

    strict=True (use whenever room counts are being controlled via
    zoning.py) restricts substitution to only `preferred_types`, never
    falling back to an arbitrary room type -- preserving exact room
    counts at the cost of occasionally leaving a rare surplus entrance
    unconverted. Pass preferred_types=("COR",) in that case specifically,
    since COR is the only type that's never itself under count control.

    prefer_diverse_adjacency: room types (e.g. ("LIV", "KIT") for a
    primary entrance near the living room plus a second one near the
    kitchen) -- when given, the kept entrances are chosen to cover as
    many of these as possible (one entrance adjacent to LIV, a different
    one adjacent to KIT), rather than blindly keeping the first
    `max_entrances` found in scan order. Without this, a second entrance
    reserved near the kitchen at the macro-planning stage could easily
    lose out to two entrances that both happen to scan-order before it,
    silently discarding the kitchen-side entrance every time raw
    generation produces more candidate entrance cells than requested
    (which is normal and expected before capping runs).
    """
    grid = [row[:] for row in tile_id_grid]
    entrances = find_entrance_cells(grid, tileset)

    if len(entrances) <= max_entrances:
        return grid, 0

    rows, cols = len(grid), len(grid[0])

    def is_adjacent_to(pos, room_type):
        r, c = pos
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and tileset.tiles[grid[nr][nc]].room_type == room_type:
                return True
        return False

    def is_border_cell(pos):
        r, c = pos
        return r == 0 or r == rows - 1 or c == 0 or c == cols - 1

    keep = []
    if prefer_diverse_adjacency:
        remaining = list(entrances)
        for target_type in prefer_diverse_adjacency:
            if len(keep) >= max_entrances:
                break
            # Prefer a candidate that's BOTH border-adjacent AND
            # touching the target type -- a non-border candidate would
            # just need relocating afterward (see ensure_entrance_on_
            # border), and at that point a border+target-adjacent cell
            # often doesn't exist anymore, silently losing the
            # adjacency this selection was trying to preserve in the
            # first place.
            match = next(
                (pos for pos in remaining if is_adjacent_to(pos, target_type) and is_border_cell(pos)),
                None,
            )
            if match is None:
                match = next((pos for pos in remaining if is_adjacent_to(pos, target_type)), None)
            if match is not None:
                keep.append(match)
                remaining.remove(match)
        # Fill any still-empty slots with whatever's left, in scan order.
        for pos in entrances:
            if len(keep) >= max_entrances:
                break
            if pos not in keep:
                keep.append(pos)
    else:
        keep = list(entrances[:max_entrances])

    to_convert = [pos for pos in entrances if pos not in keep]
    conversions = 0
    for (r, c) in to_convert:
        substitute = _find_room_type_substitute(tileset, grid[r][c], preferred_types, strict=strict)
        if substitute is None:
            if verbose:
                print(f"  No safe substitute found for surplus entrance at ({r},{c}); left as-is.")
            continue
        grid[r][c] = substitute
        conversions += 1
        if verbose:
            new_type = tileset.tiles[substitute].room_type
            print(f"  Converted surplus entrance at ({r},{c}) -> {new_type}")

    return grid, conversions