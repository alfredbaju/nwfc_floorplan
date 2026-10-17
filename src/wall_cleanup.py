"""Post-generation wall cleanup.

O ("open") edges are meant to represent a connection within a single
multi-cell room instance, and C ("corridor") edges are meant to
represent an actual hallway connection. Neither restriction is enforced
during generation (an earlier attempt to enforce it live in
wfc_core.propagate cut generation success rates roughly in half -- see
project history), so a generated grid can end up with an O or C edge
connecting two *different*, non-corridor room types. Structurally this
is still valid (both sides' edges match), but visually it renders as a
faint dotted or dashed line where a person would expect a solid wall or
a clear door -- reading as a "gap."

This module finds and fixes those specific edges after generation,
converting them to a proper wall (or a door, if the two room types are
Layer 2 compatible and a door substitute exists) using the same
edge-preserving tile substitution technique as repair.py and
entrance_cap.py.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from wfc_core import DIRS, OPPOSITE  # noqa: E402


def _find_edge_substitute(tileset, current_tile_id, direction, target_edge, room_type=None):
    """Find a tile identical to current_tile_id on every edge except
    `direction` (set to target_edge instead), optionally forcing a
    specific room_type (defaults to keeping the same room_type). Mirrors
    repair.py's _find_door_substitute but generalized to any target edge
    value, not just 'D'."""
    current = tileset.tiles[current_tile_id]
    wanted_room_type = room_type or current.room_type
    target = {d: current.edge(d) for d in ("n", "s", "w", "e")}
    target[direction] = target_edge

    for tid, t in tileset.tiles.items():
        if t.room_type != wanted_room_type:
            continue
        if all(t.edge(d) == target[d] for d in ("n", "s", "w", "e")):
            return tid
    return None


def cleanup_cross_room_edges(tile_id_grid, tileset, verbose=False):
    """Scan every pair of adjacent, different-room-type cells and fix any
    O edge, or any C edge where neither side is a corridor, by converting
    it to a wall (or a door, if the room types are Layer 2 compatible and
    a door substitute is available) on both sides. Returns (grid, fixes).
    Never mutates the input grid.
    """
    grid = [row[:] for row in tile_id_grid]
    rows, cols = len(grid), len(grid[0])
    fixes = 0

    for r in range(rows):
        for c in range(cols):
            tile = tileset.tiles[grid[r][c]]
            for direction, (dr, dc) in DIRS.items():
                nr, nc = r + dr, c + dc
                if not (0 <= nr < rows and 0 <= nc < cols):
                    continue
                neighbor = tileset.tiles[grid[nr][nc]]
                if tile.room_type == neighbor.room_type:
                    continue

                edge = tile.edge(direction)
                needs_fix = (edge == "O") or (edge == "C" and "COR" not in (tile.room_type, neighbor.room_type))
                if not needs_fix:
                    continue

                # Only process each boundary once (from its lower-index side).
                if (nr, nc) < (r, c):
                    continue

                opp = OPPOSITE[direction]
                target_edge = "D" if tileset.is_room_compatible(tile.room_type, neighbor.room_type) else "W"

                new_self = _find_edge_substitute(tileset, grid[r][c], direction, target_edge)
                new_neighbor = _find_edge_substitute(tileset, grid[nr][nc], opp, target_edge)

                if new_self is None or new_neighbor is None:
                    # Fall back to a wall if a door substitute wasn't available.
                    if target_edge == "D":
                        new_self = _find_edge_substitute(tileset, grid[r][c], direction, "W")
                        new_neighbor = _find_edge_substitute(tileset, grid[nr][nc], opp, "W")

                if new_self is None or new_neighbor is None:
                    if verbose:
                        print(f"  No substitute found to fix edge at ({r},{c})-{direction}->({nr},{nc}); left as-is.")
                    continue

                grid[r][c] = new_self
                grid[nr][nc] = new_neighbor
                fixes += 1
                if verbose:
                    print(f"  Fixed ({r},{c})-{direction}->({nr},{nc}): {edge} -> "
                          f"{tileset.tiles[new_self].edge(direction)}")

    return grid, fixes