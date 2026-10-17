"""Enforces exactly one door for a given room instance (used for the
Living Room requirement: not walled in, but also not multiply-connected
-- exactly one door, and that door must lead to a corridor).
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from wfc_core import DIRS, OPPOSITE  # noqa: E402
from wall_cleanup import _find_edge_substitute  # noqa: E402
from graph_utils import find_room_components  # noqa: E402


def enforce_single_corridor_door(tile_id_grid, tileset, room_type="LIV", verbose=False):
    """For every instance (connected component) of `room_type`, ensure
    it has exactly one door, and that door connects to a COR cell.

    Any door to a non-corridor room is always closed first (it violates
    the corridor-only rule regardless of how many doors exist). Then,
    among the remaining corridor-connected doors: extras beyond the
    first are closed, or if none exist, one is opened to an adjacent
    COR cell.

    Never mutates the input grid. Returns (grid, results) where results
    is a list of per-instance dicts: {"doors_after": int, "fixed": bool}.
    """
    grid = [row[:] for row in tile_id_grid]
    rows, cols = len(grid), len(grid[0])
    results = []

    _, components = find_room_components(grid, tileset)
    target_components = [
        members for members in components.values()
        if tileset.tiles[grid[members[0][0]][members[0][1]]].room_type == room_type
    ]

    for members in target_components:
        member_set = set(members)

        def find_exit_doors():
            cor_edges, noncor_edges = [], []
            for (r, c) in members:
                tile = tileset.tiles[grid[r][c]]
                for direction, (dr, dc) in DIRS.items():
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < rows and 0 <= nc < cols):
                        continue
                    if (nr, nc) in member_set:
                        continue
                    if tile.edge(direction) != "D":
                        continue
                    neighbor = tileset.tiles[grid[nr][nc]]
                    entry = (r, c, direction, nr, nc)
                    if neighbor.room_type == "COR":
                        cor_edges.append(entry)
                    else:
                        noncor_edges.append(entry)
            return cor_edges, noncor_edges

        cor_edges, noncor_edges = find_exit_doors()
        fixed = True

        # Step 1: close every direct door to a non-corridor room --
        # these violate the corridor-only rule regardless of count.
        for (r, c, direction, nr, nc) in noncor_edges:
            opp = OPPOSITE[direction]
            new_self = _find_edge_substitute(tileset, grid[r][c], direction, "W")
            new_neighbor = _find_edge_substitute(tileset, grid[nr][nc], opp, "W")
            if new_self is not None and new_neighbor is not None:
                grid[r][c] = new_self
                grid[nr][nc] = new_neighbor
                if verbose:
                    print(f"  Closed direct non-corridor {room_type} door at ({r},{c})-{direction}->({nr},{nc}).")
            else:
                fixed = False
                if verbose:
                    print(f"  Could not close non-corridor {room_type} door at ({r},{c}); no wall substitute found.")

        # Step 2: among corridor doors, keep exactly one.
        if len(cor_edges) > 1:
            for (r, c, direction, nr, nc) in cor_edges[1:]:
                opp = OPPOSITE[direction]
                new_self = _find_edge_substitute(tileset, grid[r][c], direction, "W")
                new_neighbor = _find_edge_substitute(tileset, grid[nr][nc], opp, "W")
                if new_self is not None and new_neighbor is not None:
                    grid[r][c] = new_self
                    grid[nr][nc] = new_neighbor
                    if verbose:
                        print(f"  Closed extra corridor {room_type} door at ({r},{c})-{direction}->({nr},{nc}).")
                else:
                    fixed = False

        elif len(cor_edges) == 0:
            opened = False
            for (r, c) in members:
                tile = tileset.tiles[grid[r][c]]
                for direction, (dr, dc) in DIRS.items():
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < rows and 0 <= nc < cols):
                        continue
                    if (nr, nc) in member_set:
                        continue
                    neighbor = tileset.tiles[grid[nr][nc]]
                    if neighbor.room_type != "COR" or tile.edge(direction) == "D":
                        continue
                    opp = OPPOSITE[direction]
                    new_self = _find_edge_substitute(tileset, grid[r][c], direction, "D")
                    new_neighbor = _find_edge_substitute(tileset, grid[nr][nc], opp, "D")
                    if new_self is not None and new_neighbor is not None:
                        grid[r][c] = new_self
                        grid[nr][nc] = new_neighbor
                        opened = True
                        if verbose:
                            print(f"  Opened {room_type} door at ({r},{c})-{direction}->({nr},{nc}).")
                        break
                if opened:
                    break
            fixed = fixed and opened

        final_cor_edges, final_noncor_edges = find_exit_doors()
        results.append({
            "doors_after": len(final_cor_edges) + len(final_noncor_edges),
            "fixed": fixed and len(final_cor_edges) == 1 and len(final_noncor_edges) == 0,
        })

    return grid, results