"""Post-generation connectivity repair.

Everything built so far (subcomplete tileset, Layer 2 enforcement, the
learned selection heuristic) only ever influences generation through
local, per-cell decisions -- and E3/E5 confirmed none of them touch
functional solvability (reachability from the entrance), because
reachability is a *global* graph property no local mechanism can
guarantee by construction.

This module takes a different, more direct approach: generate as usual,
then explicitly detect disconnected room components and repair them by
finding a valid tile substitution that converts an isolating wall (W)
into a door (D) on both sides of the boundary -- without breaking any of
that cell's OTHER edge constraints. This guarantees the property we
actually care about, rather than biasing toward it and hoping.
"""

import sys
import os
from collections import deque

sys.path.insert(0, os.path.dirname(__file__))

from wfc_core import DIRS, OPPOSITE  # noqa: E402
from wall_cleanup import _find_edge_substitute  # noqa: E402
from graph_utils import find_room_components  # noqa: E402


def _connected_components(tile_id_grid, tileset, exterior_room_type="EXT"):
    """Partition all non-exterior cells into connected components, using
    traversable (D/O/C) edges. Returns a list of sets of (r, c) cells."""
    rows, cols = len(tile_id_grid), len(tile_id_grid[0])
    traversable = {"D", "O", "C"}

    room_cells = {
        (r, c)
        for r in range(rows)
        for c in range(cols)
        if tileset.tiles[tile_id_grid[r][c]].room_type != exterior_room_type
    }

    visited = set()
    components = []
    for start in room_cells:
        if start in visited:
            continue
        comp = set()
        queue = deque([start])
        visited.add(start)
        while queue:
            r, c = queue.popleft()
            comp.add((r, c))
            tile = tileset.tiles[tile_id_grid[r][c]]
            for direction, (dr, dc) in DIRS.items():
                if tile.edge(direction) not in traversable:
                    continue
                nr, nc = r + dr, c + dc
                if (nr, nc) not in room_cells or (nr, nc) in visited:
                    continue
                neighbor = tileset.tiles[tile_id_grid[nr][nc]]
                if neighbor.edge(OPPOSITE[direction]) not in traversable:
                    continue
                visited.add((nr, nc))
                queue.append((nr, nc))
        components.append(comp)
    return components


def enforce_corridor_only_doors(tile_id_grid, tileset, verbose=False):
    """Convert any direct room-to-room door (a D edge where NEITHER side
    is COR) into a wall. Every room's access must be mediated by a
    corridor -- no direct bedroom-to-bathroom doors, no direct
    entrance-to-living-room doors, etc. Uses the same edge-preserving
    substitution technique as repair.py. Never mutates the input grid.

    This intentionally runs BEFORE repair_connectivity in the pipeline:
    removing direct doors can disconnect rooms that were only reachable
    through them, and repair_connectivity (called with
    prefer_corridor=True) is what re-establishes reachability the
    correct way, via a corridor-mediated door instead of undoing this
    rule.
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
                if tile.edge(direction) != "D":
                    continue
                if "COR" in (tile.room_type, neighbor.room_type):
                    continue  # fine -- one side is a corridor
                if (nr, nc) < (r, c):
                    continue  # process each boundary once

                opp = OPPOSITE[direction]
                new_self = _find_edge_substitute(tileset, grid[r][c], direction, "W")
                new_neighbor = _find_edge_substitute(tileset, grid[nr][nc], opp, "W")
                if new_self is not None and new_neighbor is not None:
                    grid[r][c] = new_self
                    grid[nr][nc] = new_neighbor
                    fixes += 1
                    if verbose:
                        print(f"  Closed direct {tile.room_type}-{neighbor.room_type} door "
                              f"at ({r},{c})-{direction}->({nr},{nc}).")
                elif verbose:
                    print(f"  Could not close direct {tile.room_type}-{neighbor.room_type} "
                          f"door at ({r},{c}); no wall substitute found.")

    return grid, fixes


def _find_wall_pairs_between(tile_id_grid, tileset, comp_a, comp_b):
    """All (r1, c1, direction, r2, c2) where (r1,c1) in comp_a and
    (r2,c2) in comp_b are orthogonally adjacent with a wall between them."""
    pairs = []
    rows, cols = len(tile_id_grid), len(tile_id_grid[0])
    for (r, c) in comp_a:
        tile = tileset.tiles[tile_id_grid[r][c]]
        for direction, (dr, dc) in DIRS.items():
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if (nr, nc) not in comp_b:
                continue
            if tile.edge(direction) == "W":
                pairs.append((r, c, direction, nr, nc))
    return pairs


def _find_door_substitute(tileset, current_tile_id, direction):
    """Find a tile with the same room_type as current_tile_id, identical
    on every edge except `direction` (which must be 'D' instead of the
    current wall), so swapping it in doesn't break any other neighbor's
    already-satisfied constraint. Returns a tile id, or None if no such
    tile exists in the tileset."""
    current = tileset.tiles[current_tile_id]
    target = {d: current.edge(d) for d in ("n", "s", "w", "e")}
    target[direction] = "D"

    for tid, t in tileset.tiles.items():
        if t.room_type != current.room_type:
            continue
        if all(t.edge(d) == target[d] for d in ("n", "s", "w", "e")):
            return tid
    return None


def repair_connectivity(tile_id_grid, tileset, entrance_room_type="ENT",
                         exterior_room_type="EXT", max_repairs=100, prefer_corridor=False,
                         protect_types=(), verbose=False):
    """Attempt to connect every room component to the entrance's
    component by converting isolating walls into doors, using only
    substitutions that keep every other edge constraint intact.

    prefer_corridor: if True, candidate wall pairs where at least one
    side is a COR cell are tried first, falling back to any candidate
    only if no corridor-mediated option exists.

    protect_types: room types (e.g. ("KIT",)) that must never end up
    directly door-connected to a non-corridor room as a repair fallback
    -- without this, prefer_corridor's fallback (used only when no
    corridor-mediated fix exists at all) could still pick a direct
    KIT-to-bedroom door to restore connectivity, silently reintroducing
    exactly the kind of connection enforce_room_connection_rules is
    supposed to prevent.

    If honoring protect_types would leave some component genuinely
    unreachable (no corridor-mediated fix exists at all for it -- e.g. a
    room whose only neighbors are other protected/non-corridor rooms),
    this automatically retries WITHOUT the protection as a fallback:
    full connectivity is the harder, non-negotiable requirement, and a
    direct room-to-room door in a rare edge case is a far better outcome
    than leaving cells unreachable entirely.

    Returns (repaired_grid, repairs_made, fully_connected). repairs_made
    is a list of (r, c, direction) edges that were converted. This never
    mutates the input grid -- a deep copy is repaired and returned.
    """
    grid, repairs_made, connected = _repair_connectivity_core(
        tile_id_grid, tileset, entrance_room_type, exterior_room_type,
        max_repairs, prefer_corridor, protect_types, verbose,
    )
    if not connected and protect_types:
        if verbose:
            print("  Could not achieve full connectivity while honoring protect_types -- "
                  "retrying without it (connectivity takes priority).")
        grid, repairs_made, connected = _repair_connectivity_core(
            tile_id_grid, tileset, entrance_room_type, exterior_room_type,
            max_repairs, prefer_corridor, (), verbose,
        )
    return grid, repairs_made, connected


def _repair_connectivity_core(tile_id_grid, tileset, entrance_room_type="ENT",
                               exterior_room_type="EXT", max_repairs=100, prefer_corridor=False,
                               protect_types=(), verbose=False):
    grid = [row[:] for row in tile_id_grid]
    repairs_made = []

    def _violates_protection(pair):
        r1, c1, direction, r2, c2 = pair
        t1 = tileset.tiles[grid[r1][c1]].room_type
        t2 = tileset.tiles[grid[r2][c2]].room_type
        if t1 == t2:
            return False
        for protected in protect_types:
            if (t1 == protected and t2 != "COR") or (t2 == protected and t1 != "COR"):
                return True
        return False

    for attempt in range(max_repairs):
        components = _connected_components(grid, tileset, exterior_room_type)
        if len(components) <= 1:
            return grid, repairs_made, True

        entrance_comp = None
        for comp in components:
            if any(tileset.tiles[grid[r][c]].room_type == entrance_room_type for (r, c) in comp):
                entrance_comp = comp
                break
        if entrance_comp is None:
            if verbose:
                print("No entrance cell found; cannot repair.")
            return grid, repairs_made, False

        other_components = [c for c in components if c is not entrance_comp]
        other_components.sort(key=len, reverse=True)

        repaired_this_round = False
        for comp in other_components:
            candidates = _find_wall_pairs_between(grid, tileset, comp, entrance_comp)
            candidates += _find_wall_pairs_between(grid, tileset, entrance_comp, comp)
            if protect_types:
                candidates = [p for p in candidates if not _violates_protection(p)]
            if prefer_corridor:
                def _is_corridor_pair(pair):
                    r1, c1, direction, r2, c2 = pair
                    t1 = tileset.tiles[grid[r1][c1]].room_type
                    t2 = tileset.tiles[grid[r2][c2]].room_type
                    return 0 if "COR" in (t1, t2) else 1
                candidates.sort(key=_is_corridor_pair)
            for (r1, c1, direction, r2, c2) in candidates:
                new_tile_1 = _find_door_substitute(tileset, grid[r1][c1], direction)
                if new_tile_1 is None:
                    continue
                new_tile_2 = _find_door_substitute(tileset, grid[r2][c2], OPPOSITE[direction])
                if new_tile_2 is None:
                    continue
                grid[r1][c1] = new_tile_1
                grid[r2][c2] = new_tile_2
                repairs_made.append((r1, c1, direction))
                repaired_this_round = True
                if verbose:
                    print(f"  Repaired: ({r1},{c1}) <-{direction}-> ({r2},{c2})")
                break
            if repaired_this_round:
                break

        if not repaired_this_round:
            if verbose:
                print(f"No valid substitution found to connect a remaining component "
                      f"after {len(repairs_made)} repairs.")
            return grid, repairs_made, False

    components = _connected_components(grid, tileset, exterior_room_type)
    return grid, repairs_made, len(components) <= 1


def enforce_internal_room_connectivity(tile_id_grid, tileset, verbose=False):
    """Ensure every multi-subgrid room (Kitchen, Bedroom, Living, etc.
    with ROOM_SIZE_SUBGRIDS > 1) is internally traversable throughout
    its own footprint, not silently split into disconnected pieces by a
    wall between two of its own subgrids.

    Nothing in the core Layer 1/2 tile rules actually REQUIRES two
    physically-adjacent cells of the same room type to have an open
    connection between them -- a wall there is structurally valid, just
    architecturally wrong for what's supposed to be one contiguous room.
    This showed up in practice as an entire multi-subgrid room's
    boundary coming out fully walled, splitting it into two same-type
    "islands" that neither ordinary reachability repair nor
    unify_corridor_network catch (both are built around
    entrance-relative and corridor-relative reachability, not a room's
    connectivity to itself).

    For every room type instance that isn't already one connected piece
    (checked via traversable D/O/C edges, not mere physical adjacency),
    this finds the shortest wall boundary between its largest fragment
    and each smaller one and opens it as an "O" edge (an internal
    doorless connection, matching how a same-room boundary should read).
    Never mutates the input grid.

    Returns (grid, fixes_made).
    """
    grid = [row[:] for row in tile_id_grid]
    rows, cols = len(grid), len(grid[0])
    traversable = {"D", "O", "C"}
    fixes_made = 0

    _, components = find_room_components(grid, tileset)
    # Group same-room-type "logical" components (from find_room_components,
    # grouped purely by physical adjacency) so we can check each one's
    # ACTUAL traversability internally.
    for members in components.values():
        room_type = tileset.tiles[grid[members[0][0]][members[0][1]]].room_type
        if room_type in ("COR", "EXT"):
            continue

        member_set = set(members)
        # Find traversable-edge sub-fragments within this same-type group.
        visited = set()
        fragments = []
        for start in members:
            if start in visited:
                continue
            frag = set()
            queue = deque([start])
            visited.add(start)
            while queue:
                r, c = queue.popleft()
                frag.add((r, c))
                tile = tileset.tiles[grid[r][c]]
                for direction, (dr, dc) in DIRS.items():
                    if tile.edge(direction) not in traversable:
                        continue
                    nr, nc = r + dr, c + dc
                    if (nr, nc) not in member_set or (nr, nc) in visited:
                        continue
                    neighbor = tileset.tiles[grid[nr][nc]]
                    if neighbor.edge(OPPOSITE[direction]) not in traversable:
                        continue
                    visited.add((nr, nc))
                    queue.append((nr, nc))
            fragments.append(frag)

        if len(fragments) <= 1:
            continue  # already internally connected

        fragments.sort(key=len, reverse=True)
        main_frag = fragments[0]
        for other_frag in fragments[1:]:
            joined = False
            for (r, c) in other_frag:
                if joined:
                    break
                tile = tileset.tiles[grid[r][c]]
                for direction, (dr, dc) in DIRS.items():
                    nr, nc = r + dr, c + dc
                    if (nr, nc) not in main_frag:
                        continue
                    if tile.edge(direction) != "W":
                        continue
                    opp = OPPOSITE[direction]
                    new_self = _find_edge_substitute(tileset, grid[r][c], direction, "O")
                    new_neighbor = _find_edge_substitute(tileset, grid[nr][nc], opp, "O")
                    if new_self is not None and new_neighbor is not None:
                        grid[r][c] = new_self
                        grid[nr][nc] = new_neighbor
                        fixes_made += 1
                        joined = True
                        if verbose:
                            print(f"  Reconnected {room_type} fragment at ({r},{c})-{direction}->({nr},{nc}).")
                        break
            if joined:
                main_frag = main_frag | other_frag

    return grid, fixes_made


def unify_corridor_network(tile_id_grid, tileset, verbose=False):
    """Ensure all COR cells form a single connected network among
    THEMSELVES (via traversable D/O/C edges between adjacent COR cells),
    not fragmented "islands" that only happen to link up indirectly
    through unrelated rooms. Real hallway systems are one continuous
    connected spine, not scattered disconnected patches -- overall
    reachability from the entrance can technically hold even when the
    corridor network itself is fragmented, so this is a distinct check
    from repair_connectivity's entrance-based reachability.

    Repairs by opening a wall between two adjacent COR cells in
    different fragments into an "O" (open) edge -- a corridor-to-corridor
    join is architecturally just open hallway continuing, not a door.
    Never mutates the input grid.
    """
    grid = [row[:] for row in tile_id_grid]
    rows, cols = len(grid), len(grid[0])
    traversable = {"D", "O", "C"}

    def corridor_components():
        cor_cells = {
            (r, c) for r in range(rows) for c in range(cols)
            if tileset.tiles[grid[r][c]].room_type == "COR"
        }
        visited = set()
        comps = []
        for start in cor_cells:
            if start in visited:
                continue
            comp = set()
            queue = deque([start])
            visited.add(start)
            while queue:
                r, c = queue.popleft()
                comp.add((r, c))
                tile = tileset.tiles[grid[r][c]]
                for direction, (dr, dc) in DIRS.items():
                    if tile.edge(direction) not in traversable:
                        continue
                    nr, nc = r + dr, c + dc
                    if (nr, nc) not in cor_cells or (nr, nc) in visited:
                        continue
                    neighbor = tileset.tiles[grid[nr][nc]]
                    if neighbor.edge(OPPOSITE[direction]) not in traversable:
                        continue
                    visited.add((nr, nc))
                    queue.append((nr, nc))
            comps.append(comp)
        return comps

    for _ in range(100):
        comps = corridor_components()
        if len(comps) <= 1:
            return grid, True

        comps.sort(key=len, reverse=True)
        main_comp = comps[0]
        joined = False

        for other_comp in comps[1:]:
            # find a wall pair directly between main_comp and other_comp
            for (r, c) in main_comp:
                if joined:
                    break
                tile = tileset.tiles[grid[r][c]]
                for direction, (dr, dc) in DIRS.items():
                    nr, nc = r + dr, c + dc
                    if (nr, nc) not in other_comp:
                        continue
                    if tile.edge(direction) != "W":
                        continue
                    opp = OPPOSITE[direction]
                    new_self = _find_edge_substitute(tileset, grid[r][c], direction, "O")
                    new_neighbor = _find_edge_substitute(tileset, grid[nr][nc], opp, "O")
                    if new_self is not None and new_neighbor is not None:
                        grid[r][c] = new_self
                        grid[nr][nc] = new_neighbor
                        joined = True
                        if verbose:
                            print(f"  Joined corridor fragments at ({r},{c})-{direction}->({nr},{nc}).")
                        break
            if joined:
                break

        if not joined:
            if verbose:
                print("Could not find a way to join a remaining corridor fragment.")
            return grid, False

    return grid, len(corridor_components()) <= 1