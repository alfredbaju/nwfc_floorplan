"""Unified room-to-corridor (and other) connection enforcement.

Every room instance gets exactly one connection (never a direct
connection to a second, unrelated room). Which room type it connects to
depends on the room:
  - The floor's "entry point" (the building ENTRANCE on the ground
    floor, or the STAIRCASE on upper floors, where there's no building
    entrance) connects directly to the LIVING ROOM, matching how a real
    entryway typically opens straight into a living space.
  - BATHROOMS connect directly to a BEDROOM (an ensuite), UNLESS there
    are more bathrooms than bedrooms overall -- in that case there
    aren't enough bedrooms for every bathroom to attach to one, so
    bathrooms fall back to a normal corridor connection instead.
  - Every other room type (BED, KIT, LIV when not claimed by the entry
    point, STAIR when not itself the entry point) connects to a
    corridor: a real door for BED/BATH/STAIR/ENT, an open archway (no
    door) for KIT/LIV.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from wfc_core import DIRS, OPPOSITE  # noqa: E402
from wall_cleanup import _find_edge_substitute  # noqa: E402
from graph_utils import find_room_components  # noqa: E402

CONNECTION_EDGE_TYPES = {"D", "O", "C"}
OPEN_ROOM_TYPES = {"KIT", "LIV", "BALC"}  # open archway/sliding door, no closable door


def _connection_edge_type_for(room_type, connected_to):
    if room_type == "BALC":
        return "O"  # a balcony's access is always an open sliding door, whatever it connects to
    if room_type in OPEN_ROOM_TYPES and connected_to == "COR":
        return "O"
    return "D"


def _find_connections(grid, tileset, members, member_set, rows, cols, target_type):
    """All connection edges (D/O/C) leaving this room instance, split
    into (matching target_type, everything else)."""
    matches, others = [], []
    for (r, c) in members:
        tile = tileset.tiles[grid[r][c]]
        for direction, (dr, dc) in DIRS.items():
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if (nr, nc) in member_set:
                continue
            if tile.edge(direction) not in CONNECTION_EDGE_TYPES:
                continue
            neighbor = tileset.tiles[grid[nr][nc]]
            entry = (r, c, direction, nr, nc)
            if neighbor.room_type == target_type:
                matches.append(entry)
            else:
                others.append(entry)
    return matches, others


def _close_edges(grid, tileset, edges):
    all_ok = True
    for (r, c, direction, nr, nc) in edges:
        opp = OPPOSITE[direction]
        new_self = _find_edge_substitute(tileset, grid[r][c], direction, "W")
        new_neighbor = _find_edge_substitute(tileset, grid[nr][nc], opp, "W")
        if new_self is not None and new_neighbor is not None:
            grid[r][c] = new_self
            grid[nr][nc] = new_neighbor
        else:
            all_ok = False
    return all_ok


def _set_edge_type(grid, tileset, edge, target_edge):
    (r, c, direction, nr, nc) = edge
    tile = tileset.tiles[grid[r][c]]
    if tile.edge(direction) == target_edge:
        return True
    opp = OPPOSITE[direction]
    new_self = _find_edge_substitute(tileset, grid[r][c], direction, target_edge)
    new_neighbor = _find_edge_substitute(tileset, grid[nr][nc], opp, target_edge)
    if new_self is not None and new_neighbor is not None:
        grid[r][c] = new_self
        grid[nr][nc] = new_neighbor
        return True
    return False


def _try_open_connection(grid, tileset, members, member_set, rows, cols, target_type, edge_type):
    for (r, c) in members:
        tile = tileset.tiles[grid[r][c]]
        for direction, (dr, dc) in DIRS.items():
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if (nr, nc) in member_set:
                continue
            neighbor = tileset.tiles[grid[nr][nc]]
            if neighbor.room_type != target_type or tile.edge(direction) != "W":
                continue
            if _set_edge_type(grid, tileset, (r, c, direction, nr, nc), edge_type):
                return True
    return False


def _enforce_single_connection(grid, tileset, room_type, members, rows, cols, preferred_target, fallback_target,
                                protected_types=()):
    """Give this room instance exactly one connection, preferring
    preferred_target (e.g. LIV for an entry point, BED for a bathroom),
    falling back to fallback_target (normally COR) if the preferred type
    isn't adjacent at all. All other connections are closed, EXCEPT
    connections to any type listed in protected_types -- used so that,
    e.g., BED's own corridor-connection enforcement doesn't close a
    BED-BATH door that BATH's own (later) pass is going to claim as its
    one connection."""
    member_set = set(members)

    pref_matches, other = _find_connections(grid, tileset, members, member_set, rows, cols, preferred_target)
    if protected_types:
        other = [
            (r, c, direction, nr, nc) for (r, c, direction, nr, nc) in other
            if tileset.tiles[grid[nr][nc]].room_type not in protected_types
        ]
    fixed = _close_edges(grid, tileset, other)

    if len(pref_matches) > 1:
        fixed = _close_edges(grid, tileset, pref_matches[1:]) and fixed
        pref_matches = pref_matches[:1]

    if len(pref_matches) == 1:
        edge_type = _connection_edge_type_for(room_type, preferred_target)
        fixed = _set_edge_type(grid, tileset, pref_matches[0], edge_type) and fixed
        return preferred_target, fixed

    if preferred_target == fallback_target:
        opened = _try_open_connection(grid, tileset, members, member_set, rows, cols, fallback_target,
                                       _connection_edge_type_for(room_type, fallback_target))
        return (fallback_target if opened else None), (fixed and opened)

    # No existing connection to the preferred target -- but it might
    # still be geometrically adjacent through a plain wall that just
    # hasn't been opened yet (e.g. a kitchen-side entrance whose
    # raw-generated tile happened not to have a door facing the
    # kitchen). Try opening one before falling back to the normal target.
    opened_preferred = _try_open_connection(grid, tileset, members, member_set, rows, cols, preferred_target,
                                             _connection_edge_type_for(room_type, preferred_target))
    if opened_preferred:
        return preferred_target, fixed

    # Preferred type not adjacent at all -- fall back to the normal target.
    fb_matches, other2 = _find_connections(grid, tileset, members, member_set, rows, cols, fallback_target)
    if protected_types:
        other2 = [
            (r, c, direction, nr, nc) for (r, c, direction, nr, nc) in other2
            if tileset.tiles[grid[nr][nc]].room_type not in protected_types
        ]
    fixed = _close_edges(grid, tileset, other2) and fixed
    if len(fb_matches) > 1:
        fixed = _close_edges(grid, tileset, fb_matches[1:]) and fixed
        fb_matches = fb_matches[:1]
    if len(fb_matches) == 1:
        edge_type = _connection_edge_type_for(room_type, fallback_target)
        fixed = _set_edge_type(grid, tileset, fb_matches[0], edge_type) and fixed
        return fallback_target, fixed

    opened = _try_open_connection(grid, tileset, members, member_set, rows, cols, fallback_target,
                                   _connection_edge_type_for(room_type, fallback_target))
    return (fallback_target if opened else None), (fixed and opened)


def enforce_room_connection_rules(tile_id_grid, tileset, verbose=False):
    """Apply the unified connection rule to every room instance. COR and
    EXT are left alone. Never mutates the input grid.

    Returns (grid, results): a list of per-instance dicts
    {"room_type": str, "connections_after": int, "edge_type": str or None,
     "connected_to": str or None, "fixed": bool}.
    """
    grid = [row[:] for row in tile_id_grid]
    rows, cols = len(grid), len(grid[0])

    _, components = find_room_components(grid, tileset)
    type_of = {
        cid: tileset.tiles[grid[members[0][0]][members[0][1]]].room_type
        for cid, members in components.items()
    }
    has_ent = "ENT" in type_of.values()
    entry_point_type = "ENT" if has_ent else "STAIR"

    # There can be more than one entry-point component (a second
    # entrance near the kitchen sits in its own component, since it's
    # not adjacent to the primary one). Each is independently assigned
    # its preferred target based on which room it's ALREADY adjacent to
    # (from macro placement) -- the primary entrance is LIV-adjacent,
    # a kitchen-side entrance is KIT-adjacent -- rather than every
    # entry-point component blindly trying LIV first, which would make
    # a second entrance fight the first one for the same living room
    # connection.
    entry_point_components = [
        (cid, members) for cid, members in components.items() if type_of[cid] == entry_point_type
    ]

    def _raw_adjacent_to(members, member_set, target_type):
        for (r, c) in members:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if not (0 <= nr < rows and 0 <= nc < cols):
                    continue
                if (nr, nc) in member_set:
                    continue
                if tileset.tiles[grid[nr][nc]].room_type == target_type:
                    return True
        return False

    def _entry_preferred_target(members):
        # Checks raw geometric adjacency (any edge, including a plain
        # wall), not just an already-existing D/O/C connection -- a
        # macro-reserved kitchen-side entrance is often touching KIT
        # through a wall edge at this point (nothing has opened it yet),
        # and only checking for pre-existing connections would miss that
        # entirely, silently falling back to LIV instead.
        member_set = set(members)
        liv_matches, _ = _find_connections(grid, tileset, members, member_set, rows, cols, "LIV")
        if liv_matches or _raw_adjacent_to(members, member_set, "LIV"):
            return "LIV"
        kit_matches, _ = _find_connections(grid, tileset, members, member_set, rows, cols, "KIT")
        if kit_matches or _raw_adjacent_to(members, member_set, "KIT"):
            return "KIT"
        return "LIV"  # default preference when neither is adjacent yet

    entry_point_targets = {cid: _entry_preferred_target(members) for cid, members in entry_point_components}

    bed_count = sum(1 for t in type_of.values() if t == "BED")
    bath_count = sum(1 for t in type_of.values() if t == "BATH")
    # Bathrooms attach directly to a bedroom (ensuite) UNLESS there are
    # more bathrooms than bedrooms -- not enough bedrooms for every
    # bathroom to have one, so the surplus falls back to the corridor.
    bath_prefers_bedroom = bath_count <= bed_count

    all_targets = [(cid, type_of[cid], members) for cid, members in components.items() if type_of[cid] not in ("COR", "EXT")]

    # Process order matters: the entry point needs LIV to still be
    # "unclaimed" when it looks for it; LIV's own rule needs to know
    # whether the entry point already claimed it; bedrooms should be
    # established before bathrooms look for one to attach to.
    order_key = {entry_point_type: 0, "LIV": 1, "BED": 2, "BATH": 3, "BALC": 4}
    all_targets.sort(key=lambda item: order_key.get(item[1], 4))

    liv_claimed_by_entry = False
    kit_claimed_by_entry = False
    results = []

    for cid, room_type, members in all_targets:
        if room_type == entry_point_type:
            preferred = entry_point_targets.get(cid, "LIV")
            connected_to, fixed = _enforce_single_connection(
                grid, tileset, room_type, members, rows, cols, preferred_target=preferred, fallback_target="COR")
            if connected_to == "LIV":
                liv_claimed_by_entry = True
            elif connected_to == "KIT":
                kit_claimed_by_entry = True

        elif room_type == "LIV":
            if liv_claimed_by_entry:
                member_set = set(members)
                entry_matches, other = _find_connections(grid, tileset, members, member_set, rows, cols, entry_point_type)
                other = [e for e in other if tileset.tiles[grid[e[3]][e[4]]].room_type != "BALC"]
                fixed = _close_edges(grid, tileset, other)
                connected_to = entry_point_type if entry_matches else None
            else:
                connected_to, fixed = _enforce_single_connection(
                    grid, tileset, room_type, members, rows, cols, preferred_target="COR", fallback_target="COR",
                    protected_types=("BALC",))

        elif room_type == "BATH":
            preferred = "BED" if bath_prefers_bedroom else "COR"
            connected_to, fixed = _enforce_single_connection(
                grid, tileset, room_type, members, rows, cols, preferred_target=preferred, fallback_target="COR")

        elif room_type == "BALC":
            # A balcony connects to whichever of LIV/BED it's actually
            # adjacent to (checked directly, since there are two
            # possible preferred partners here rather than one).
            member_set = set(members)
            liv_matches, _ = _find_connections(grid, tileset, members, member_set, rows, cols, "LIV")
            bed_matches, _ = _find_connections(grid, tileset, members, member_set, rows, cols, "BED")
            preferred = "LIV" if liv_matches else ("BED" if bed_matches else "COR")
            connected_to, fixed = _enforce_single_connection(
                grid, tileset, room_type, members, rows, cols, preferred_target=preferred, fallback_target="COR")

        elif room_type == "KIT" and kit_claimed_by_entry:
            # A second entrance near the kitchen claimed this KIT
            # instance as its one connection -- keep that connection
            # (still open/no-door, matching KIT's normal style) and
            # close anything else, rather than separately trying to
            # connect to a corridor too.
            member_set = set(members)
            entry_matches, other = _find_connections(grid, tileset, members, member_set, rows, cols, entry_point_type)
            fixed = _close_edges(grid, tileset, other)
            if entry_matches:
                fixed = _set_edge_type(grid, tileset, entry_matches[0], "O") and fixed
            connected_to = entry_point_type if entry_matches else None

        else:
            protected = []
            if room_type == "BED" and bath_prefers_bedroom:
                protected.append("BATH")
            if room_type in ("BED", "LIV"):
                protected.append("BALC")
            connected_to, fixed = _enforce_single_connection(
                grid, tileset, room_type, members, rows, cols, preferred_target="COR", fallback_target="COR",
                protected_types=tuple(protected))

        member_set = set(members)
        final_matches, final_other = _find_connections(grid, tileset, members, member_set, rows, cols, connected_to or "COR")
        n_final = len(final_matches) + len(final_other)
        used_edge_type = None
        if final_matches:
            r0, c0, d0, _, _ = final_matches[0]
            used_edge_type = tileset.tiles[grid[r0][c0]].edge(d0)

        results.append({
            "room_type": room_type,
            "connections_after": n_final,
            "edge_type": used_edge_type,
            "connected_to": connected_to,
            "fixed": bool(fixed) and n_final == 1,
        })
        if verbose:
            print(f"  {room_type}: connected_to={connected_to}, connections_after={n_final}, fixed={fixed}")

    return grid, results