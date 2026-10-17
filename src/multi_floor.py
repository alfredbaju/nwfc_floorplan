"""Multi-floor building generation.

Generates 1-3 floors, each an independent N-WFC floor plan, connected by
a staircase kept at the exact same macro subgrid position on every floor
(a real stairwell is a fixed vertical shaft, not a different spot per
floor). Only the ground floor gets a building entrance; upper floors are
reached via the staircase instead, so their reachability is validated
from the staircase position rather than an entrance.

This is deliberately built as an orchestration layer on top of the
existing single-floor pipeline (solve_zoned_with_exact_counts,
wall_cleanup, entrance_cap, repair, ensure_entrance_on_border,
ensure_living_room_entrance_door) -- no per-floor generation logic is
duplicated here.
"""

import sys
import os
import random
import time as _time

sys.path.insert(0, os.path.dirname(__file__))

from zoning import (  # noqa: E402
    solve_zoned_with_exact_counts,
    count_actual_room_instances,
    ensure_entrance_on_border,
    compute_grid_size_for_rooms,
    compute_room_sizes_and_grid,
)
from wall_cleanup import cleanup_cross_room_edges  # noqa: E402
from entrance_cap import cap_entrance_count  # noqa: E402
from repair import repair_connectivity, unify_corridor_network, enforce_internal_room_connectivity  # noqa: E402
from room_connections import enforce_room_connection_rules  # noqa: E402
from validity import validate, check_reachability, check_structural_validity  # noqa: E402


def _strip_stray_entrances(tile_id_grid, tileset):
    """Convert any ENT cells to plain COR -- used on upper floors, which
    have no building entrance (they're reached via the staircase
    instead). Uses the same edge-preserving substitution as elsewhere so
    structural validity is never affected."""
    grid = [row[:] for row in tile_id_grid]
    rows, cols = len(grid), len(grid[0])
    for r in range(rows):
        for c in range(cols):
            if tileset.tiles[grid[r][c]].room_type != "ENT":
                continue
            current = tileset.tiles[grid[r][c]]
            for tid, t in tileset.tiles.items():
                if t.room_type == "COR" and all(
                    t.edge(d) == current.edge(d) for d in ("n", "s", "w", "e")
                ):
                    grid[r][c] = tid
                    break
    return grid


class FloorResult:
    def __init__(self, floor_number, grid, is_ground_floor, structural_valid,
                 reachable, unreachable_cells, room_counts_actual, stair_positions):
        self.floor_number = floor_number
        self.grid = grid
        self.is_ground_floor = is_ground_floor
        self.structural_valid = structural_valid
        self.reachable = reachable
        self.unreachable_cells = unreachable_cells
        self.room_counts_actual = room_counts_actual
        self.stair_positions = stair_positions


def distribute_rooms_across_floors(room_counts, num_floors):
    """Split building-wide TOTAL room counts across floors.

    LIV is always exactly 1 per floor -- a fixed architectural
    requirement (every floor has its own living room), not part of the
    distributed pool. Any LIV value in room_counts is ignored for this
    purpose.

    BALC (balcony) is a special case: it's ONLY distributed among UPPER
    floors (floor 2 and above) -- a real ground floor has a verandah or
    patio at most, not an elevated balcony, so the ground floor never
    gets one regardless of how many were requested. If num_floors == 1
    (no upper floors exist at all), any requested balconies are simply
    not placed anywhere -- there's nowhere architecturally valid to put
    them.

    Other room types (BED, BATH, KIT, ...) are the TOTAL desired across
    the whole building, split as evenly as possible across ALL floors
    (extra remainder rooms go to the lower floors first).

    Returns a list of per-floor room_counts dicts, one per floor.
    """
    per_floor = [dict() for _ in range(num_floors)]
    for room_type, total in room_counts.items():
        if room_type in ("LIV", "BALC"):
            continue
        base = total // num_floors
        remainder = total % num_floors
        for i in range(num_floors):
            count = base + (1 if i < remainder else 0)
            if count > 0:
                per_floor[i][room_type] = count

    balc_total = room_counts.get("BALC", 0)
    num_upper_floors = num_floors - 1
    if balc_total > 0 and num_upper_floors > 0:
        base = balc_total // num_upper_floors
        remainder = balc_total % num_upper_floors
        for i in range(1, num_floors):  # floors index 1..N-1 are the upper floors
            upper_idx = i - 1
            count = base + (1 if upper_idx < remainder else 0)
            if count > 0:
                per_floor[i]["BALC"] = count

    for i in range(num_floors):
        per_floor[i]["LIV"] = 1
    return per_floor


def generate_multi_floor_building(tileset, num_floors, room_counts, C=4,
                                   enforce_layer2=True, context_scorer=None,
                                   max_attempts_per_floor=30, seed=None,
                                   second_entrance_near_kitchen=False, total_time_budget=6.0,
                                   verbose=False):
    """Generate `num_floors` (1-3) connected floor plans.

    total_time_budget: a soft ceiling (seconds) on the WHOLE generation
    across all floors combined, not per floor -- each floor's own
    "quality search" (see solve_zoned_with_exact_counts) gets whatever
    time remains from this shared budget divided among the floors still
    left to generate, so an early floor that finishes quickly leaves
    more room for a harder later one, and floor budgets no longer simply
    stack on top of each other (which previously made a 2-floor building
    take up to 2x a single floor's own budget).

    second_entrance_near_kitchen: adds a service/back door near the
    kitchen on the GROUND FLOOR only -- upper floors have no building
    entrance at all (they're reached via the staircase), so a second
    entrance doesn't apply to them.

    room_counts: the BUILDING-WIDE TOTAL room request (e.g. "5 bedrooms"
    means 5 bedrooms total across every floor combined, not 5 on each
    floor) -- see distribute_rooms_across_floors. LIV is always exactly
    1 per floor regardless of what's passed here. STAIR is added
    automatically when num_floors > 1.

    Returns a list of FloorResult, one per floor, floor 1 first (ground).
    """
    if not (1 <= num_floors <= 3):
        raise ValueError("num_floors must be between 1 and 3")

    rng = random.Random(seed)
    per_floor_counts = distribute_rooms_across_floors(room_counts, num_floors)

    # Grid size must accommodate whichever floor needs the most actual
    # space, not just whichever has the most room instances -- the
    # sizing algorithm's total space need for a floor depends heavily
    # on its specific room composition (e.g. a floor with just 1
    # bedroom but 2 balconies can need MORE total area than a floor
    # with 3 bedrooms and a kitchen, since balcony size derives from
    # living room size, which grows to absorb padding when there are
    # few bedrooms to distribute it across).
    #
    # Each dimension is maxed independently across floors, rather than
    # picking one floor's exact (A, B) pair -- picking a single floor's
    # pair verified to force a DIFFERENT floor into a grid shaped
    # nothing like its own composition (e.g. forced from its own
    # preferred 6x5 into an unrelated 6x6), which in turn caused a
    # severe slowdown for THAT floor (237s, worse than the original
    # problem this was meant to fix). Maxing each dimension keeps both
    # floors closer to a grid shaped like their own actual needs.
    A, B = 1, 1
    for floor_counts in per_floor_counts:
        candidate_counts = dict(floor_counts)
        if num_floors > 1:
            candidate_counts["STAIR"] = 1
        cand_A, cand_B, _, _ = compute_room_sizes_and_grid(candidate_counts)
        A, B = max(A, cand_A), max(B, cand_B)

    floors = []
    stair_positions = None
    bath_positions = None
    generation_start = _time.time()
    # Reserve a slice of the total budget for post-processing (repair,
    # cleanup, corridor unification, etc. after each floor's raw
    # generation) -- that overhead isn't captured by
    # solve_zoned_with_exact_counts's own internal budget, but still
    # counts against the "generate everything within N seconds" goal.
    overhead_reserve = min(2.0, total_time_budget * 0.2)

    for floor_number in range(1, num_floors + 1):
        is_ground = floor_number == 1
        floor_room_counts = dict(per_floor_counts[floor_number - 1])
        if num_floors > 1:
            floor_room_counts["STAIR"] = 1

        forced = {"STAIR": stair_positions} if (stair_positions and not is_ground) else None
        exterior_backtrack_budget = 110 if forced else 30
        floor_high_density = floor_room_counts.get("BED", 0) + floor_room_counts.get("BATH", 0) >= 8
        density_multiplier = 2 if floor_high_density else 1
        floor_max_attempts = max_attempts_per_floor * 3 * density_multiplier if forced else max_attempts_per_floor * density_multiplier

        floors_remaining = num_floors - floor_number + 1
        time_remaining = total_time_budget - overhead_reserve - (_time.time() - generation_start)
        # Never let a floor's budget drop below a reasonable floor
        # (5.0s, matching the per-attempt cap inside
        # solve_zoned_with_exact_counts) -- an earlier floor consuming
        # more than its "fair share" of the shared budget (verified to
        # happen routinely: a hard first-floor scenario, e.g. a second
        # entrance combined with a large kitchen, can easily take
        # 7-13s on its own) must never be allowed to starve a later
        # floor down to the point where EVERY one of its attempts gets
        # interrupted before completing, which was causing outright
        # generation failures ("last_grid never gets set") rather than
        # just running long. Preventing a hard failure takes priority
        # over strictly enforcing the nominal total time target.
        floor_time_budget = max(18.0 * density_multiplier, time_remaining / floors_remaining)

        random.seed(rng.randint(0, 10_000_000))
        preferred = {"BATH": bath_positions} if (bath_positions and not is_ground) else None
        grid, plan, attempts, matched = solve_zoned_with_exact_counts(
            tileset, A=A, B=B, C=C, room_counts=floor_room_counts,
            enforce_layer2=enforce_layer2, context_scorer=context_scorer,
            max_attempts=floor_max_attempts, max_exterior_backtracks=exterior_backtrack_budget,
            forced_positions=forced, verbose=verbose,
            second_entrance_near_kitchen=(second_entrance_near_kitchen and is_ground),
            quality_search_time_budget=floor_time_budget, preferred_positions=preferred,
        )
        if grid is None:
            raise RuntimeError(
                f"Failed to generate floor {floor_number} after {max_attempts_per_floor} attempts."
            )

        grid, wall_fixes = cleanup_cross_room_edges(grid, tileset)
        # Guarantee every multi-subgrid room (KIT/BED/LIV/etc.) is
        # internally traversable throughout its own footprint -- nothing
        # in the core tile rules actually requires two physically
        # adjacent cells of the same room type to have an open
        # connection, so a wall between two of a room's own subgrids is
        # structurally valid but architecturally wrong, and can silently
        # split one requested room into disconnected same-type islands
        # that neither corridor-unification nor entrance-repair (both
        # built around reachability FROM the entrance, not a room's
        # connectivity to ITSELF) would ever catch.
        grid, _ = enforce_internal_room_connectivity(grid, tileset)

        if is_ground:
            max_ent = 2 if second_entrance_near_kitchen else 1
            diverse = ("LIV", "KIT") if second_entrance_near_kitchen else ()
            grid, _ = cap_entrance_count(grid, tileset, max_entrances=max_ent, preferred_types=("COR",),
                                          strict=True, prefer_diverse_adjacency=diverse)
            grid, _, _ = repair_connectivity(grid, tileset, prefer_corridor=True, protect_types=("KIT", "BALC"))
            grid, _ = ensure_entrance_on_border(grid, tileset)
        else:
            # Upper floors: repair connectivity relative to the
            # staircase (their "entrance" from below) instead of ENT --
            # repair_connectivity supports this directly via
            # entrance_room_type, no change needed to that module.
            grid, _, _ = repair_connectivity(grid, tileset, entrance_room_type="STAIR", prefer_corridor=True, protect_types=("KIT", "BALC"))
            # An upper floor has no building entrance -- any ENT tile
            # that emerged (the CORRIDOR_ENTRANCE mechanism doesn't know
            # it's on a floor where that's meaningless) gets converted
            # back to a plain corridor cell.
            grid = _strip_stray_entrances(grid, tileset)

        # Every room gets exactly one connection to a corridor -- a real
        # door for BED/BATH/STAIR/ENT, an open archway (no door) for
        # KIT/LIV -- and no direct connections to any other room. Run
        # AFTER the connectivity repair above, then repair once more --
        # closing a room's extra connections can occasionally orphan
        # whatever was only reachable through them. In the rare case
        # where full connectivity genuinely requires a second connection,
        # this final repair pass prioritizes connectivity (a hard
        # requirement) over the single-connection rule (a softer
        # preference), rather than looping forever between the two.
        grid, room_connection_results = enforce_room_connection_rules(grid, tileset)
        entrance_type_for_repair = "ENT" if is_ground else "STAIR"
        grid, repairs, _ = repair_connectivity(grid, tileset, entrance_room_type=entrance_type_for_repair,
                                                prefer_corridor=True, protect_types=("KIT", "BALC"))
        # Ensure this floor's corridor network forms one continuous
        # spine, not fragmented islands -- largely already guaranteed at
        # the planning stage (zoning.py's is_corridor_macro_connected),
        # this is a final safety check/patch for any residual case.
        grid, corridor_unified = unify_corridor_network(grid, tileset)

        # Record this floor's actual staircase macro position(s) directly
        # from the plan dict (not by reverse-computing from raw cell
        # coordinates -- N-WFC subgrids OVERLAP by one shared row/column
        # with their neighbors, so a naive r//C, c//C conversion
        # incorrectly attributes shared boundary cells to multiple
        # adjacent macro positions).
        if num_floors > 1 and stair_positions is None:
            macro_positions = [pos for pos, rt in plan.items() if rt == "STAIR"]
            stair_positions = macro_positions if macro_positions else None
        if num_floors > 1 and bath_positions is None:
            bath_macro_positions = [pos for pos, rt in plan.items() if rt == "BATH"]
            bath_positions = bath_macro_positions if bath_macro_positions else None

        rows, cols = len(grid), len(grid[0])
        stair_cells = [(r, c) for r in range(rows) for c in range(cols)
                       if tileset.tiles[grid[r][c]].room_type == "STAIR"]

        if is_ground:
            report = validate(grid, tileset)
            structural_valid = report.structural_valid
            reachable = report.functional_solvable
            unreachable = report.unreachable_cells
        else:
            structural_valid, _ = check_structural_validity(grid, tileset)
            all_reachable, unreachable, sealed, used = check_reachability(
                grid, tileset, entrance_cells=stair_cells if stair_cells else None
            )
            reachable = all_reachable

        actual_counts = count_actual_room_instances(grid, tileset)

        floors.append(FloorResult(
            floor_number=floor_number, grid=grid, is_ground_floor=is_ground,
            structural_valid=structural_valid, reachable=reachable,
            unreachable_cells=unreachable, room_counts_actual=actual_counts,
            stair_positions=stair_positions,
        ))

        if verbose:
            print(f"Floor {floor_number}: structural={structural_valid}, "
                  f"reachable={reachable}, rooms={actual_counts}")

    return floors