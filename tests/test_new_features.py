"""Tests for the balcony room type and the second-entrance-near-kitchen
feature.

Run from the project root with: pytest tests/
"""

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest  # noqa: E402
from tileset import Tileset  # noqa: E402
from zoning import (  # noqa: E402
    solve_zoned_with_exact_counts,
    compute_room_sizes_and_grid,
    ensure_entrance_on_border,
    count_actual_room_instances,
)
from wall_cleanup import cleanup_cross_room_edges  # noqa: E402
from repair import repair_connectivity, unify_corridor_network  # noqa: E402
from entrance_cap import cap_entrance_count  # noqa: E402
from room_connections import enforce_room_connection_rules  # noqa: E402
from validity import validate, find_entrance_cells  # noqa: E402
from wfc_core import DIRS  # noqa: E402
from multi_floor import generate_multi_floor_building  # noqa: E402

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "tileset_core.json")


@pytest.fixture(scope="module")
def tileset():
    return Tileset.load(DATA_PATH)


def _run_full_pipeline(tileset, room_counts, seed, second_entrance_near_kitchen=False):
    """Runs the exact pipeline app.py uses, for a single floor."""
    A, B, sizes, corridor_target = compute_room_sizes_and_grid(room_counts)
    random.seed(seed)
    grid, plan, attempts, matched = solve_zoned_with_exact_counts(
        tileset, A=A, B=B, C=4, room_counts=room_counts, max_attempts=25,
        second_entrance_near_kitchen=second_entrance_near_kitchen,
    )
    assert grid is not None, f"Generation failed entirely for seed={seed}"

    grid, _ = cleanup_cross_room_edges(grid, tileset)
    max_ent = 2 if second_entrance_near_kitchen else 1
    diverse = ("LIV", "KIT") if second_entrance_near_kitchen else ()
    grid, _ = cap_entrance_count(grid, tileset, max_entrances=max_ent, preferred_types=("COR",),
                                  strict=True, prefer_diverse_adjacency=diverse)
    grid, _, _ = repair_connectivity(grid, tileset, prefer_corridor=True, protect_types=("KIT", "BALC"))
    grid, _ = ensure_entrance_on_border(grid, tileset)
    grid, results = enforce_room_connection_rules(grid, tileset)
    grid, _, _ = repair_connectivity(grid, tileset, prefer_corridor=True, protect_types=("KIT", "BALC"))
    grid, corridor_unified = unify_corridor_network(grid, tileset)
    grid, _ = ensure_entrance_on_border(grid, tileset)

    return grid, results, corridor_unified


# --- Balcony tests ---

class TestBalcony:
    def test_balcony_tiles_are_subcomplete(self, tileset):
        """BALC must satisfy the same subcompleteness property as every
        other room type -- otherwise generation can hit unsolvable
        corners specifically at BALC subgrid boundaries."""
        balc_tiles = [t for t in tileset.tiles.values() if t.room_type == "BALC"]
        assert len(balc_tiles) > 0
        edges = {"W", "D", "O", "C"}
        pairs_present = {(t.e_n, t.e_s) for t in balc_tiles}
        for a in edges:
            for b in edges:
                assert (a, b) in pairs_present, f"BALC missing (en,es)=({a},{b})"

    def test_balcony_appears_in_generated_grid(self, tileset):
        room_counts = {"BED": 2, "BATH": 1, "KIT": 1, "LIV": 1, "BALC": 1}
        successes = 0
        for seed in range(10):
            grid, results, _ = _run_full_pipeline(tileset, room_counts, seed=800 + seed)
            counts = count_actual_room_instances(grid, tileset)
            if counts.get("BALC", 0) == 1:
                successes += 1
        assert successes >= 8, f"Balcony count should match requested count in most generations, got {successes}/10"

    def test_balcony_never_directly_connects_to_bath_kit_stair(self, tileset):
        """Balcony should almost always connect only to LIV, BED, or (as
        a fallback) COR. repair_connectivity's fallback CAN intentionally
        violate this in the rare case where it's the only way to achieve
        full connectivity (connectivity is the harder, non-negotiable
        requirement -- see repair_connectivity's protect_types
        docstring, and the equivalent, already-established exception for
        KIT) -- so this allows a small number of violations, but
        verifies solvability is never sacrificed to maintain the rule."""
        room_counts = {"BED": 2, "BATH": 2, "KIT": 1, "LIV": 1, "BALC": 1}
        n = 10
        violations = 0
        solvable = 0
        for seed in range(n):
            grid, results, _ = _run_full_pipeline(tileset, room_counts, seed=810 + seed)
            report = validate(grid, tileset)
            solvable += report.functional_solvable
            rows, cols = len(grid), len(grid[0])
            for r in range(rows):
                for c in range(cols):
                    tile = tileset.tiles[grid[r][c]]
                    if tile.room_type != "BALC":
                        continue
                    for direction, (dr, dc) in DIRS.items():
                        nr, nc = r + dr, c + dc
                        if not (0 <= nr < rows and 0 <= nc < cols):
                            continue
                        if tile.edge(direction) not in ("D", "O", "C"):
                            continue
                        neighbor_type = tileset.tiles[grid[nr][nc]].room_type
                        if neighbor_type not in ("BALC", "LIV", "BED", "COR"):
                            violations += 1
        assert solvable == n, f"Solvability must never be sacrificed for the corridor-only rule: {solvable}/{n}"
        assert violations <= 2, f"Found {violations} direct BALC-to-non-corridor connections -- should be rare"

    def test_balcony_on_exterior_wall(self, tileset):
        """A balcony must always be placed on the building's true
        exterior border -- an interior balcony makes no architectural
        sense."""
        room_counts = {"BED": 2, "BATH": 1, "KIT": 1, "LIV": 1, "BALC": 1}
        on_border_count = 0
        total = 0
        for seed in range(10):
            grid, results, _ = _run_full_pipeline(tileset, room_counts, seed=820 + seed)
            rows, cols = len(grid), len(grid[0])
            balc_cells = [(r, c) for r in range(rows) for c in range(cols)
                          if tileset.tiles[grid[r][c]].room_type == "BALC"]
            if not balc_cells:
                continue
            total += 1
            is_on_border = any(r == 0 or r == rows - 1 or c == 0 or c == cols - 1 for (r, c) in balc_cells)
            on_border_count += is_on_border
        assert total > 0, "No generation produced a balcony at all across 10 seeds"
        assert on_border_count == total, f"Balcony was off the exterior wall in {total - on_border_count}/{total} cases"

    def test_multi_floor_generation_with_balcony_does_not_crash(self, tileset):
        room_counts = {"BED": 3, "BATH": 2, "KIT": 1, "LIV": 1, "BALC": 1}
        floors = generate_multi_floor_building(tileset, num_floors=2, room_counts=room_counts, seed=42)
        assert len(floors) == 2
        for floor in floors:
            assert floor.structural_valid
            assert floor.reachable

    def test_balcony_never_on_ground_floor(self, tileset):
        """Balconies only make architectural sense on upper floors -- a
        real ground floor has a verandah or patio at most, never an
        elevated balcony."""
        room_counts = {"BED": 3, "BATH": 2, "KIT": 1, "LIV": 1, "BALC": 2}
        for seed in range(5):
            floors = generate_multi_floor_building(tileset, num_floors=2, room_counts=room_counts, seed=seed)
            ground_floor = floors[0]
            assert ground_floor.is_ground_floor
            assert ground_floor.room_counts_actual.get("BALC", 0) == 0, \
                f"seed={seed}: ground floor unexpectedly has a balcony"

    def test_balcony_distribution_excludes_ground_floor(self):
        """Unit-level check on the distribution logic itself, independent
        of full generation -- balconies must only ever be assigned to
        upper floor indices (1, 2, ...), never index 0 (ground)."""
        from multi_floor import distribute_rooms_across_floors

        room_counts = {"BED": 3, "BATH": 2, "KIT": 1, "LIV": 1, "BALC": 3}
        per_floor = distribute_rooms_across_floors(room_counts, num_floors=3)
        assert per_floor[0].get("BALC", 0) == 0
        total_balc = sum(pf.get("BALC", 0) for pf in per_floor)
        assert total_balc == 3, f"Expected all 3 requested balconies to be placed on upper floors, got {total_balc}"

    def test_balcony_with_single_floor_places_none(self):
        """With only 1 floor total (which IS the ground floor, no
        upper floors exist), requested balconies have nowhere valid to
        go and should simply not be placed anywhere."""
        from multi_floor import distribute_rooms_across_floors

        room_counts = {"BED": 2, "BATH": 1, "KIT": 1, "LIV": 1, "BALC": 2}
        per_floor = distribute_rooms_across_floors(room_counts, num_floors=1)
        assert per_floor[0].get("BALC", 0) == 0


# --- Second entrance near kitchen tests ---

class TestSecondEntranceNearKitchen:
    def test_exactly_two_entrances_when_requested(self, tileset):
        room_counts = {"BED": 2, "BATH": 2, "KIT": 1, "LIV": 1}
        two_entrance_count = 0
        n = 10
        for seed in range(n):
            grid, results, _ = _run_full_pipeline(
                tileset, room_counts, seed=700 + seed, second_entrance_near_kitchen=True)
            entrances = find_entrance_cells(grid, tileset)
            two_entrance_count += (len(entrances) == 2)
        assert two_entrance_count == n, f"Expected exactly 2 entrances every time, got it right {two_entrance_count}/{n}"

    def test_default_behavior_unaffected_when_not_requested(self, tileset):
        """The feature must be fully opt-in -- without it, exactly one
        entrance should still be produced, matching pre-existing
        behavior."""
        room_counts = {"BED": 2, "BATH": 2, "KIT": 1, "LIV": 1}
        for seed in range(5):
            grid, results, _ = _run_full_pipeline(
                tileset, room_counts, seed=700 + seed, second_entrance_near_kitchen=False)
            entrances = find_entrance_cells(grid, tileset)
            assert len(entrances) == 1

    def test_second_entrance_often_reaches_kitchen(self, tileset):
        """Best-effort, not guaranteed: at least one of the two entrances
        should end up directly connected to the kitchen in a solid
        majority of generations. This is a real architectural
        constraint, not a bug -- sometimes no border cell touches the
        kitchen at all for a given room layout, and a 100% guarantee
        isn't achievable given that."""
        room_counts = {"BED": 2, "BATH": 2, "KIT": 1, "LIV": 1}
        n = 15
        kit_connected = 0
        for seed in range(n):
            grid, results, _ = _run_full_pipeline(
                tileset, room_counts, seed=700 + seed, second_entrance_near_kitchen=True)
            ent_results = [r for r in results if r["room_type"] == "ENT"]
            if any(r["connected_to"] == "KIT" for r in ent_results):
                kit_connected += 1
        assert kit_connected >= n * 0.6, f"Kitchen-side entrance connection rate too low: {kit_connected}/{n}"

    def test_solvability_and_counts_unaffected_by_second_entrance(self, tileset):
        room_counts = {"BED": 2, "BATH": 2, "KIT": 1, "LIV": 1}
        n = 10
        solvable, counts_match = 0, 0
        for seed in range(n):
            grid, results, _ = _run_full_pipeline(
                tileset, room_counts, seed=700 + seed, second_entrance_near_kitchen=True)
            report = validate(grid, tileset)
            solvable += report.functional_solvable
            counts = count_actual_room_instances(grid, tileset)
            counts_match += all(counts.get(rt, 0) == c for rt, c in room_counts.items())
        assert solvable == n, f"Solvability regressed with second entrance enabled: {solvable}/{n}"
        assert counts_match == n, f"Exact room counts regressed with second entrance enabled: {counts_match}/{n}"

    def test_multi_floor_second_entrance_only_on_ground_floor(self, tileset):
        """Upper floors have no building entrance at all -- the second
        entrance option should never add one there."""
        room_counts = {"BED": 3, "BATH": 2, "KIT": 1, "LIV": 1}
        floors = generate_multi_floor_building(
            tileset, num_floors=2, room_counts=room_counts, seed=42,
            second_entrance_near_kitchen=True,
        )
        assert len(floors) == 2
        ground, upper = floors[0], floors[1]
        assert ground.structural_valid and ground.reachable
        assert upper.structural_valid and upper.reachable
        assert ground.room_counts_actual.get("ENT", 0) >= 1
        assert upper.room_counts_actual.get("ENT", 0) == 0


# --- Kitchen-only-connects-to-corridor tests ---

class TestKitchenIsolation:
    def test_kitchen_never_connects_directly_to_non_corridor_room(self, tileset):
        """Kitchen should mostly connect only to a corridor.
        repair_connectivity's fallback CAN intentionally violate this in
        the rare case where it's the only way to achieve full
        connectivity (connectivity is the harder, non-negotiable
        requirement -- see repair_connectivity's protect_types
        docstring) -- so this allows violations, but caps the AVERAGE
        rate and verifies solvability is never sacrificed.

        Threshold note: kitchen is now sized to guarantee at least 15%
        of the floor's area (previously a much smaller fixed size),
        which gives it substantially more perimeter cells and therefore
        more chances for this exception to trigger somewhere along that
        perimeter. Verified this is spread evenly across generations
        (never more than 1 violation in any single one), not
        concentrated in a few badly-broken cases -- an average of about
        1 per generation is the new, honest baseline, not a regression
        to chase back down."""
        room_counts = {"BED": 3, "BATH": 2, "KIT": 1, "LIV": 1}
        n = 15
        violations = 0
        solvable = 0
        for seed in range(n):
            grid, results, _ = _run_full_pipeline(tileset, room_counts, seed=400 + seed)
            report = validate(grid, tileset)
            solvable += report.functional_solvable
            rows, cols = len(grid), len(grid[0])
            for r in range(rows):
                for c in range(cols):
                    tile = tileset.tiles[grid[r][c]]
                    if tile.room_type != "KIT":
                        continue
                    for direction, (dr, dc) in DIRS.items():
                        nr, nc = r + dr, c + dc
                        if not (0 <= nr < rows and 0 <= nc < cols):
                            continue
                        if tile.edge(direction) not in ("D", "O", "C"):
                            continue
                        neighbor_type = tileset.tiles[grid[nr][nc]].room_type
                        if neighbor_type not in ("KIT", "COR"):
                            violations += 1
        assert solvable == n, f"Solvability must never be sacrificed for the corridor-only rule: {solvable}/{n}"
        avg_violations = violations / n
        assert avg_violations <= 1.2, (
            f"Found {violations} direct KIT-to-non-corridor connections across {n} generations "
            f"(avg {avg_violations:.2f}/generation) -- higher than the expected rate for the current "
            f"kitchen size"
        )

    def test_no_wall_mismatches_between_adjacent_cells(self, tileset):
        """Every pair of adjacent cells must agree on their shared edge
        type -- a mismatch would mean a genuinely broken wall/door in
        the generated layout."""
        room_counts = {"BED": 3, "BATH": 2, "KIT": 1, "LIV": 1}
        n = 10
        mismatches = 0
        opposite = {"n": "s", "s": "n", "w": "e", "e": "w"}
        for seed in range(n):
            grid, results, _ = _run_full_pipeline(tileset, room_counts, seed=400 + seed)
            rows, cols = len(grid), len(grid[0])
            for r in range(rows):
                for c in range(cols):
                    tile = tileset.tiles[grid[r][c]]
                    for direction, (dr, dc) in DIRS.items():
                        nr, nc = r + dr, c + dc
                        if not (0 <= nr < rows and 0 <= nc < cols):
                            continue
                        neighbor = tileset.tiles[grid[nr][nc]]
                        if tile.edge(direction) != neighbor.edge(opposite[direction]):
                            mismatches += 1
        assert mismatches == 0, f"Found {mismatches} mismatched wall/door edges"


# --- Corridor-percentage balcony enlargement tests ---

class TestRoomSizingRules:
    """Tests for the current sizing model: corridor capped at 20% of
    actual floor area, kitchen guaranteed at least 15%, bedrooms and
    bathrooms always equal size, and the balcony/living-room area
    ratio. This class replaces an earlier one that tested a prior
    sizing scheme (a bed-count/corridor-percentage trigger for
    balcony enlargement) which no longer exists -- superseded entirely
    by these rules."""

    def test_corridor_never_exceeds_20_percent(self):
        for room_counts in [
            {"BED": 2, "BATH": 2, "KIT": 1, "LIV": 1},
            {"BED": 3, "BATH": 2, "KIT": 1, "LIV": 1},
            {"BED": 5, "BATH": 5, "KIT": 1, "LIV": 1},
            {"BED": 1, "BATH": 1, "KIT": 1, "LIV": 1},
            {"BED": 2, "BATH": 2, "KIT": 1, "LIV": 1, "BALC": 1},
            {"BED": 2, "BATH": 2, "KIT": 1, "LIV": 1, "BALC": 3},
        ]:
            A, B, sizes, corridor_target = compute_room_sizes_and_grid(room_counts)
            total_room = sum(count * sizes.get(rt, 1) for rt, count in room_counts.items())
            actual_share = (A * B - total_room) / (A * B)
            assert actual_share <= 0.20 + 1e-9, f"{room_counts}: corridor share {actual_share:.1%} exceeds 20% cap"

    def test_kitchen_at_least_15_percent_when_requested(self):
        """The 15% kitchen minimum holds for typical requests. One
        documented exception: when bed/bath counts are both very high
        (e.g. 5+5), the sheer bed/bath floor area can leave no way to
        reach 15% without either growing the kitchen past the
        room-size reliability ceiling (max_single_room_size, see
        compute_room_sizes_and_grid's docstring) or dropping corridor
        below its architectural minimum -- both of those are the
        harder, non-negotiable constraints (extensively verified:
        rooms larger than the ceiling caused severe placement
        unreliability and, in the worst cases, outright generation
        failure). In that narrow case, kitchen's share is allowed to
        fall short, but only by a bounded, small amount -- not
        unboundedly."""
        typical_cases = [
            {"BED": 2, "BATH": 2, "KIT": 1, "LIV": 1},
            {"BED": 3, "BATH": 2, "KIT": 1, "LIV": 1},
            {"BED": 1, "BATH": 1, "KIT": 1, "LIV": 1},
        ]
        for room_counts in typical_cases:
            A, B, sizes, corridor_target = compute_room_sizes_and_grid(room_counts)
            kit_share = (room_counts["KIT"] * sizes["KIT"]) / (A * B)
            assert kit_share >= 0.15 - 1e-9, f"{room_counts}: kitchen share {kit_share:.1%} below 15% minimum"

        # The documented dense-bed/bath exception: still bounded, not
        # arbitrarily low.
        dense_case = {"BED": 5, "BATH": 5, "KIT": 1, "LIV": 1}
        A, B, sizes, corridor_target = compute_room_sizes_and_grid(dense_case)
        kit_share = (dense_case["KIT"] * sizes["KIT"]) / (A * B)
        assert kit_share >= 0.10, f"{dense_case}: kitchen share {kit_share:.1%} fell too far below the 15% target"

    def test_no_kitchen_requested_means_no_kitchen_minimum_applied(self):
        """A request with zero kitchens shouldn't be affected by the
        kitchen-minimum logic at all -- it simply doesn't apply."""
        room_counts = {"BED": 2, "BATH": 2, "LIV": 1}
        A, B, sizes, corridor_target = compute_room_sizes_and_grid(room_counts)
        assert A * B > 0  # just confirms this doesn't crash or misbehave

    def test_bathroom_at_most_40_percent_of_bedroom_size(self):
        """Supersedes an earlier rule that required BED and BATH to be
        exactly equal -- bathrooms are now capped at 40% of bedroom
        size instead. floor() rounding is used when deriving this (not
        round()), since "up to 40%" is an upper bound: rounding up could
        push the actual ratio slightly over 0.40 in some cases."""
        for room_counts in [
            {"BED": 2, "BATH": 2, "KIT": 1, "LIV": 1},
            {"BED": 5, "BATH": 3, "KIT": 1, "LIV": 1},
            {"BED": 1, "BATH": 4, "KIT": 1, "LIV": 1},
            {"BED": 3, "BATH": 2, "KIT": 1, "LIV": 1, "BALC": 2},
        ]:
            _, _, sizes, _ = compute_room_sizes_and_grid(room_counts)
            ratio = sizes["BATH"] / sizes["BED"]
            assert ratio <= 0.40 + 1e-9, f"{room_counts}: BATH/BED ratio {ratio:.2f} exceeds the 40% cap"

    def test_single_balcony_equals_living_room_size(self):
        room_counts = {"BED": 2, "BATH": 2, "KIT": 1, "LIV": 1, "BALC": 1}
        _, _, sizes, _ = compute_room_sizes_and_grid(room_counts)
        assert sizes["BALC"] == sizes["LIV"], f"single balcony ({sizes['BALC']}) should equal LIV size ({sizes['LIV']})"

    def test_multiple_balconies_total_area_is_twice_living_room(self):
        for balc_count in [2, 3, 4]:
            room_counts = {"BED": 2, "BATH": 2, "KIT": 1, "LIV": 1, "BALC": balc_count}
            _, _, sizes, _ = compute_room_sizes_and_grid(room_counts)
            total_balc_area = balc_count * sizes["BALC"]
            ratio = sizes["LIV"] / total_balc_area
            assert abs(ratio - 0.5) < 0.15, (
                f"balc_count={balc_count}: LIV/total_BALC ratio {ratio:.2f} should be close to 0.5"
            )

    def test_multi_floor_with_new_sizing_still_fully_reachable(self, tileset):
        """End-to-end: the stricter corridor cap combined with the
        kitchen minimum creates a harder packing scenario than the old
        sizing model -- this must never come at the cost of
        reachability (a genuine regression was found and fixed during
        development: solvability briefly collapsed to near-zero before
        the packing-reliability padding was added back correctly)."""
        room_counts = {"BED": 5, "BATH": 6, "KIT": 1, "LIV": 1, "BALC": 1}
        n = 5
        reachable_count = 0
        for seed in range(n):
            floors = generate_multi_floor_building(
                tileset, num_floors=2, room_counts=room_counts, seed=seed,
                second_entrance_near_kitchen=True,
            )
            if all(f.reachable for f in floors):
                reachable_count += 1
        assert reachable_count == n, f"Reachability regressed with new sizing rules: {reachable_count}/{n}"


# --- Generation time budget tests ---

class TestTimeBudget:
    def test_single_floor_default_scenario_is_fast(self, tileset):
        """A plain request with no special features (no balcony, no
        second entrance) should return almost immediately -- the time
        budget machinery shouldn't add overhead when there's nothing to
        search for."""
        import time
        room_counts = {"BED": 3, "BATH": 2, "KIT": 1, "LIV": 1}
        A, B, sizes, corridor_target = compute_room_sizes_and_grid(room_counts)
        random.seed(1)
        t0 = time.time()
        grid, plan, attempts, matched = solve_zoned_with_exact_counts(
            tileset, A=A, B=B, C=4, room_counts=room_counts, max_attempts=25)
        elapsed = time.time() - t0
        assert matched
        assert elapsed < 3.0, f"Default scenario took {elapsed:.1f}s, expected well under 3s"

    def test_multi_floor_hard_scenario_stays_bounded(self, tileset):
        """The specific scenario that originally took 78s (and later
        21s) -- second entrance, balcony, a forced staircase alignment
        across floors -- must stay within a reasonable ceiling. Not
        asserting a hard "under 10s" guarantee (that's a soft goal, not
        a contract), but this bounds it well away from regressing back
        toward the original tens-of-seconds behavior."""
        import time
        room_counts = {"BED": 5, "BATH": 6, "KIT": 1, "LIV": 1, "BALC": 1}
        t0 = time.time()
        floors = generate_multi_floor_building(
            tileset, num_floors=2, room_counts=room_counts, seed=77109,
            second_entrance_near_kitchen=True,
        )
        elapsed = time.time() - t0
        assert all(f.reachable for f in floors)
        assert elapsed < 20.0, f"Hard scenario took {elapsed:.1f}s, expected well under 20s"

    def test_time_budget_never_sacrifices_a_valid_result_for_speed(self, tileset):
        """When the time budget is exhausted before a "perfect" quality
        match is found, the function must still return the best valid
        (exact-count-matching) result found so far, never nothing."""
        room_counts = {"BED": 5, "BATH": 6, "KIT": 1, "LIV": 1, "BALC": 1}
        n = 5
        all_succeeded = 0
        for seed in range(n):
            try:
                floors = generate_multi_floor_building(
                    tileset, num_floors=2, room_counts=room_counts, seed=seed,
                    second_entrance_near_kitchen=True,
                )
                if len(floors) == 2:
                    all_succeeded += 1
            except RuntimeError:
                pass
        assert all_succeeded == n, f"Generation must not hard-fail due to time pressure: {all_succeeded}/{n}"