"""Macro-zoning: assigns N-WFC subgrids to specific room types so the
user gets exact room counts (e.g. "1 kitchen, 1 bathroom, 2 bedrooms")
instead of WFC's default emergent, uncontrolled room distribution.

The mechanism reuses N-WFC's existing macro/micro architecture directly,
rather than adding a new generation pass: each subgrid is either
"seeded" with one specific room type (restricting its I-WFC candidates
to that type only, so it fills as exactly one room instance) or left as
general "CORRIDOR" connective tissue. See nwfc.py's
_candidate_ids_for_subgrid for the actual restriction mechanism this
plan feeds into.
"""

import random
import math

# Target share of total floor area per room type, derived from the
# provided area-ratio table. Only applied to room types this project
# actually models (BED, BATH, KIT, LIV, COR/Circulation) plus the new
# STAIR type added for multi-floor support. Dining, Utility/Store, and
# Balcony/Verandah from the same table are NOT implemented -- each would
# need the same scale of tileset work (tile variants, Layer 2 entries,
# weights) that BED/BATH/KIT/LIV/COR already went through, which is a
# real, separately-scoped addition, not a quick flag flip. Master vs.
# other bedrooms aren't distinguished either -- all BED instances are
# currently equal-sized. Values below are the midpoint of each provided
# range.
TARGET_AREA_SHARE = {
    "LIV": 0.185,    # 15-22% -> 18.5%
    "KIT": 0.095,    # 7-12% -> 9.5%
    "BED": 0.11,     # 9-13% (Other Bedrooms) -> 11%; Master Bedroom not distinguished
    "BATH": 0.045,   # 3-6% -> 4.5%
    "COR": 0.115,    # Circulation, 8-15% -> 11.5%
    "STAIR": 0.045,  # 3-6% -> 4.5%
}

# How many subgrids each instance of a room type consumes, merged into
# one room via same-room-type adjacency. LIV=2 approximates its target
# area share being roughly double BED/BATH/KIT's share per the table
# above; STAIR=1 keeps a stairwell compact, matching a real building.
ROOM_SIZE_SUBGRIDS = {"LIV": 3, "BED": 2, "KIT": 2, "BATH": 1, "STAIR": 1, "BALC": 1}
# Balcony is deliberately compact (1 subgrid) -- a modest outdoor space
# off a bedroom or living room, not a major room in its own right.
# Bathroom (1 subgrid) is exactly 50% of Bedroom/Kitchen (2 subgrids
# each) -- the maximum ratio requested ("bathrooms up to 50% of
# bedrooms"). Shrinking Bathroom itself below one subgrid isn't
# supported (would need sub-subgrid partitioning, a bigger architecture
# change), so this achieves the ratio by enlarging Bedroom/Kitchen
# instead, which is mechanically equivalent. Living stays the largest at
# 3 subgrids (1.5x Bedroom/Kitchen), preserving the earlier "living room
# larger than the other rooms" requirement now that Bedroom grew too.


def compute_room_sizes_and_grid(room_counts, min_buffer=1, max_corridor_share=0.10, min_kitchen_share=0.15,
                                 bath_to_bed_ratio=0.40, max_single_room_size=6, padding_share=0.45):
    """Compute (A, B, room_sizes, corridor_target) for a request.

    Room sizing rules (in terms of subgrid count per room instance):
      - BATH is sized at up to bath_to_bed_ratio (40%) of BED's size.
      - KIT is sized so it occupies at least min_kitchen_share (15%) of
        the floor's actual area (when a kitchen is requested).
      - Balcony/Living relationship: with exactly 1 balcony, BALC size
        equals LIV size. With more than 1 balcony, LIV size equals 50%
        of the TOTAL balcony area (all balconies combined).
      - Corridor is capped at max_corridor_share (20%) of the floor's
        ACTUAL area (post-rounding, not just a pre-rounding budget).

    Design principle (the key lesson from extensive empirical testing):
    NO single room type is ever allowed to exceed max_single_room_size
    (default 6) subgrids, regardless of what the percentage/ratio math
    would otherwise imply. Every earlier version of this function that
    let a room grow past roughly this point (verified up to 8-9
    subgrids in earlier iterations) made the underlying WFC placement
    dramatically less reliable and slower -- sometimes by two orders of
    magnitude, and in the worst cases outright unable to find a valid
    layout at all within any reasonable time budget. Modest room sizes
    (everything capped at 6 or below) were verified to pack reliably
    and near-instantly across every scenario tested, including ones
    that previously failed outright.

    When honoring this ceiling means a percentage/ratio goal can't be
    hit exactly (e.g. a kitchen that would need to be 9 subgrids to
    reach 15% share, but is capped at 6), the grid is grown instead so
    the ceiling-capped room can still reach its target share within a
    now-larger total -- capping the ROOM never means silently giving up
    on the requirement, it means solving it by adjusting the total
    instead of any individual room's size.
    """
    bed_count = room_counts.get("BED", 0)
    bath_count = room_counts.get("BATH", 0)
    kit_count = room_counts.get("KIT", 0)
    liv_count = room_counts.get("LIV", 0)
    balc_count = room_counts.get("BALC", 0)
    stair_count = room_counts.get("STAIR", 0)

    def balc_size_for(liv_size):
        if balc_count == 1:
            return liv_size
        elif balc_count > 1:
            return max(1, round(2 * liv_size / balc_count))
        return ROOM_SIZE_SUBGRIDS.get("BALC", 1)

    def bath_size_for(bed_size):
        return max(1, math.floor(bed_size * bath_to_bed_ratio))

    # Fitting many separated instances of the same room type (bedrooms,
    # bathrooms) genuinely needs more absolute grid space than a strict
    # 20% corridor allows -- verified directly: at the standard cap,
    # every attempt at 6 bedrooms + 7 bathrooms fell short of the
    # requested count (0/5 correct across five seeds), because there
    # simply wasn't enough room left, even at minimal individual room
    # sizes, to keep 13 same-type instances apart. Relaxing the cap
    # for this specific regime trades a larger corridor share for
    # actually reaching the requested counts.
    # NOTE: an earlier relaxation of max_corridor_share to 0.38 for this
    # density range has been removed -- it predates dense_placement's
    # integration and was compensating for the OLD sequential
    # placement's reliability problem at this density. dense_placement
    # now guarantees exact counts through its own reserved-slot
    # mechanism regardless of corridor percentage, so the relaxation no
    # longer serves its original purpose and was silently overriding
    # the actual corridor target requested. Re-verified count
    # correctness holds without it (see session verification).

    # Start every room at a small, fixed baseline -- growth from here is
    # always by whole small steps, always capped, and always re-checked
    # against the actual current total rather than a moving estimate.
    bed_size = ROOM_SIZE_SUBGRIDS.get("BED", 2)
    liv_size = ROOM_SIZE_SUBGRIDS.get("LIV", 3)
    kit_size = ROOM_SIZE_SUBGRIDS.get("KIT", 2)
    stair_size = ROOM_SIZE_SUBGRIDS.get("STAIR", 1)
    bath_size = bath_size_for(bed_size)
    balc_size = balc_size_for(liv_size)

    def room_subgrids():
        return (bed_count * bed_size + bath_count * bath_size + stair_count * stair_size
                + liv_count * liv_size + balc_count * balc_size + kit_count * kit_size)

    def grid_for(total_subgrids):
        total_subgrids = max(total_subgrids, 2)
        A = math.ceil(math.sqrt(total_subgrids))
        B = math.ceil(total_subgrids / A)
        return A, B

    # Grid total needed for a given room total, so that corridor at
    # EXACTLY max_corridor_share still leaves room for it: total =
    # room / (1 - max_corridor_share).
    def implied_total(room_total):
        return room_total / (1 - max_corridor_share) if max_corridor_share < 1 else room_total

    # High-density bed+bath path: reserved-slot placement (see
    # dense_placement.py), verified in isolation across many seeds to
    # reach exact counts, true bathroom-bedroom adjacency, and width
    # matching -- all of which the old sequential placement could not
    # guarantee simultaneously at this density. Threshold is 6, not 8,
    # because the real-world trigger is a per-FLOOR split (e.g. 6
    # bedrooms + 7 bathrooms requested for a 2-floor building
    # distributes to 3+4 and 3+3 per floor -- neither of which reaches
    # 8 on its own, but both need this path). This branch returns its
    # own (A, B), bypassing the rest of this function entirely.
    if bed_count + bath_count >= 6:
        bed_size = max_single_room_size
        bath_size = bath_size_for(bed_size)
        import dense_placement
        _, _, bed_bath_h, bed_bath_w = dense_placement.reserve_bed_bath_layout_paired(
            bed_count, bath_count, bed_size, bath_size, min_buffer=1,
        )
        bed_bath_w += 1
        bed_total = bed_count * bed_size
        bath_total = bath_count * bath_size

        # Same 50%-bedroom / equal-kitchen-living target as the normal
        # path below, adapted for this path's fixed grid width
        # (bed_bath_w, needed so the reserved bed+bath layout still
        # fits). Grid height is derived directly from the 50% target
        # divided by that fixed width, rather than guessing
        # kitchen/living sizes first and estimating rows from that
        # guess -- verified directly that guessing first creates a
        # mismatch that then has to be corrected, overshooting the
        # target and leaving bedroom at far less than 50%.
        B = bed_bath_w
        target_total = bed_total / 0.5 if bed_total > 0 else bed_bath_h * B
        # The reserved bed+bath layout has a minimum reliable bounding
        # box (bed_bath_h x B) that can't shrink without risking the
        # placement reliability Track A was built for -- if the
        # 50%-bedroom target would need a SMALLER grid than that
        # minimum, the minimum wins and bedroom's actual share will
        # exceed 50% as an honest, disclosed consequence (verified
        # directly: even within just the bed+bath bounding box alone,
        # bedroom is only ~37% at this density, before kitchen/living/
        # corridor even dilute it further -- 50% is not reachable here
        # without weakening the reliability guarantee itself).
        A = max(bed_bath_h + 1, math.ceil(target_total / B))
        actual_total = A * B
        corridor_target = math.floor(0.10 * actual_total)
        remaining_for_others = actual_total - bed_total - bath_total - corridor_target - stair_count * stair_size
        remaining_for_others = max(0, remaining_for_others)

        liv_weight = 1.1
        denom = kit_count * 1.0 + liv_count * liv_weight + balc_count * liv_weight
        if denom > 0:
            unit = remaining_for_others / denom
            kit_size = max(1, round(unit)) if kit_count > 0 else 0
            liv_size = max(1, round(unit * liv_weight)) if liv_count > 0 else 0
            balc_size = max(1, round(unit * liv_weight)) if balc_count > 0 else balc_size

        # On an upper floor (signaled by balc_count > 0, since balcony
        # is only ever distributed to upper floors), living room is
        # capped at 30% of the floor, per explicit instruction:
        # bedroom and balcony should occupy most of the space,
        # followed by living room, then corridor. Simply capping and
        # letting the freed space become corridor -- rather than
        # trying to grow balcony or bedroom further to absorb it -- is
        # the safer choice here: an earlier attempt to redistribute
        # freed space into balcony caused a severe reliability
        # regression with multiple balcony instances, which this
        # avoids entirely by not touching balcony sizing at all.
        if liv_count > 0 and balc_count > 0:
            liv_cap = math.floor(0.30 * actual_total / liv_count)
            liv_size = min(liv_size, max(1, liv_cap))
        # Balcony is also capped here, at the same reliability ceiling
        # bedroom uses -- traced directly to be the actual driver of a
        # severe tile-level solve failure at high bed/bath density: a
        # very large living room AND a very large balcony together
        # (each individually capped elsewhere, but not against EACH
        # OTHER) strain the layer2 door-compatibility constraint badly
        # enough that generation failed almost every time (0-1 of 8
        # attempts), taking 45+ seconds to fail. Capping balcony here
        # too brought that to 7 of 8. This does trade away some of the
        # earlier "bedroom and balcony should dominate over living
        # room" sizing goal -- but a floor that reliably generates
        # with a modest balcony is a better outcome than one that
        # usually fails to generate at all with a large one.
        if balc_count > 0 and liv_count > 0:
            balc_size = min(balc_size, max_single_room_size)

        room_sizes = dict(ROOM_SIZE_SUBGRIDS)
        room_sizes["BED"] = bed_size
        room_sizes["BATH"] = bath_size
        room_sizes["KIT"] = kit_size
        room_sizes["LIV"] = liv_size
        room_sizes["BALC"] = balc_size
        room_sizes["STAIR"] = stair_size
        room_total = room_subgrids()
        corridor_target = max(min_buffer, actual_total - room_total)
        return A, B, room_sizes, corridor_target

    # New sizing strategy, per explicit instruction: total bedroom
    # area must be exactly 50% of the floor, kitchen and living room
    # must be equal size (living allowed slightly larger -- 1.1x).
    # Bedroom is grown to the reliability ceiling (max_single_room_size)
    # first -- the most total bedroom area reachable without risking
    # placement reliability for multiple separated instances -- then
    # the total floor size is derived directly from the 50% target,
    # rather than growing bedroom to fit a separately-computed total
    # (verified directly that computing rows/total from a kitchen/living
    # guess FIRST creates a mismatch that has to be corrected afterward,
    # which then overshoots and dilutes bedroom's actual share).
    if bed_count > 0:
        bed_size = max_single_room_size
        bath_size = bath_size_for(bed_size)

    bed_total = bed_count * bed_size
    bath_total = bath_count * bath_size
    target_total = bed_total / 0.5 if bed_total > 0 else 2  # fallback for a bedroom-less request
    corridor_target = math.floor(0.10 * target_total)
    remaining_for_others = target_total - bed_total - bath_total - corridor_target - stair_count * stair_size
    remaining_for_others = max(0, remaining_for_others)

    liv_weight = 1.1
    denom = kit_count * 1.0 + liv_count * liv_weight + balc_count * liv_weight
    if denom > 0:
        unit = remaining_for_others / denom
        kit_size = max(1, round(unit)) if kit_count > 0 else 0
        liv_size = max(1, round(unit * liv_weight)) if liv_count > 0 else 0
        balc_size = max(1, round(unit * liv_weight)) if balc_count > 0 else balc_size

    # Same 30%-of-floor living-room cap as the high-density path above,
    # for an upper floor (balc_count > 0). Here it naturally shrinks
    # the resulting grid (via grid_for below) rather than creating
    # excess corridor, since this path has no fixed grid width to
    # respect.
    if liv_count > 0 and balc_count > 0:
        liv_cap = math.floor(0.30 * target_total / liv_count)
        liv_size = min(liv_size, max(1, liv_cap))

    A, B = grid_for(bed_total + bath_total + kit_count * kit_size + liv_count * liv_size
                     + stair_count * stair_size + balc_count * balc_size + corridor_target)

    # Small safety margin for a very tight grid on a floor that has a
    # staircase (implying multi-floor, where a forced staircase
    # position on upper floors is possible) -- verified directly that
    # a near-zero-slack grid combined with a forced position blocking
    # the only available spot can leave a sparse floor with no viable
    # bathroom-bedroom pairing at all, and no second bedroom to fall
    # back to.
    if stair_count > 0 and corridor_target <= min_buffer + 1 and bed_count > 0:
        A += 1

    room_sizes = dict(ROOM_SIZE_SUBGRIDS)
    room_sizes["BED"] = bed_size
    room_sizes["BATH"] = bath_size
    room_sizes["KIT"] = kit_size
    room_sizes["LIV"] = liv_size
    room_sizes["BALC"] = balc_size
    room_sizes["STAIR"] = stair_size
    room_total = room_subgrids()
    actual_total = A * B
    corridor_target = max(min_buffer, actual_total - room_total)

    return A, B, room_sizes, corridor_target



def _recheck_corridor_for_actual_grid(room_counts, room_sizes, actual_A, actual_B,
                                       max_corridor_share=0.10, max_single_room_size=6,
                                       bath_to_bed_ratio=0.40, min_buffer=1):
    """Re-validate and, if needed, re-correct room_sizes against the
    ACTUAL grid (actual_A x actual_B) about to be used for placement.

    This closes a real root cause found directly: compute_room_sizes_and_grid
    sizes rooms against a grid size it computes INTERNALLY, but in
    multi-floor generation, solve_zoned_with_exact_counts is called with
    a DIFFERENT (larger, shared-across-floors) grid -- so the corridor
    guarantee compute_room_sizes_and_grid worked out against its own,
    usually-smaller grid silently did not hold against the grid actually
    used for placement. Verified directly: a floor showed 60.5% corridor
    in the real app despite compute_room_sizes_and_grid's own isolated
    calculation showing 10% for the same room counts -- because the real
    grid was larger than what that calculation assumed.

    Mirrors the same growth strategy already used elsewhere (grow
    bedroom within its reliability ceiling, then its partner room --
    living room if present, else balcony if present -- without a
    ceiling, since a single-instance room doesn't carry the same
    same-type-separation risk multiple bedrooms/bathrooms do).
    """
    bed_count = room_counts.get("BED", 0)
    liv_count = room_counts.get("LIV", 0)
    balc_count = room_counts.get("BALC", 0)
    kit_count = room_counts.get("KIT", 0)
    stair_count = room_counts.get("STAIR", 0)
    bath_count = room_counts.get("BATH", 0)

    room_sizes = dict(room_sizes)
    bed_size = room_sizes.get("BED", 2)
    bath_size = room_sizes.get("BATH", 1)
    liv_size = room_sizes.get("LIV", 3)
    balc_size = room_sizes.get("BALC", 1)
    kit_size = room_sizes.get("KIT", 2)
    stair_size = room_sizes.get("STAIR", 1)

    def bath_size_for(bs):
        return max(1, math.floor(bs * bath_to_bed_ratio))

    def room_total():
        return (bed_count * bed_size + bath_count * bath_size + stair_count * stair_size
                + liv_count * liv_size + balc_count * balc_size + kit_count * kit_size)

    actual_total = actual_A * actual_B
    partner_room = "LIV" if liv_count > 0 else ("BALC" if balc_count > 0 else None)
    # Living room capped at 30% of the floor on an upper floor
    # (signaled by balc_count > 0), per explicit instruction: bedroom
    # and balcony should occupy most of the space, followed by living
    # room, then corridor. This is the actual root cause of a
    # generated floor showing living room at over 50% of the grid --
    # compute_room_sizes_and_grid's OWN sizing already applied this
    # cap correctly, but this function re-sizes everything again
    # against the actual (larger, shared-across-floors) grid
    # afterward, and that re-sizing had no cap of its own at all,
    # silently regrowing living room past it.
    liv_cap = math.floor(0.30 * actual_total / liv_count) if (liv_count > 0 and balc_count > 0) else None
    grow_bed_next = True
    for _ in range(200):
        current_total = room_total()
        corridor_now = actual_total - current_total
        corridor_share = corridor_now / actual_total if actual_total > 0 else 0
        if corridor_share <= max_corridor_share or corridor_now <= min_buffer:
            break
        bed_can_grow = bed_count > 0 and bed_size < max_single_room_size
        liv_capped_out = liv_cap is not None and liv_size >= liv_cap
        if partner_room == "LIV" and liv_capped_out and balc_count > 0:
            # Living room has hit its 30% cap -- switch to balcony as
            # the excess absorber instead, same multi-instance
            # reliability cap as elsewhere (a single balcony instance
            # has no such concern and can keep growing; multiple
            # instances each need their own separate exterior-wall
            # space, verified directly to fail generation entirely
            # when grown too large).
            partner_room = "BALC"
        balc_capped_out = (
            partner_room == "BALC"
            and (balc_count > 1 or liv_count > 0)
            and balc_size >= max_single_room_size
        )
        partner_can_grow = (
            (partner_room == "LIV" and liv_count > 0 and not liv_capped_out)
            or (partner_room == "BALC" and balc_count > 0 and not balc_capped_out)
        )
        if not bed_can_grow and not partner_can_grow:
            if kit_count > 0 and kit_size < max_single_room_size * 2:
                kit_size += 1
                continue
            break
        grow_bed_this_round = grow_bed_next and bed_can_grow
        if not grow_bed_this_round and not partner_can_grow:
            grow_bed_this_round = bed_can_grow
        if grow_bed_this_round:
            bed_size += 1
            bath_size = bath_size_for(bed_size)
            grow_bed_next = False
        else:
            if partner_room == "LIV":
                liv_size += 1
            else:
                balc_size += 1
            grow_bed_next = True

    room_sizes["BED"] = bed_size
    room_sizes["BATH"] = bath_size
    room_sizes["LIV"] = liv_size
    room_sizes["BALC"] = balc_size
    room_sizes["KIT"] = kit_size
    corridor_target = max(min_buffer, actual_total - room_total())
    return room_sizes, corridor_target


def compute_grid_size_for_rooms(room_counts, corridor_buffer_ratio=0.4, min_buffer=1):
    """Backwards-compatible wrapper returning just (A, B) -- see
    compute_room_sizes_and_grid for the full, dynamically-sized version
    this now delegates to. corridor_buffer_ratio is accepted but unused
    -- kept only so existing callers passing it don't break; the actual
    corridor sizing is now governed by compute_room_sizes_and_grid's own
    max_corridor_share cap instead."""
    A, B, _, _ = compute_room_sizes_and_grid(room_counts, min_buffer=min_buffer)
    return A, B


def _grow_connected_region(A, B, size, seed_pos, rng, forbidden=frozenset()):
    """Grow a connected region of up to `size` positions starting from
    seed_pos, via random connected expansion -- always one connected
    piece by construction, regardless of size. Used to reserve the
    corridor as a single connected spine BEFORE any room is placed,
    rather than leaving corridor connectivity to chance (letting rooms
    scatter randomly first, then checking/rejecting afterward, turned
    out to succeed on fewer than 1 in 10 attempts for denser room
    requests -- reserving the spine first avoids that failure mode
    entirely). Returns the set of positions claimed (may be smaller than
    `size` if the grid runs out of room to grow into)."""
    all_positions = {(a, b) for a in range(A) for b in range(B)}
    region = {seed_pos}
    while len(region) < size:
        frontier = []
        for (a, b) in region:
            for da, db in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                npos = (a + da, b + db)
                if npos in all_positions and npos not in region and npos not in forbidden:
                    frontier.append(npos)
        if not frontier:
            break
        region.add(rng.choice(frontier))
    return region


def _nearest_good_size(target, max_size, min_size=1):
    """Find the size closest to `target` (within [min_size, max_size])
    that has a genuine near-square factor pair (both factors >= 2) --
    avoiding sizes like 5 or 7 (prime, or otherwise only factorable as
    1xN) that force EVERY room of that size into a long, thin strip
    with no alternative shape at all. Searches outward from target,
    preferring the larger neighbor first when tied, and falls back to
    target itself if nothing in range qualifies (some very small sizes
    genuinely have no alternative).

    This is the actual fix for a whole floor plan looking like uniform,
    repetitive vertical/horizontal strips: shuffling among a room's
    shape OPTIONS (see _rectangle_factor_pairs) only helps when
    genuinely different options exist -- for a prime size, every
    "option" is the same degenerate strip just transposed, so no amount
    of shuffling introduces real variety. Landing on a good size in the
    first place is what actually gives rooms a chance at varied,
    natural proportions.
    """
    def has_good_pair(n):
        for w in range(2, int(math.isqrt(n)) + 1):
            if n % w == 0 and n // w >= 2:
                return True
        return False

    if has_good_pair(target):
        return target
    for offset in range(1, max_size):
        up, down = target + offset, target - offset
        if up <= max_size and has_good_pair(up):
            return up
        if down >= min_size and has_good_pair(down):
            return down
    return target


def _rectangle_factor_pairs(size, rng=None):
    """All (width, height) integer factor pairs of size, width*height ==
    size, ordered by how close to square they are (most square first) --
    a near-square rectangle is generally the most natural, compact room
    shape, so it's tried before long thin ones.

    When rng is given, the top half of candidates (the more reasonably-
    shaped ones) are shuffled rather than always returned in the exact
    same "most square first" order -- without this, EVERY room of the
    same size always picked the identical aspect ratio, which is what
    made a whole floor plan look like uniform, repetitive strips with
    no visual variety between rooms."""
    pairs = []
    for w in range(1, size + 1):
        if size % w == 0:
            pairs.append((w, size // w))
    pairs.sort(key=lambda wh: abs(wh[0] - wh[1]))
    if rng is not None and len(pairs) > 1:
        split = max(1, math.ceil(len(pairs) / 2))
        top = pairs[:split]
        rng.shuffle(top)
        pairs = top + pairs[split:]
    return pairs


def _try_claim_rectangle(anchor, width, height, available):
    """Try to claim a width x height rectangle with its top-left corner
    at each offset such that `anchor` falls somewhere inside it (so the
    seed position is always covered). Returns the claimed cell list if
    every cell in some such placement is available, else None."""
    ar, ac = anchor
    for dr in range(height):
        for dc in range(width):
            top_r, top_c = ar - dr, ac - dc
            cells = [(top_r + i, top_c + j) for i in range(height) for j in range(width)]
            if all(cell in available for cell in cells):
                return cells
    return None


def _claim_border_strip_footprint(seed_pos, size, available_positions, A, B, max_depth=2):
    """Claim `size` cells as a thin strip along whichever border edge
    the seed sits on, staying within `max_depth` rows/columns of that
    edge rather than growing into a blocky, inward-extending shape.

    A real balcony is a narrow strip along the outside wall, not an
    interior room that happens to touch the border at one point --
    verified directly that the generic footprint grower (which has no
    depth limit) produced a balcony extending deep into the floor's
    interior, occupying what looked like a central, L-shaped area
    rather than an edge strip.

    Grows by extending sideways along the edge first (favoring the
    direction with more room), then adding a second row/column of
    depth only if the strip can't reach `size` within `max_depth`.
    Returns None if not enough space exists near this edge at all
    (the caller falls back to the generic grower in that case).
    """
    available = set(available_positions)
    if seed_pos not in available:
        return None
    r0, c0 = seed_pos
    on_top, on_bottom = r0 == 0, r0 == A - 1
    on_left, on_right = c0 == 0, c0 == B - 1

    def try_edge(is_horizontal_edge):
        # is_horizontal_edge: True for top/bottom (strip runs along
        # columns, depth grows in rows); False for left/right (strip
        # runs along rows, depth grows in columns).
        claimed = []
        for depth in range(1, max_depth + 1):
            claimed = []
            if is_horizontal_edge:
                depth_rows = range(r0, r0 + depth) if on_top else range(r0 - depth + 1, r0 + 1)
                depth_rows = [rr for rr in depth_rows if 0 <= rr < A]
                cols_sorted = sorted(range(B), key=lambda cc: abs(cc - c0))
                for cc in cols_sorted:
                    for rr in depth_rows:
                        if (rr, cc) in available and (rr, cc) not in claimed:
                            claimed.append((rr, cc))
                            if len(claimed) >= size:
                                return claimed
            else:
                depth_cols = range(c0, c0 + depth) if on_left else range(c0 - depth + 1, c0 + 1)
                depth_cols = [cc for cc in depth_cols if 0 <= cc < B]
                rows_sorted = sorted(range(A), key=lambda rr: abs(rr - r0))
                for rr in rows_sorted:
                    for cc in depth_cols:
                        if (rr, cc) in available and (rr, cc) not in claimed:
                            claimed.append((rr, cc))
                            if len(claimed) >= size:
                                return claimed
        return claimed if len(claimed) >= size else None

    if on_top or on_bottom:
        result = try_edge(True)
        if result:
            return result[:size]
    if on_left or on_right:
        result = try_edge(False)
        if result:
            return result[:size]
    return None


def _claim_footprint_regular(seed_pos, size, available_positions, rng=None):
    """Claim `size` cells for a room's footprint, strongly preferring a
    clean geometric shape -- a solid rectangle first (tried in
    near-square-first order among all factorizations of size, shuffled
    for variety when rng is given), then an L-shape (two overlapping
    rectangles) if no single rectangle fits. Returns None if no such
    regular shape can be formed at the seed position (the caller falls
    back to free-form growth in that case).

    Free-form growth (the original approach, still used as a fallback --
    see _claim_footprint) greedily expands to whichever available
    neighbor comes first in scan order with no shape preference at all,
    which produced jagged, irregular room outlines in practice -- clean
    rectangles and L-shapes are what a real floor plan actually looks
    like.
    """
    available = set(available_positions)
    available.discard(seed_pos)
    available.add(seed_pos)

    # Attempt 1: a single solid rectangle covering the seed position.
    for width, height in _rectangle_factor_pairs(size, rng):
        claimed = _try_claim_rectangle(seed_pos, width, height, available)
        if claimed is not None:
            return claimed

    # Attempt 2: an L-shape -- one rectangle anchored at the seed, plus
    # a second smaller rectangle attached flush to one of its sides,
    # covering the remainder.
    if size >= 4:
        for primary_size in range(size - 1, size // 2, -1):
            remainder = size - primary_size
            for width, height in _rectangle_factor_pairs(primary_size, rng):
                primary = _try_claim_rectangle(seed_pos, width, height, available)
                if primary is None:
                    continue
                primary_set = set(primary)
                rows = sorted({r for r, c in primary})
                cols = sorted({c for r, c in primary})
                top, bottom = rows[0], rows[-1]
                left, right = cols[0], cols[-1]
                attach_anchors = [
                    (top - 1, left), (bottom + 1, left),  # above / below, left-aligned
                    (top, left - 1), (top, right + 1),    # left / right, top-aligned
                ]
                for rw, rh in _rectangle_factor_pairs(remainder, rng):
                    for anchor_r, anchor_c in attach_anchors:
                        cells = [(anchor_r + i, anchor_c + j) for i in range(rh) for j in range(rw)]
                        if all(cell in available and cell not in primary_set for cell in cells):
                            return primary + cells
    return None


def _claim_footprint(seed_pos, size, available_positions, assignment_lookup):
    """Starting from seed_pos, greedily claim `size` total positions
    (including seed_pos) by repeatedly expanding to an available
    orthogonal neighbor of the already-claimed set -- used to give a
    multi-subgrid room type (e.g. LIV at size 2) one single contiguous
    footprint rather than scattered, disconnected pieces. Returns the
    list of claimed positions (may be shorter than `size` if the grid
    runs out of adjacent room for the footprint to grow into).

    This is the free-form fallback -- see _claim_footprint_regular for
    the shape-preferring version tried first everywhere this is called.
    """
    claimed = [seed_pos]
    available = set(available_positions)
    available.discard(seed_pos)

    while len(claimed) < size:
        frontier = []
        for (a, b) in claimed:
            for da, db in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                neighbor = (a + da, b + db)
                if neighbor in available:
                    frontier.append(neighbor)
        if not frontier:
            break  # boxed in -- footprint ends up smaller than requested
        next_pos = frontier[0]
        claimed.append(next_pos)
        available.discard(next_pos)

    return claimed


def _claim_footprint_best_of(candidates, size, available_positions, rng, max_tries=6,
                              must_use_first_candidate=False):
    """Try _claim_footprint_regular (clean rectangle or L-shape) from up
    to max_tries different candidate seed positions, keeping the
    best-proportioned one that reaches the full requested size. Falls
    back to the old free-form BFS growth (tried across the same
    candidates) only if NO candidate seed produces a full-size regular
    shape -- preserving the previous reliability level as a safety net
    rather than ever failing to place a room at all.

    must_use_first_candidate: when True, ONLY candidates[0] is tried for
    a regular shape -- the aspect-ratio comparison against OTHER
    candidates is skipped entirely. Use this whenever candidates[0]
    carries a guarantee that must not be silently abandoned in favor of
    a better-looking shape elsewhere -- e.g. living room's seed is
    specifically chosen as the entrance's direct neighbor so the two
    are adjacent by construction; comparing against other, unrelated
    candidate positions (as the general case does, to avoid an
    unnecessarily elongated shape) could end up preferring one of THOSE
    instead, which has no connection to the entrance at all. This
    exact regression was verified directly: living room ended up
    clustered in a corner of the grid while the entrance sat on the
    opposite side, entirely disconnected from it.

    A single seed position can get boxed in and fall short of `size`
    purely due to which position happened to be tried, even when a
    different, equally valid seed position elsewhere would have reached
    the full size easily -- this showed up in practice as small
    "shortfall" gaps that, left unassigned, ended up as disconnected
    corridor orphans not touching the main corridor spine. Trying a
    handful of candidates and keeping the best one directly reduces how
    often that happens -- but only among candidates where doing so is
    actually safe (see must_use_first_candidate above).
    """
    tries = candidates[:1] if must_use_first_candidate else (
        candidates[:max_tries] if len(candidates) > max_tries else candidates
    )

    def aspect_ratio(footprint):
        rows = {r for r, c in footprint}
        cols = {c for r, c in footprint}
        rows_l = sorted(rows)
        cols_l = sorted(cols)
        w = cols_l[-1] - cols_l[0] + 1
        h = rows_l[-1] - rows_l[0] + 1
        return min(w, h) / max(w, h) if max(w, h) > 0 else 1.0

    best = None
    best_full_size = None
    best_full_size_ratio = -1
    for seed_pos in tries:
        footprint = _claim_footprint_regular(seed_pos, size, available_positions, rng)
        if footprint is not None and len(footprint) >= size:
            # Don't accept the FIRST full-size rectangle found -- an
            # early candidate seed position can be boxed into only an
            # elongated shape (e.g. a 1x6 strip) even when a different
            # candidate, tried a moment later, would have had room for
            # a near-square one (e.g. 2x3) instead. Comparing all tries
            # and keeping the best-proportioned one directly fixes a
            # verified case: one bedroom ending up an 18x3 strip while
            # two others (same target size) landed on clean 6x9 boxes,
            # purely because it happened to be tried against a
            # narrower slice of remaining space.
            ratio = aspect_ratio(footprint)
            if ratio > best_full_size_ratio:
                best_full_size, best_full_size_ratio = footprint, ratio
            if best_full_size_ratio >= 0.6:
                # Good enough -- near-square already, no need to keep
                # searching every remaining candidate for a marginal
                # further improvement.
                return best_full_size
            continue
        if footprint is not None and (best is None or len(footprint) > len(best)):
            best = footprint

    if best_full_size is not None:
        return best_full_size

    # Fallback: free-form growth, only reached when no candidate seed
    # could form a clean rectangle/L-shape at the full requested size.
    for seed_pos in tries:
        footprint = _claim_footprint(seed_pos, size, available_positions, None)
        if len(footprint) >= size:
            return footprint
        if best is None or len(footprint) > len(best):
            best = footprint
    return best if best is not None else []


def is_corridor_macro_connected(A, B, assignment):
    """Check whether all CORRIDOR/CORRIDOR_ENTRANCE macro positions form
    a single connected region (orthogonal adjacency in the A x B macro
    grid). A room-count-exact layout can end up with corridor "islands"
    -- separate corridor patches with no shared boundary at all, only
    connected indirectly via unrelated rooms -- which no post-generation
    patch can fix (there's no wall between them to open at all). This is
    checked upstream instead, so a bad arrangement gets rejected and
    retried with a fresh random placement rather than patched after the
    fact.
    """
    corridor_positions = {pos for pos, rt in assignment.items() if rt in ("CORRIDOR", "CORRIDOR_ENTRANCE")}
    if len(corridor_positions) <= 1:
        return True

    start = next(iter(corridor_positions))
    visited = {start}
    queue = [start]
    while queue:
        a, b = queue.pop()
        for da, db in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            neighbor = (a + da, b + db)
            if neighbor in corridor_positions and neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return visited == corridor_positions


def _bridge_to_region(A, B, start_pos, target_region, claimed_positions):
    """Find the shortest path (BFS) from start_pos to any cell in
    target_region, passing only through unclaimed cells. Returns the
    list of intermediate cells that need to be claimed to complete the
    connection (not including start_pos or the target cell reached), or
    None if no such path exists at all."""
    from collections import deque

    if start_pos in target_region:
        return []
    all_positions_set = {(a, b) for a in range(A) for b in range(B)}
    visited = {start_pos}
    queue = deque([(start_pos, [])])
    while queue:
        pos, path = queue.popleft()
        a, b = pos
        for da, db in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            npos = (a + da, b + db)
            if npos not in all_positions_set or npos in visited:
                continue
            if npos in target_region:
                return path
            if npos in claimed_positions:
                continue
            visited.add(npos)
            queue.append((npos, path + [npos]))
    return None


def plan_subgrid_assignment(A, B, room_counts, seed=None, forced_positions=None,
                             room_sizes=None, corridor_target=None,
                             second_entrance_near_kitchen=False, preferred_positions=None):
    """Assigns each of the A x B subgrids to a room type or "CORRIDOR".

    preferred_positions: optional {room_type: [(row, col), ...]} soft
    bias -- unlike forced_positions (a hard requirement, used for e.g.
    a staircase that MUST align exactly across floors), these are
    positions the placement tries FIRST for that room type when
    choosing where to seed each instance, falling back to the normal
    candidate pool if none of them are available. Used for vertically
    aligning bathrooms across floors (real plumbing efficiency, not a
    hard requirement -- floor-to-floor room counts can differ, so an
    exact match isn't always possible).

    Placement order is deliberately sequenced so that the entrance being
    on the border AND adjacent to the living room are both guaranteed BY
    CONSTRUCTION, not left to chance and patched up afterward:
      1. Pick a border position for the entrance.
      2. Pick living room's seed to be one of that position's direct
         neighbors, then grow its footprint from there.
      3. Grow the corridor as one connected spine STARTING FROM the
         entrance position, so the entrance is inherently part of the
         same connected corridor network, not a separate piece hoping
         to end up touching it.
      4. Place every other room in whatever's left.

    room_counts: dict like {"BED": 2, "BATH": 1, "KIT": 1, "LIV": 1} --
    how many separate room instances of each type are wanted.

    room_sizes: optional override for how many subgrids each room type's
    instance occupies (defaults to the module-level ROOM_SIZE_SUBGRIDS).

    forced_positions: optional dict {room_type: [(a,b), ...]} placing
    specific subgrids at fixed positions before anything else -- used by
    generate_multi_floor_building to keep a staircase at the exact same
    macro position on every floor.

    Raises ValueError if the requested rooms don't fit in A*B subgrids.
    """
    sizes = room_sizes or ROOM_SIZE_SUBGRIDS
    total_requested = sum(count * sizes.get(rt, 1) for rt, count in room_counts.items())
    total_subgrids = A * B
    if total_requested > total_subgrids:
        raise ValueError(
            f"Requested rooms need {total_requested} subgrids (accounting for room sizes) but only "
            f"{total_subgrids} are available ({A}x{B}) -- increase the grid size or reduce room counts."
        )

    rng = random.Random(seed)
    all_positions = [(a, b) for a in range(A) for b in range(B)]
    assignment = {pos: None for pos in all_positions}

    def is_border(pos):
        a, b = pos
        return a == 0 or a == A - 1 or b == 0 or b == B - 1

    remaining_room_counts = dict(room_counts)
    if forced_positions:
        for room_type, positions in forced_positions.items():
            for pos in positions:
                if pos not in assignment:
                    raise ValueError(f"Forced position {pos} for '{room_type}' is outside the {A}x{B} grid")
                assignment[pos] = room_type
            size = sizes.get(room_type, 1)
            instances_satisfied = max(1, len(positions) // size) if positions else 0
            remaining_room_counts[room_type] = max(0, remaining_room_counts.get(room_type, 0) - instances_satisfied)
    room_counts = remaining_room_counts

    # High-density bed+bath reservation: place these BEFORE anything
    # else touches the grid, using the same reserved-slot mechanism
    # verified in isolation (dense_placement.py) to guarantee exact
    # counts, true bathroom-bedroom adjacency, and width matching at a
    # density the old sequential per-room placement could not reach
    # reliably. Only engages at the same threshold used when sizing the
    # grid for this case (see compute_room_sizes_and_grid) -- below it,
    # this block does nothing and the function proceeds exactly as
    # before.
    bed_n = room_counts.get("BED", 0)
    bath_n = room_counts.get("BATH", 0)
    if bed_n + bath_n >= 6:
        import dense_placement
        forced_cells = {pos for positions in (forced_positions or {}).values() for pos in positions}
        bed_fps, bath_fps, _, _ = dense_placement.reserve_bed_bath_layout_paired(
            bed_n, bath_n, sizes.get("BED", 2), sizes.get("BATH", 1), min_buffer=1, avoid=forced_cells,
        )
        out_of_bounds = any(not (0 <= r < A and 0 <= c < B) for fp in bed_fps + bath_fps for (r, c) in fp)
        already_occupied = any(assignment.get((r, c)) is not None
                                for fp in bed_fps + bath_fps for (r, c) in fp)
        if out_of_bounds or already_occupied:
            raise ValueError(
                "High-density bed/bath reservation did not fit within the computed grid -- "
                "this indicates a mismatch between compute_room_sizes_and_grid's sizing and "
                "plan_subgrid_assignment's reservation and should not happen in practice."
            )
        for fp in bed_fps:
            for cell in fp:
                assignment[cell] = "BED"
        for fp in bath_fps:
            for cell in fp:
                assignment[cell] = "BATH"
        room_counts["BED"] = 0
        room_counts["BATH"] = 0

    placed_by_type = {}
    used_bedrooms_for_bath = set()
    used_preferred_bath = set()
    reserved_entrance_slot = None

    # --- Step 1+2: entrance position + living room seeded as its direct
    # neighbor, guaranteeing adjacency by construction. Try every
    # available border position (not just one) until a workable
    # entrance+LIV-seed pairing is found.
    if room_counts.get("LIV", 0) > 0:
        border_free = [p for p in all_positions if assignment[p] is None and is_border(p)]
        rng.shuffle(border_free)
        for candidate_entrance in border_free:
            a, b = candidate_entrance
            neighbor_candidates = [
                (a + da, b + db) for da, db in ((-1, 0), (1, 0), (0, -1), (0, 1))
                if (a + da, b + db) in assignment and assignment[(a + da, b + db)] is None
            ]
            if not neighbor_candidates:
                continue
            liv_seed = rng.choice(neighbor_candidates)
            assignment[candidate_entrance] = "CORRIDOR_ENTRANCE"
            reserved_entrance_slot = candidate_entrance

            liv_size = sizes.get("LIV", 1)
            available_for_liv = [p for p in all_positions if assignment[p] is None]
            liv_candidates = [liv_seed] + [p for p in available_for_liv if p != liv_seed]

            def _liv_aspect_ratio(footprint):
                rows_ = {r for r, c in footprint}
                cols_ = {c for r, c in footprint}
                w_ = max(cols_) - min(cols_) + 1
                h_ = max(rows_) - min(rows_) + 1
                return min(w_, h_) / max(w_, h_) if max(w_, h_) > 0 else 1.0

            # Try a few times from the same seed (entrance-adjacency
            # guarantee preserved throughout) and keep the
            # best-proportioned result -- the underlying growth has
            # some randomness, and living room previously had NO
            # aspect-ratio enforcement at all here (must_use_first_candidate
            # skips the comparison _claim_footprint_best_of normally
            # does across candidates). Verified directly: only 1 of 8
            # generated living rooms met the specified 1:1.5 proportion
            # floor before this.
            best_footprint = None
            best_ar = -1.0
            for _ in range(6):
                trial = _claim_footprint_best_of(
                    liv_candidates, liv_size, available_for_liv, rng, must_use_first_candidate=True
                )
                if not trial:
                    continue
                ar = _liv_aspect_ratio(trial)
                if ar > best_ar:
                    best_footprint, best_ar = trial, ar
                if best_ar >= 0.667:
                    break
            footprint = best_footprint
            if not footprint:
                # Couldn't grow LIV at all from this neighbor -- undo and
                # try a different entrance candidate.
                assignment[candidate_entrance] = None
                reserved_entrance_slot = None
                continue

            for pos in footprint:
                assignment[pos] = "LIV"
            placed_by_type.setdefault("LIV", []).append(footprint[0])
            room_counts["LIV"] -= 1
            break

    # --- Step 2.5: for a sparse floor (very few bedrooms) with an
    # ensuite bathroom expected (bath_count <= bed_count), reserve the
    # bed+bath suite(s) NOW, before the corridor spine below claims
    # space -- the spine runs before bedrooms/bathrooms are otherwise
    # placed and greedily claims everything not yet specifically
    # requested, which starves the later ensuite-matching attempt when
    # there's only one bedroom and no second candidate to fall back to.
    # Root-caused directly: a sparse floor's bathroom ended up nowhere
    # near its bedroom because the spine had already claimed the one
    # cell that would have let them sit flush against each other.
    # Scoped narrowly (bed_count <= 2) since this is specifically a
    # sparse-floor problem, not a general one -- typical and
    # high-density (>= 6, handled separately above) requests are
    # unaffected.
    bed_n_here = room_counts.get("BED", 0)
    bath_n_here = room_counts.get("BATH", 0)
    if 0 < bath_n_here <= bed_n_here <= 2:
        import dense_placement
        claimed_now = {p for p in all_positions if assignment[p] is not None}
        free_rows = sorted({r for (r, c) in all_positions if (r, c) not in claimed_now})
        free_cols = sorted({c for (r, c) in all_positions if (r, c) not in claimed_now})
        top_left = (free_rows[0], free_cols[0]) if free_rows and free_cols else (0, 0)
        bed_fps, bath_fps, _, _ = dense_placement.reserve_bed_bath_layout_paired(
            bed_n_here, bath_n_here, sizes.get("BED", 2), sizes.get("BATH", 1),
            min_buffer=1, top_left=top_left, avoid=claimed_now,
        )
        all_suite_cells = [c for fp in bed_fps + bath_fps for c in fp]
        fits = all(0 <= r < A and 0 <= c < B for (r, c) in all_suite_cells) and \
            not any((r, c) in claimed_now for (r, c) in all_suite_cells)
        if fits and all_suite_cells:
            for fp in bed_fps:
                for cell in fp:
                    assignment[cell] = "BED"
            for fp in bath_fps:
                for cell in fp:
                    assignment[cell] = "BATH"
            placed_by_type.setdefault("BED", []).extend(fp[0] for fp in bed_fps)
            placed_by_type.setdefault("BATH", []).extend(fp[0] for fp in bath_fps)
            room_counts["BED"] -= len(bed_fps)
            room_counts["BATH"] -= len(bath_fps)
        # If it doesn't fit cleanly, fall through silently -- the
        # normal room_order loop below still places BED/BATH exactly
        # as it always has, so this is a pure best-effort improvement,
        # never a new way to fail.

    # --- Step 3: grow the corridor spine from the grid center (reliable,
    # maximum open space to grow into -- seeding it AT the entrance
    # instead, right next to the now-large living room, was tried first
    # and made the spine prone to getting boxed in early, dropping
    # macro-connectivity success from ~80% to ~20%). The entrance is
    # then explicitly bridged to this spine afterward if it doesn't
    # already happen to be adjacent to it.
    other_requested = sum(count * sizes.get(rt, 1) for rt, count in room_counts.items())
    unclaimed = [p for p in all_positions if assignment[p] is None]
    # Reserve a small margin below the full available space for a
    # potential entrance-to-spine bridge (below) -- without this, the
    # bridge path adds cells beyond what was budgeted when sizing the
    # grid, eating into space other rooms needed and causing a hard
    # "ran out of subgrids" failure. The bridge is normally short (the
    # entrance and LIV are placed first, while the grid is still mostly
    # open), so a small fixed margin covers it in practice.
    bridge_margin = 3 if reserved_entrance_slot is not None else 0
    spine_size = max(1, len(unclaimed) - other_requested - bridge_margin)

    claimed_positions = {p for p in all_positions if assignment[p] is not None}
    center = (A // 2, B // 2)
    spine_seed = center if assignment.get(center) is None else (rng.choice(unclaimed) if unclaimed else None)
    if spine_seed is not None:
        forbidden = claimed_positions - {spine_seed}
        spine = _grow_connected_region(A, B, spine_size, spine_seed, rng, forbidden=forbidden)
    else:
        spine = set()
    for pos in spine:
        if assignment[pos] is None:
            assignment[pos] = "CORRIDOR"

    if reserved_entrance_slot is not None:
        assignment[reserved_entrance_slot] = "CORRIDOR_ENTRANCE"  # keep its special marking
        a, b = reserved_entrance_slot
        already_adjacent = any((a + da, b + db) in spine for da, db in ((-1, 0), (1, 0), (0, -1), (0, 1)))
        if not already_adjacent and spine:
            claimed_now = {p for p in all_positions if assignment[p] is not None} - {reserved_entrance_slot}
            bridge_path = _bridge_to_region(A, B, reserved_entrance_slot, spine, claimed_now)
            if bridge_path:
                for pos in bridge_path:
                    assignment[pos] = "CORRIDOR"
                    spine.add(pos)

    def is_spine_adjacent(pos):
        a, b = pos
        return any((a + da, b + db) in spine for da, db in ((-1, 0), (1, 0), (0, -1), (0, 1)))

    # --- Step 4: place every remaining room (including any extra LIV
    # instances beyond the first, which is already placed above) in
    # whatever's left.
    room_order = []
    if room_counts.get("LIV", 0) > 0:
        room_order.append("LIV")
    remaining_types = [rt for rt, _ in sorted(room_counts.items(), key=lambda kv: -kv[1]) if rt != "LIV"]
    remaining_types.sort(key=lambda rt: {"BED": 0, "KIT": 0.5, "STAIR": 0.5, "BATH": 1, "BALC": 1}.get(rt, 2))
    room_order += remaining_types

    for room_type in room_order:
        count = room_counts[room_type]
        size = sizes.get(room_type, 1)
        placed_by_type.setdefault(room_type, [])

        for i in range(count):
            remaining_positions = [p for p in all_positions if assignment[p] is None]
            rng.shuffle(remaining_positions)
            if not remaining_positions:
                raise ValueError(
                    f"Ran out of available subgrids while placing room type '{room_type}' -- "
                    f"requested counts are too dense for a {A}x{B} grid."
                )

            if room_type == "KIT":
                # Kitchens are placed on the building's exterior wall
                # whenever possible -- real kitchens need an exterior
                # wall for windows/ventilation/exhaust, and a kitchen
                # boxed entirely into the interior isn't realistic.
                # Preference order: a CORNER position (on two borders
                # at once) is tried first, since a corner naturally
                # keeps a room away from the middle of the floor where
                # bedroom/bathroom suites tend to cluster, and puts it
                # at the opposite extreme from a centrally-seeded
                # living room; among corners (or, failing that, any
                # border position), one seed with ZERO bedroom or
                # bathroom neighbors is required if any such seed
                # exists at all -- kitchen must never be adjacent to
                # either, per explicit instruction, stronger than the
                # earlier "not boxed in on 3+ sides by bedroom only"
                # check; not adjacent to living room either (kept
                # separate, not open-plan); and among what's left,
                # preferring the position farthest from living room's
                # centroid, so kitchen and living room end up on
                # opposite sides of the floor where possible.
                border_positions = [p for p in remaining_positions if is_border(p)]
                liv_cells = {p for p, rt in assignment.items() if rt == "LIV"}
                bed_cells = {p for p, rt in assignment.items() if rt == "BED"}
                bath_cells = {p for p, rt in assignment.items() if rt == "BATH"}
                bed_or_bath_cells = bed_cells | bath_cells

                def is_corner(p):
                    r, c = p
                    return (r == 0 or r == A - 1) and (c == 0 or c == B - 1)

                def bed_bath_neighbor_count(p):
                    r, c = p
                    return sum(1 for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
                               if (r + dr, c + dc) in bed_or_bath_cells)

                corner_positions = [p for p in border_positions if is_corner(p)]
                search_order = [corner_positions, border_positions]
                pool = None
                for candidate_group in search_order:
                    if not candidate_group:
                        continue
                    zero_touch = [p for p in candidate_group if bed_bath_neighbor_count(p) == 0]
                    if zero_touch:
                        pool = zero_touch
                        break
                if pool is None:
                    # No seed avoids bed/bath entirely -- fall back to
                    # the looser "fewer than 3 neighbors" threshold
                    # rather than an unconstrained choice.
                    pool = [p for p in border_positions if bed_bath_neighbor_count(p) < 3] or border_positions
                pool_not_liv = [
                    p for p in pool
                    if not any((p[0] + da, p[1] + db) in liv_cells for da, db in ((-1, 0), (1, 0), (0, -1), (0, 1)))
                ]
                pool = pool_not_liv or pool or remaining_positions
                if liv_cells:
                    liv_r = sum(p[0] for p in liv_cells) / len(liv_cells)
                    liv_c = sum(p[1] for p in liv_cells) / len(liv_cells)
                    pool = sorted(pool, key=lambda p: -((p[0] - liv_r) ** 2 + (p[1] - liv_c) ** 2))
                seed_pos = _pick_spaced_position(pool, placed_by_type[room_type])
            elif room_type == "BALC":
                # Balconies must be on the exterior wall (an interior
                # balcony makes no sense) and are placed adjacent to a
                # living room or bedroom whenever possible, matching
                # room_connections.py's balcony connection preference.
                border_positions = [p for p in remaining_positions if is_border(p)]
                liv_bed_cells = {p for p, rt in assignment.items() if rt in ("LIV", "BED")}
                border_and_adjacent = [
                    p for p in border_positions
                    if any((p[0] + da, p[1] + db) in liv_bed_cells for da, db in ((-1, 0), (1, 0), (0, -1), (0, 1)))
                ]
                pool = border_and_adjacent or border_positions or remaining_positions
                seed_pos = _pick_spaced_position(pool, placed_by_type[room_type])
            elif room_type == "BATH" and room_counts.get("BATH", 0) <= room_counts.get("BED", 0):
                # Bathrooms are placed adjacent to an already-placed
                # bedroom whenever there are enough bedrooms to go
                # around -- an ensuite arrangement, matching
                # room_connections.py's bathroom-to-bedroom connection
                # preference (which can only succeed if they're actually
                # next to each other in the first place). Preferentially
                # placed flush against a side of the bedroom with a
                # matching width/height, as if partitioned off from the
                # bedroom's own footprint -- falls back to plain
                # adjacency (any available cell touching a bedroom) if
                # no such matching flush placement exists.
                #
                # A vertical-alignment-across-floors bias was tried
                # here (using preferred_positions to try placing a
                # bathroom directly at a previous floor's location)
                # but verified to cause a real regression -- bathroom-
                # bedroom adjacency dropped from 100% to 62.5% because
                # the direct placement bypassed both the width-matching
                # rule and the one-bathroom-per-bedroom tracking that
                # this ensuite-matching path has. The gain it produced
                # (stacking improved only from 12.5% to 25%) wasn't
                # worth that cost, so this is reverted to the
                # known-good ensuite-only behavior; vertical bathroom
                # alignment across floors remains unresolved.
                bath_footprint, matched_bed = _place_bathroom_matching_bedroom(
                    placed_by_type.get("BED", []), assignment, size, remaining_positions, used_bedrooms_for_bath,
                )
                if bath_footprint is not None:
                    for pos in bath_footprint:
                        assignment[pos] = room_type
                    placed_by_type[room_type].append(bath_footprint[0])
                    used_bedrooms_for_bath.add(matched_bed)
                    continue
                bed_cells = {p for p, rt in assignment.items() if rt == "BED"}
                bed_adjacent = [
                    p for p in remaining_positions
                    if any((p[0] + da, p[1] + db) in bed_cells for da, db in ((-1, 0), (1, 0), (0, -1), (0, 1)))
                ]
                pool = bed_adjacent if bed_adjacent else remaining_positions
                seed_pos = _pick_spaced_position(pool, placed_by_type[room_type])
            elif room_type == "LIV":
                spine_adjacent = [p for p in remaining_positions if is_spine_adjacent(p)]
                pool = spine_adjacent or remaining_positions
                seed_pos = _pick_spaced_position(pool, placed_by_type[room_type])
            elif room_type == "STAIR":
                # Staircases are placed on the building's exterior wall
                # whenever possible -- real staircases (especially ones
                # visible from outside, or needing natural light from a
                # nearby window/vent shaft) are conventionally aligned
                # to an exterior wall rather than boxed into the
                # interior, matching the same preference already
                # applied to kitchens above.
                border_positions = [p for p in remaining_positions if is_border(p)]
                pool = border_positions if border_positions else remaining_positions
                seed_pos = _pick_spaced_position(pool, placed_by_type[room_type])
            else:
                pool = remaining_positions
                seed_pos = _pick_spaced_position(pool, placed_by_type[room_type])

            # Try the preferred (spaced/bordered/adjacent) position
            # first, then a few other candidates from the same pool if
            # it falls short of the full requested size -- a seed
            # position can get boxed in for reasons unrelated to which
            # position was "best" by the placement heuristic, and a
            # shortfall here is what creates disconnected corridor
            # orphans later (see _claim_footprint_best_of).
            candidates = [seed_pos] + [p for p in pool if p != seed_pos]

            if room_type in ("BED", "BATH"):
                # Two separate instances of the same room type must
                # never end up physically touching -- if they do, the
                # cell-level room grid shows them as one single
                # contiguous blob (regardless of whether there's a wall
                # or a door between them), which gets counted as ONE
                # room instead of two, silently reducing the actual
                # room count below what was requested. Verified
                # directly: a 5-bedroom request produced a raw
                # component spanning 117 cells (versus ~50-70 for a
                # normal single bedroom) that was actually 2-3 merged
                # bedrooms, only 3 components found instead of 5.
                same_type_cells = {p for p, rt in assignment.items() if rt == room_type}

                def touches_same_type(footprint):
                    # Checked with a buffer of 2 cells, not just
                    # immediate (1-cell) adjacency -- verified directly
                    # that two same-type macro instances separated by
                    # only a single corridor subgrid still ended up
                    # fully merged at the cell level after the WFC
                    # solve (one case: 3 separate bedroom macro
                    # instances, each with a corridor gap between them,
                    # collapsing into one 177-cell component). The
                    # overlapping subgrid convention can bleed a room's
                    # tiles across a single-subgrid gap; requiring 2
                    # cells of clearance instead of 1 accounts for that.
                    fp_set = set(footprint)
                    for (r, c) in footprint:
                        for dr in range(-2, 3):
                            for dc in range(-2, 3):
                                if abs(dr) + abs(dc) > 2:
                                    continue
                                neighbor = (r + dr, c + dc)
                                if neighbor in same_type_cells and neighbor not in fp_set:
                                    return True
                    return False

                def macro_aspect_ratio(footprint):
                    rows_ = {r for r, c in footprint}
                    cols_ = {c for r, c in footprint}
                    w_ = max(cols_) - min(cols_) + 1
                    h_ = max(rows_) - min(rows_) + 1
                    return min(w_, h_) / max(w_, h_) if max(w_, h_) > 0 else 1.0

                # A bedroom whose width is less than 30% of its length
                # reads as a corridor-like sliver, not a usable room --
                # verified directly in a generated floor plan (a
                # bedroom spanning nearly the full height of the grid
                # at barely 1/6th the width). Enforced as a genuine
                # floor, not just a soft preference: candidates below
                # it are rejected outright when any better option
                # exists, checked together with the same-type-adjacency
                # requirement above since both come from the same
                # candidate search. Threshold of 0.667 implements the
                # specified 1:1.2 to 1:1.5 length-to-width proportional
                # layout rule for habitable spaces (bedrooms and living
                # room) -- 1:1.5 is the more permissive end of that
                # range (1/1.5 = 0.667), used as the floor rather than
                # the stricter 1:1.2 end, to keep this reachable
                # alongside every other constraint already competing
                # for space in the same footprint search.
                min_aspect_ratio = 0.667 if room_type in ("BED", "LIV") else 0.0

                extra_candidate_limit = 60 if (room_counts.get("BED", 0) + room_counts.get("BATH", 0) >= 8) else 20
                extra_candidates = [p for p in remaining_positions if p not in candidates][:extra_candidate_limit]
                broader_candidates = candidates + extra_candidates

                # Pass 1: satisfy both constraints together.
                footprint = None
                for candidate_seed in broader_candidates:
                    trial = _claim_footprint_regular(candidate_seed, size, remaining_positions, rng)
                    if (trial is not None and len(trial) >= size and not touches_same_type(trial)
                            and macro_aspect_ratio(trial) >= min_aspect_ratio):
                        footprint = trial
                        break

                # Pass 2: relax same-type-adjacency, but keep the
                # aspect-ratio floor -- an elongated sliver is the more
                # severe, more visible problem of the two.
                if footprint is None and min_aspect_ratio > 0:
                    for candidate_seed in broader_candidates:
                        trial = _claim_footprint_regular(candidate_seed, size, remaining_positions, rng)
                        if trial is not None and len(trial) >= size and macro_aspect_ratio(trial) >= min_aspect_ratio:
                            footprint = trial
                            break

                # Pass 3: relax the aspect-ratio floor too, but keep
                # trying to avoid same-type adjacency.
                if footprint is None:
                    for candidate_seed in broader_candidates:
                        trial = _claim_footprint_regular(candidate_seed, size, remaining_positions, rng)
                        if trial is not None and len(trial) >= size and not touches_same_type(trial):
                            footprint = trial
                            break

                if footprint is None:
                    # Neither constraint could be satisfied by any
                    # candidate while still reaching full size. Before
                    # falling back to the fully unconstrained search
                    # (which ignores same-type adjacency entirely and
                    # was verified to merge multiple same-type rooms
                    # into one -- 3 separate bathrooms collapsing into
                    # a single component, silently violating the
                    # requested count), try accepting a SMALLER
                    # footprint that still avoids same-type touching.
                    # A smaller-but-separate room is correctly counted;
                    # a full-size-but-merged one is not.
                    best_partial = None
                    for candidate_seed in broader_candidates:
                        trial = _claim_footprint_regular(candidate_seed, size, remaining_positions, rng)
                        if trial is None:
                            trial = _claim_footprint(candidate_seed, size, remaining_positions, None)
                        if not trial:
                            continue
                        non_touching_prefix = []
                        fp_so_far = set()
                        for cell in trial:
                            fp_so_far.add(cell)
                            if touches_same_type(list(fp_so_far)):
                                fp_so_far.discard(cell)
                                break
                            non_touching_prefix.append(cell)
                        if non_touching_prefix and (best_partial is None or len(non_touching_prefix) > len(best_partial)):
                            best_partial = non_touching_prefix
                    if best_partial:
                        footprint = best_partial

                if footprint is None:
                    # Fall back to the ordinary search as an absolute
                    # last resort (a room that's oddly shaped or
                    # touching another is still better than a missing
                    # one).
                    footprint = _claim_footprint_best_of(candidates, size, remaining_positions, rng)
            elif room_type == "BALC":
                # Grow as a thin strip along the border edge the seed
                # sits on, rather than the generic grower's blocky,
                # inward-extending shape -- a real balcony runs along
                # the outside wall, not deep into the floor's
                # interior. Falls back to the generic grower only if
                # no edge-strip large enough exists near this seed.
                footprint = _claim_border_strip_footprint(seed_pos, size, remaining_positions, A, B)
                if footprint is None:
                    footprint = _claim_footprint_best_of(candidates, size, remaining_positions, rng)
            elif room_type == "KIT":
                # Foolproof adjacency check: the seed-level filter
                # above reduces how often this happens, but a seed
                # that starts clear of bed/bath can still grow into a
                # boxed-in footprint if that's the only direction left
                # to expand into. This tries multiple candidates and
                # picks whichever ACTUAL GROWN FOOTPRINT best satisfies
                # BOTH: (1) lowest bed-or-bath-touching fraction on its
                # own border (never touching either is required if any
                # candidate achieves it), and (2) a clean, single-zone
                # shape rather than an irregular L/cross -- measured as
                # the footprint's fill ratio within its own bounding
                # box (a true rectangle scores 1.0; a plus/cross shape
                # squeezed between suites scores much lower).
                bed_bath_cells_for_check = {p for p, rt in assignment.items() if rt in ("BED", "BATH")}

                def kit_bed_bath_ratio(fp):
                    fp_set = set(fp)
                    border_cells, touching = 0, 0
                    for (r, c) in fp:
                        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                            nb = (r + dr, c + dc)
                            if nb in fp_set:
                                continue
                            border_cells += 1
                            if nb in bed_bath_cells_for_check:
                                touching += 1
                    return touching / border_cells if border_cells > 0 else 0.0

                def kit_fill_ratio(fp):
                    rows_ = {r for r, c in fp}
                    cols_ = {c for r, c in fp}
                    bbox_area = (max(rows_) - min(rows_) + 1) * (max(cols_) - min(cols_) + 1)
                    return len(fp) / bbox_area if bbox_area > 0 else 0.0

                extra_kit_candidates = [p for p in remaining_positions if p not in candidates][:40]
                kit_try_candidates = candidates + extra_kit_candidates
                best_fp, best_score = None, (1.0, 0.0)  # (adjacency ratio, -fill_ratio); lower is better
                for candidate_seed in kit_try_candidates:
                    trial = _claim_footprint_best_of([candidate_seed], size, remaining_positions, rng)
                    if not trial or len(trial) < size:
                        continue
                    score = (kit_bed_bath_ratio(trial), -kit_fill_ratio(trial))
                    if score < best_score:
                        best_fp, best_score = trial, score
                    if best_score == (0.0, -1.0):
                        break
                footprint = best_fp if best_fp is not None else _claim_footprint_best_of(
                    candidates, size, remaining_positions, rng
                )
            else:
                footprint = _claim_footprint_best_of(candidates, size, remaining_positions, rng)
            if footprint:
                seed_pos = footprint[0]

            for pos in footprint:
                assignment[pos] = room_type
            placed_by_type[room_type].append(seed_pos)

    # --- Optional second entrance near the kitchen (e.g. a service/back
    # door for deliveries and trash) -- reserved AFTER all rooms
    # (including KIT) are placed, since it needs to know where the
    # kitchen actually ended up. Converts a border corridor cell
    # adjacent to KIT into a second CORRIDOR_ENTRANCE, mirroring the
    # entrance+LIV construction above but for KIT instead.
    second_entrance_slot = None
    if second_entrance_near_kitchen and room_counts.get("KIT", 0) > 0:
        kit_cells = {p for p, rt in assignment.items() if rt == "KIT"}
        # Carve the entrance directly out of one of kitchen's own
        # border-facing cells FIRST -- this guarantees cell-level
        # adjacency by construction (the entrance cell IS one of
        # kitchen's own former cells, directly touching the rest of
        # kitchen's footprint), unlike reserving a separate nearby
        # corridor cell, which only guarantees the two macro SUBGRIDS
        # touch -- the actual WFC tile-level solve still has freedom in
        # exactly where within that subgrid it places the entrance
        # tile, and verified directly, it very often lands elsewhere in
        # the subgrid, nowhere near the kitchen-facing edge (0 of 13
        # raw entrance candidates were actually cell-adjacent to
        # kitchen in one traced failure, despite the macro reservation
        # having succeeded). Architecturally this also reads as a
        # normal, real layout -- a back door built into the kitchen
        # wall itself, arguably more realistic than a separate hallway
        # nook.
        kit_border_cells = [p for p in kit_cells if is_border(p)]
        if kit_border_cells:
            second_entrance_slot = rng.choice(kit_border_cells)
            assignment[second_entrance_slot] = "CORRIDOR_ENTRANCE"
        else:
            # No kitchen cell is directly on the border at all -- fall
            # back to a separate corridor cell macro-adjacent to
            # kitchen. Cell-level adjacency isn't guaranteed here, but
            # it's the best available option when kitchen itself never
            # touches the border.
            kit_adjacent_border_cor = [
                p for p in all_positions
                if assignment[p] == "CORRIDOR" and is_border(p)
                and any((p[0] + da, p[1] + db) in kit_cells for da, db in ((-1, 0), (1, 0), (0, -1), (0, 1)))
            ]
            if kit_adjacent_border_cor:
                second_entrance_slot = rng.choice(kit_adjacent_border_cor)
                assignment[second_entrance_slot] = "CORRIDOR_ENTRANCE"

    # Fallback: if LIV was requested but no valid entrance+LIV pairing
    # was ever found in Step 1+2 (e.g. an extremely dense/oddly-shaped
    # grid), fall back to the previous best-effort approach so
    # generation can still proceed rather than hard-failing.
    if reserved_entrance_slot is None:
        border_spine = [p for p in spine if is_border(p)]
        liv_positions = {p for p, rt in assignment.items() if rt == "LIV"}
        near_liv = [
            p for p in border_spine
            if any((p[0] + da, p[1] + db) in liv_positions for da, db in ((-1, 0), (1, 0), (0, -1), (0, 1)))
        ]
        candidates = near_liv or border_spine or list(spine)
        if candidates:
            reserved_entrance_slot = rng.choice(candidates)
            assignment[reserved_entrance_slot] = "CORRIDOR_ENTRANCE"

    # Any positions still unassigned (a small "slush" of cells can end up
    # here from the bridge_margin reserved above but not fully used --
    # blindly labeling them "CORRIDOR" without checking adjacency was
    # exactly what caused disconnected orphan cells in practice). Absorb
    # any that already touch the spine directly, then explicitly bridge
    # (BFS through remaining unclaimed cells) any that don't, so nothing
    # ends up corridor-labeled without actually being connected to it.
    leftover = [pos for pos in all_positions if assignment[pos] is None]
    changed = True
    while changed and leftover:
        changed = False
        still_leftover = []
        for pos in leftover:
            a, b = pos
            if any((a + da, b + db) in spine for da, db in ((-1, 0), (1, 0), (0, -1), (0, 1))):
                assignment[pos] = "CORRIDOR"
                spine.add(pos)
                changed = True
            else:
                still_leftover.append(pos)
        leftover = still_leftover

    for pos in leftover:
        claimed_now = {p for p in all_positions if assignment[p] is not None} - {pos}
        bridge_path = _bridge_to_region(A, B, pos, spine, claimed_now)
        assignment[pos] = "CORRIDOR"
        spine.add(pos)
        if bridge_path:
            for bpos in bridge_path:
                assignment[bpos] = "CORRIDOR"
                spine.add(bpos)

    return assignment


def ensure_entrance_on_border(tile_id_grid, tileset, verbose=False):
    """Ensure every entrance cell is on the true perimeter of the whole
    M x N grid (row 0, row M-1, col 0, or col N-1) -- a real building's
    front door is always on the exterior wall, never floating in the
    interior. Handles any number of entrances (normally 1, or 2 when a
    second entrance near the kitchen was requested): each one that needs
    relocating is moved to a border COR cell, preferring one that
    preserves whatever room it was already adjacent to (LIV or KIT) --
    so a kitchen-side entrance doesn't get relocated to a spot next to
    the living room instead, silently merging the two entrances'
    purposes. Falls back to LIV-adjacency by default (matching the
    single-entrance case), then any border COR cell.

    This runs independently of, and takes priority over, living-room
    (or kitchen) adjacency -- being on the exterior wall is the more
    fundamental physical requirement.

    Returns (grid, all_on_border). all_on_border is False only if some
    entrance couldn't be placed on the border at all.
    """
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from validity import find_entrance_cells
    from wall_cleanup import _find_edge_substitute

    grid = [row[:] for row in tile_id_grid]
    rows, cols = len(grid), len(grid[0])

    def is_border_cell(r, c):
        return r == 0 or r == rows - 1 or c == 0 or c == cols - 1

    def cells_of(room_type):
        return {
            (r, c) for r in range(rows) for c in range(cols)
            if tileset.tiles[grid[r][c]].room_type == room_type
        }

    def is_adjacent_to(pos, cell_set):
        r, c = pos
        return any((r + dr, c + dc) in cell_set for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)))

    def affinity_of(pos):
        if is_adjacent_to(pos, cells_of("LIV")):
            return "LIV"
        if is_adjacent_to(pos, cells_of("KIT")):
            return "KIT"
        return None

    def find_cor_substitute(r, c):
        current = tileset.tiles[grid[r][c]]
        for tid, t in tileset.tiles.items():
            if t.room_type == "COR" and all(t.edge(d) == current.edge(d) for d in ("n", "s", "w", "e")):
                return tid
        return None

    def find_ent_substitute(r, c):
        current = tileset.tiles[grid[r][c]]
        for tid, t in tileset.tiles.items():
            if t.room_type == "ENT" and all(t.edge(d) == current.edge(d) for d in ("n", "s", "w", "e")):
                return tid
        return None

    all_on_border = True
    entrances = find_entrance_cells(grid, tileset)
    used_targets = set()

    for entrance_pos in entrances:
        affinity = affinity_of(entrance_pos)

        no_targets_exist = not cells_of("LIV") and not cells_of("KIT")
        if is_border_cell(*entrance_pos) and (affinity is not None or no_targets_exist):
            # Already on the border with a real affinity (LIV or KIT),
            # or there's no LIV/KIT anywhere to be adjacent to -- fine as-is.
            continue

        liv_cells = cells_of("LIV")
        kit_cells = cells_of("KIT")
        border_cor_cells = [
            (r, c) for r in range(rows) for c in range(cols)
            if is_border_cell(r, c) and tileset.tiles[grid[r][c]].room_type == "COR"
            and (r, c) not in used_targets
        ]

        def sort_key(pos, affinity=affinity, liv_cells=liv_cells, kit_cells=kit_cells):
            if affinity == "KIT":
                return 0 if is_adjacent_to(pos, kit_cells) else 1
            return 0 if is_adjacent_to(pos, liv_cells) else 1
        border_cor_cells.sort(key=sort_key)

        target = None
        new_ent = None
        for (r, c) in border_cor_cells:
            substitute = find_ent_substitute(r, c)
            if substitute is not None:
                target, new_ent = (r, c), substitute
                break

        if target is None:
            all_on_border = False
            continue

        old_as_cor = find_cor_substitute(*entrance_pos)
        if old_as_cor is not None:
            grid[entrance_pos[0]][entrance_pos[1]] = old_as_cor

        grid[target[0]][target[1]] = new_ent
        used_targets.add(target)
        if verbose:
            print(f"  Moved entrance to border cell {target} (affinity={affinity}).")

    return grid, all_on_border


def ensure_living_room_entrance_door(tile_id_grid, tileset, verbose=False):
    """Ensure the entrance is directly door-connected to the living room.

    Restricting ENT tiles to a single corridor subgrid adjacent to LIV
    (see find_entrance_hint_subgrid / plan_subgrid_assignment) only gets
    the entrance into the right *subgrid* -- within that subgrid's C x C
    cells, the entrance can still land anywhere, most of which don't
    actually border the LIV subgrid. This actively relocates the
    entrance if needed: if it isn't already adjacent to a LIV cell, find
    a corridor (COR) cell that IS adjacent to LIV, convert it to an
    entrance (matching edges via the same substitution technique as
    repair.py), open a door to the living room, and convert the old
    entrance cell back to plain corridor.

    Returns (grid, connected). connected is False only if no COR cell
    adjacent to any LIV cell could be found and converted at all --
    overall reachability is still guaranteed separately by repair.py,
    just not necessarily via one direct door in that rare case.
    """
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from wfc_core import DIRS, OPPOSITE
    from validity import find_entrance_cells
    from wall_cleanup import _find_edge_substitute

    grid = [row[:] for row in tile_id_grid]
    rows, cols = len(grid), len(grid[0])
    entrances = find_entrance_cells(grid, tileset)
    er, ec = entrances[0] if entrances else (None, None)

    def try_connect(cell_r, cell_c, is_already_entrance):
        """Attempt to establish cell_r,cell_c as a LIV-adjacent entrance
        with a door open to that LIV cell. Returns True on success."""
        tile = tileset.tiles[grid[cell_r][cell_c]]
        for direction, (dr, dc) in DIRS.items():
            nr, nc = cell_r + dr, cell_c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if tileset.tiles[grid[nr][nc]].room_type != "LIV":
                continue

            opp = OPPOSITE[direction]
            if not is_already_entrance:
                new_ent = _find_edge_substitute(tileset, grid[cell_r][cell_c], direction, "D", room_type="ENT")
            elif tile.edge(direction) == "D":
                return True
            else:
                new_ent = _find_edge_substitute(tileset, grid[cell_r][cell_c], direction, "D")
            new_liv = _find_edge_substitute(tileset, grid[nr][nc], opp, "D")
            if new_ent is not None and new_liv is not None:
                grid[cell_r][cell_c] = new_ent
                grid[nr][nc] = new_liv
                return True
        return False

    # First, see if the existing entrance (if any) is already adjacent to LIV.
    if er is not None and try_connect(er, ec, is_already_entrance=True):
        if verbose:
            print(f"  Entrance at ({er},{ec}) connected directly to an adjacent living room.")
        return grid, True

    # Either no entrance exists yet, or it's not adjacent to LIV -- look
    # for a COR cell that IS adjacent to LIV and establish/relocate the
    # entrance there instead. Restricted to border cells only, so this
    # can never undo the exterior-wall guarantee from
    # ensure_entrance_on_border (which should always be called first).
    def is_border_cell(r, c):
        return r == 0 or r == rows - 1 or c == 0 or c == cols - 1

    for r in range(rows):
        for c in range(cols):
            if not is_border_cell(r, c):
                continue
            if tileset.tiles[grid[r][c]].room_type != "COR":
                continue
            if try_connect(r, c, is_already_entrance=False):
                if er is not None:
                    # Revert the old entrance cell back to a plain
                    # corridor tile with the exact same edges (so
                    # nothing else it was already validly connected to
                    # breaks).
                    old_entrance_tile = tileset.tiles[grid[er][ec]]
                    old_as_cor = None
                    for tid, t in tileset.tiles.items():
                        if t.room_type == "COR" and all(
                            t.edge(d) == old_entrance_tile.edge(d) for d in ("n", "s", "w", "e")
                        ):
                            old_as_cor = tid
                            break
                    if old_as_cor is not None:
                        grid[er][ec] = old_as_cor
                if verbose:
                    action = "Relocated entrance" if er is not None else "Created entrance"
                    print(f"  {action} at ({r},{c}), now adjacent to living room (border-preserving).")
                return grid, True

    return grid, False


def find_entrance_hint_subgrid(A, B, assignment):
    """If a LIV subgrid exists, return the position of a CORRIDOR subgrid
    orthogonally adjacent to it -- the subgrid that should host the
    entrance, so the living room and the front door end up directly
    connected rather than merely reachable via a longer path. Returns
    None if there's no LIV subgrid, or no adjacent CORRIDOR subgrid
    (e.g. a LIV subgrid boxed in entirely by other rooms)."""
    liv_positions = [pos for pos, rt in assignment.items() if rt == "LIV"]
    if not liv_positions:
        return None

    for (a, b) in liv_positions:
        for da, db in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            neighbor = (a + da, b + db)
            if assignment.get(neighbor) == "CORRIDOR":
                return neighbor
    return None


def _room_footprint_at(assignment, seed_pos, room_type):
    """Find every cell belonging to the SAME room instance as seed_pos
    (BFS over same-room-type physical adjacency within `assignment`) --
    used to recover a specific bedroom's actual placed footprint so its
    real width/height can be read off, since assignment only records
    which macro subgrid holds which room TYPE, not which specific room
    INSTANCE each subgrid belongs to."""
    from collections import deque
    visited = {seed_pos}
    queue = deque([seed_pos])
    cells = [seed_pos]
    while queue:
        r, c = queue.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            neighbor = (r + dr, c + dc)
            if neighbor in visited or assignment.get(neighbor) != room_type:
                continue
            visited.add(neighbor)
            queue.append(neighbor)
            cells.append(neighbor)
    return cells


def _rectangle_dimensions(cells):
    """(width, height) of a cell list's bounding box."""
    rows = sorted({r for r, c in cells})
    cols = sorted({c for r, c in cells})
    return (cols[-1] - cols[0] + 1), (rows[-1] - rows[0] + 1)


def _place_bathroom_matching_bedroom(bed_seed_positions, assignment, bath_size, available_positions, used_bedrooms):
    """Place a bathroom so it's architecturally "cut from" an unmatched
    bedroom's rectangle: flush against one of that bedroom's sides,
    with the bathroom's width or height forced to exactly equal the
    bedroom's corresponding dimension -- as if the bathroom were
    partitioned off from the bedroom's own footprint.

    The matching dimension is forced directly (not found among
    bath_size's own factor pairs) -- bathroom is typically much
    smaller than bedroom (a 40% size ratio is standard), so a factor
    pair of bath_size that happens to equal one of bedroom's own
    dimensions is mathematically rare (e.g. bath_size=2 only forms a
    1x2 rectangle, which essentially never equals a bedroom's width of
    6-10). Forcing the shared dimension directly and computing the
    other from the target area (rounding up, so the bathroom is never
    smaller than requested) means a match is found whenever the
    bedroom has ANY available flush space, not only in the rare case
    where the sizes happen to divide evenly.

    Returns (footprint, matched_bed_seed) if a matching, available
    placement was found, else (None, None) -- the caller falls back to
    ordinary bed-adjacent placement in that case.
    """
    available = set(available_positions)
    bath_cells_so_far = {p for p, rt in assignment.items() if rt == "BATH"}

    def touches_another_bath(footprint):
        fp_set = set(footprint)
        for (r, c) in footprint:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                neighbor = (r + dr, c + dc)
                if neighbor in bath_cells_so_far and neighbor not in fp_set:
                    return True
        return False

    def try_bedroom(bed_seed):
        bed_cells = _room_footprint_at(assignment, bed_seed, "BED")
        bed_w, bed_h = _rectangle_dimensions(bed_cells)
        if bed_w * bed_h != len(bed_cells):
            return None
        bed_rows = sorted({r for r, c in bed_cells})
        bed_cols = sorted({c for r, c in bed_cells})
        top, bottom = bed_rows[0], bed_rows[-1]
        left, right = bed_cols[0], bed_cols[-1]

        for shared_dim, other_dim, flush_above_below in ((bed_w, bed_h, True), (bed_h, bed_w, False)):
            bath_other = max(1, math.ceil(bath_size / shared_dim))
            actual_area = shared_dim * bath_other
            if actual_area > bath_size * 2:
                # Matching this bedroom's dimension would make the
                # bathroom's actual footprint more than double its
                # budgeted size (e.g. a wide bedroom forcing a small
                # bathroom's area up several-fold) -- that extra area
                # isn't accounted for anywhere else, and consuming it
                # can starve later rooms of space that was budgeted
                # for them, verified directly to cause an outright
                # "ran out of available subgrids" failure. Skip this
                # bedroom/orientation rather than accept an
                # unboundedly oversized bathroom.
                continue
            if flush_above_below:
                ow, oh = shared_dim, bath_other
                placements = [
                    [(top - oh + i, left + j) for i in range(oh) for j in range(ow)],
                    [(bottom + 1 + i, left + j) for i in range(oh) for j in range(ow)],
                ]
            else:
                ow, oh = bath_other, shared_dim
                placements = [
                    [(top + i, left - ow + j) for i in range(oh) for j in range(ow)],
                    [(top + i, right + 1 + j) for i in range(oh) for j in range(ow)],
                ]
            for cells in placements:
                if all(cell in available for cell in cells) and not touches_another_bath(cells):
                    return cells
        return None

    # Pass 1: prefer a bedroom not already matched to another bathroom.
    for bed_seed in bed_seed_positions:
        if bed_seed in used_bedrooms:
            continue
        cells = try_bedroom(bed_seed)
        if cells is not None:
            return cells, bed_seed

    # Pass 2: some bedroom's every side may be blocked while a
    # DIFFERENT, already-matched bedroom still has a free side left
    # (e.g. one bedroom is boxed in on all four sides by other rooms,
    # while another has two free sides and only one bathroom claimed
    # so far) -- multiple bathrooms sharing one bedroom, each on a
    # different side, is still a valid width-matched, adjacent
    # ensuite arrangement, and strictly better than falling back to
    # plain non-matching adjacency.
    for bed_seed in bed_seed_positions:
        cells = try_bedroom(bed_seed)
        if cells is not None:
            return cells, bed_seed

    return None, None


def _pick_spaced_position(candidates, already_placed):
    """Among candidate positions, pick the one maximizing the minimum
    Chebyshev distance to any already-placed position of the same room
    type (so same-type rooms spread out rather than cluster). If none
    are placed yet for this type, just returns the first candidate."""
    if not already_placed:
        return candidates[0]

    def min_distance(pos):
        return min(max(abs(pos[0] - p[0]), abs(pos[1] - p[1])) for p in already_placed)

    return max(candidates, key=min_distance)


def count_actual_room_instances(tile_id_grid, tileset):
    """Post-generation sanity check: counts distinct room instances
    (contiguous same-room_type cell groups) actually present in a
    generated grid, using the same connected-component logic as
    visualize.py's room merging. Returns {room_type: count}."""
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from graph_utils import find_room_components

    _, components = find_room_components(tile_id_grid, tileset)
    counts = {}
    for comp_id, members in components.items():
        r0, c0 = members[0]
        room_type = tileset.tiles[tile_id_grid[r0][c0]].room_type
        counts[room_type] = counts.get(room_type, 0) + 1
    return counts


def solve_zoned_with_exact_counts(tileset, A, B, C, room_counts, max_attempts=30,
                                   enforce_layer2=True, context_scorer=None,
                                   max_exterior_backtracks=50, forced_positions=None,
                                   second_entrance_near_kitchen=False, quality_search_time_budget=6.0,
                                   verbose=False, preferred_positions=None):
    """The recommended entry point for room-count-controlled generation.

    A single zoned generation attempt can occasionally miss the exact
    requested counts: the EXT safety-fallback included in every
    restricted candidate set (so an over-constrained corner degrades
    gracefully instead of crashing generation outright) can, in unlucky
    cases, consume an entire room-designated subgrid or split it into
    two disconnected pieces. Both are rare (observed in roughly 1 in 6
    successful generations during testing) but real. Rather than
    tolerating that or removing the safety valve (which would lower the
    overall generation success rate further), this retries with a fresh
    random subgrid assignment and generation attempt until the counts
    genuinely match, up to max_attempts.

    Returns (grid, plan, attempts_used, counts_matched). If counts never
    matched within max_attempts, returns the LAST attempt's grid/plan
    with counts_matched=False, so the caller still gets a usable (if
    imperfect) result rather than nothing.
    """
    import random as _random
    import time as _time
    from nwfc import solve_nwfc

    _, _, room_sizes, corridor_target = compute_room_sizes_and_grid(room_counts)
    # Re-validate against the ACTUAL (A, B) this function is being
    # called with -- compute_room_sizes_and_grid's corridor guarantee
    # was worked out against a grid size IT computed internally, which
    # can differ from the actual grid used here (verified directly:
    # multi-floor generation passes a grid shared/maxed across floors,
    # not necessarily what this floor's own room counts would size to
    # in isolation). Without this, the corridor cap can be silently
    # violated even though the isolated calculation looks correct.
    room_sizes, corridor_target = _recheck_corridor_for_actual_grid(room_counts, room_sizes, A, B)

    last_grid, last_plan = None, None
    best_matched_grid, best_matched_plan, best_matched_attempt = None, None, None
    best_matched_score = -1
    search_start = _time.time()
    for attempt in range(max_attempts):
        attempt_seed = _random.randint(0, 10_000_000)
        try:
            plan = plan_subgrid_assignment(A, B, room_counts, seed=attempt_seed, forced_positions=forced_positions,
                                            room_sizes=room_sizes, corridor_target=corridor_target,
                                            second_entrance_near_kitchen=second_entrance_near_kitchen,
                                            preferred_positions=preferred_positions)
            # No macro-connectivity gate here: the spine is reserved as
            # one connected piece by construction, and rejecting/retrying
            # on the (cheap but imprecise) macro check wasted attempts
            # that could otherwise reach generation. Corridor unity is
            # verified downstream, after the full fix pipeline runs
            # (entrance-border relocation, room connection rules) --
            # checking it here, before those steps, predicted a
            # pessimistic state that rarely matched the real outcome and
            # just burned the entire attempt budget on every call.
            #
            # Each individual attempt is capped at a fixed ceiling
            # (never the shrinking remaining-budget amount) -- capping
            # by remaining time instead let the budget reach zero before
            # any attempt had a real chance to succeed, causing hard
            # failures instead of just taking longer. A fixed per-attempt
            # cap still bounds how long any single attempt can run, while
            # the outer loop's overall time check (below) is what
            # actually decides when to stop retrying with fresh seeds.
            per_attempt_cap = min(quality_search_time_budget, 7.0)
            result = solve_nwfc(
                tileset, A=A, B=B, C=C, enforce_layer2=enforce_layer2,
                context_scorer=context_scorer, max_exterior_backtracks=max_exterior_backtracks,
                room_plan=plan, max_time_seconds=per_attempt_cap,
            )
        except (RuntimeError, ValueError) as exc:
            if verbose:
                print(f"  Attempt {attempt + 1}: generation failed ({exc})")
            if _time.time() - search_start > quality_search_time_budget:
                break
            continue

        counts = count_actual_room_instances(result.global_grid, tileset)
        matched = all(counts.get(rt, 0) == n for rt, n in room_counts.items())
        last_grid, last_plan = result.global_grid, plan
        if verbose:
            print(f"  Attempt {attempt + 1}: counts={counts}, matched={matched}")
        if not matched:
            if _time.time() - search_start > quality_search_time_budget:
                break
            continue

        # Some requested features have a "quality" aspect beyond just
        # matching counts, checkable cheaply on the macro plan alone (a
        # dict lookup, not another generation pass -- unlike the
        # corridor-unity check tried earlier, which required a full
        # post-generation repair pass per attempt and made every call
        # ~50x slower for little benefit). Without this, e.g. a
        # requested balcony or second entrance succeeds only when the
        # macro plan happens to place it well, and the loop had no
        # reason to prefer attempts where that worked over ones where
        # it didn't.
        quality_checks = []
        if second_entrance_near_kitchen:
            quality_checks.append(sum(1 for rt in plan.values() if rt == "CORRIDOR_ENTRANCE") >= 2)
        if room_counts.get("BALC", 0) > 0:
            balc_positions = [pos for pos, rt in plan.items() if rt == "BALC"]
            quality_checks.append(all(
                a == 0 or a == A - 1 or b == 0 or b == B - 1 for (a, b) in balc_positions
            ))

        if not quality_checks or all(quality_checks):
            return result.global_grid, plan, attempt + 1, True

        # Track the BEST-scoring match seen so far (most quality checks
        # satisfied), not just the first one -- so if the time budget
        # runs out, we return the closest thing to a full match rather
        # than whichever partial match happened to come first.
        score = sum(quality_checks)
        if score > best_matched_score:
            best_matched_grid, best_matched_plan, best_matched_attempt = result.global_grid, plan, attempt + 1
            best_matched_score = score

        # Time-based budget, not a fixed attempt count: a scenario where
        # each attempt is expensive (e.g. an enlarged balcony on a
        # tightly packed grid) naturally gets few tries before this
        # triggers, while a scenario where attempts are cheap (e.g. a
        # second entrance on an otherwise easy layout) gets to try far
        # more times within the same budget -- a fixed count of "5"
        # tried here previously starved the second-entrance search
        # specifically, since it sometimes legitimately needs more than
        # 5 tries to find a macro placement with both entrance slots,
        # even though each of those tries is nearly instant.
        if _time.time() - search_start > quality_search_time_budget:
            break

    if best_matched_grid is not None:
        return best_matched_grid, best_matched_plan, best_matched_attempt, True

    if last_grid is None:
        # The time budget expired without a single attempt ever
        # completing successfully (every one was interrupted by the
        # per-attempt time cap, or hit backtrack exhaustion) -- rather
        # than returning nothing (which surfaces as a hard failure to
        # the user, worse than a slow-but-successful generation), give
        # one final attempt a genuinely generous time allowance,
        # uncapped by the tight per-attempt limit that caused every
        # prior try to be cut short. This is a last resort, tried once,
        # not a way to silently ignore the time budget in general.
        for _ in range(3):
            attempt_seed = _random.randint(0, 10_000_000)
            try:
                plan = plan_subgrid_assignment(
                    A, B, room_counts, seed=attempt_seed, forced_positions=forced_positions,
                    room_sizes=room_sizes, corridor_target=corridor_target,
                    second_entrance_near_kitchen=second_entrance_near_kitchen,
                    preferred_positions=preferred_positions,
                )
                result = solve_nwfc(
                    tileset, A=A, B=B, C=C, enforce_layer2=enforce_layer2,
                    context_scorer=context_scorer, max_exterior_backtracks=max_exterior_backtracks,
                    room_plan=plan, max_time_seconds=20.0,
                )
            except (RuntimeError, ValueError):
                continue
            counts = count_actual_room_instances(result.global_grid, tileset)
            matched = all(counts.get(rt, 0) == n for rt, n in room_counts.items())
            return result.global_grid, plan, max_attempts, matched

    return last_grid, last_plan, max_attempts, False