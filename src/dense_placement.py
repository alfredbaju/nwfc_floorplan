"""Standalone reserved-slot placement for high-density same-type room
requests (many bedrooms and/or bathrooms).

This module has NO dependencies on the rest of the codebase and is not
called by anything yet -- it is built and verified entirely in
isolation first.

Version 2: rewritten for space efficiency after the v1 layout was
verified (via direct rendering) to be far too sparse for practical use
-- v1 padded a buffer margin onto EVERY slot's own dimensions (wasting
one extra buffer's worth of space at the edge of every row/column) and
stacked bedrooms and bathrooms as two independently-sized regions with
a mismatched shared width, leaving large unused margins. Both issues
are fixed here:
  - Buffers are now placed ONLY between adjacent slots (n slots need
    n-1 buffer gaps, not n), which was the larger source of waste.
  - Bedrooms and bathrooms are packed into ONE shared bounding width
    computed from both types together, rather than each computing its
    own width independently and taking the max (which wasted space
    whenever the two widths didn't match).
"""
import math


def _room_dims_for_size(size, min_aspect_ratio=0.30):
    """Best (width, height) for a room of `size` subgrids satisfying the
    minimum aspect ratio, preferring the closest-to-square factor pair
    and the smallest area that achieves it."""
    best = None
    for w in range(1, size + 1):
        for h in range(1, size + 1):
            if w * h < size:
                continue
            ratio = min(w, h) / max(w, h)
            if ratio < min_aspect_ratio:
                continue
            area = w * h
            if best is None or area < best[0] or (area == best[0] and abs(w - h) < abs(best[1] - best[2])):
                best = (area, w, h)
    if best is None:
        return size, 1
    return best[1], best[2]


def _best_layout(n, room_w, room_h, min_buffer, min_block_ratio=0.35):
    """Search all (rows, cols) with rows*cols >= n and rows*cols <= n +
    max(1, n//3) (a small padding allowance, not a jump to the next
    "nicer" composite count) for the one minimizing actual bounding-box
    area given this room's specific (possibly non-square) shape and the
    buffer cost -- among layouts whose OVERALL block also has a
    reasonable aspect ratio (>= min_block_ratio). Minimizing raw area
    alone can pick an extremely elongated block (verified directly: a
    29x2 arrangement for 3 rooms, which minimized total area but would
    look like a corridor, not a house, regardless of how little space
    it wasted). Falls back to the minimal-area option if no candidate
    meets the ratio (rare, only for very large n where every option is
    already elongated one way)."""
    candidates = []
    max_slots = n + max(1, n // 3)
    for rows in range(1, n + 1):
        cols = math.ceil(n / rows)
        if rows * cols > max_slots:
            continue
        width = cols * room_w + (cols - 1) * min_buffer
        height = rows * room_h + (rows - 1) * min_buffer
        area = width * height
        block_ratio = min(width, height) / max(width, height) if max(width, height) > 0 else 1.0
        candidates.append((area, block_ratio, rows, cols))

    reasonable = [c for c in candidates if c[1] >= min_block_ratio]
    pool = reasonable if reasonable else candidates
    best = min(pool, key=lambda c: c[0])
    return best[2], best[3]


def _packed_block_size(room_w, room_h, count, min_buffer):
    """Total (width, height) needed to pack `count` slots of room_w x
    room_h into the most compact valid grid for this room's shape, with
    min_buffer cells ONLY between adjacent slots (not around every
    slot's outer edge)."""
    rows, cols = _best_layout(count, room_w, room_h, min_buffer)
    width = cols * room_w + (cols - 1) * min_buffer
    height = rows * room_h + (rows - 1) * min_buffer
    return width, height, rows, cols


def reserve_bed_bath_layout(bed_count, bath_count, bed_size, bath_size,
                             min_buffer=2, min_aspect_ratio=0.30, top_left=(0, 0)):
    """Compute separated footprints for `bed_count` bedrooms and
    `bath_count` bathrooms together, sharing one bounding width and
    stacked with a single buffer gap between the two groups (not two
    independently-sized regions as in v1).

    Returns (bed_footprints, bath_footprints, total_height, total_width)
    -- both footprint lists are guaranteed full count, correct size,
    minimum aspect ratio, and mutual separation (within each type and
    across types) by construction.
    """
    bed_w, bed_h = _room_dims_for_size(bed_size, min_aspect_ratio) if bed_count > 0 else (0, 0)
    bath_w, bath_h = _room_dims_for_size(bath_size, min_aspect_ratio) if bath_count > 0 else (0, 0)

    bed_block_w, bed_block_h, _, _ = (
        _packed_block_size(bed_w, bed_h, bed_count, min_buffer) if bed_count > 0 else (0, 0, 0, 0)
    )
    bath_block_w, bath_block_h, _, _ = (
        _packed_block_size(bath_w, bath_h, bath_count, min_buffer) if bath_count > 0 else (0, 0, 0, 0)
    )

    shared_width = max(bed_block_w, bath_block_w)
    ox, oy = top_left

    bed_footprints = []
    if bed_count > 0:
        bed_rows_eff, bed_cols_eff = _best_layout(bed_count, bed_w, bed_h, min_buffer)
        idx = 0
        for gr in range(bed_rows_eff):
            for gc in range(bed_cols_eff):
                if idx >= bed_count:
                    break
                top = ox + gr * (bed_h + min_buffer)
                left = oy + gc * (bed_w + min_buffer)
                bed_footprints.append([(top + i, left + j) for i in range(bed_h) for j in range(bed_w)])
                idx += 1
            if idx >= bed_count:
                break
        bed_block_h = bed_rows_eff * bed_h + (bed_rows_eff - 1) * min_buffer

    bath_footprints = []
    if bath_count > 0:
        bath_top = ox + (bed_block_h + min_buffer if bed_count > 0 else 0)
        bath_rows_eff, bath_cols_eff = _best_layout(bath_count, bath_w, bath_h, min_buffer)
        idx = 0
        for gr in range(bath_rows_eff):
            for gc in range(bath_cols_eff):
                if idx >= bath_count:
                    break
                top = bath_top + gr * (bath_h + min_buffer)
                left = oy + gc * (bath_w + min_buffer)
                bath_footprints.append([(top + i, left + j) for i in range(bath_h) for j in range(bath_w)])
                idx += 1
            if idx >= bath_count:
                break
        bath_block_h = bath_rows_eff * bath_h + (bath_rows_eff - 1) * min_buffer

    total_height = (bed_block_h if bed_count > 0 else 0) \
        + (min_buffer if bed_count > 0 and bath_count > 0 else 0) \
        + (bath_block_h if bath_count > 0 else 0)
    total_width = shared_width

    return bed_footprints, bath_footprints, total_height, total_width


def reserve_bed_bath_layout_paired(bed_count, bath_count, bed_size, bath_size,
                                    min_buffer=1, min_aspect_ratio=0.30, top_left=(0, 0), avoid=None):
    """Like reserve_bed_bath_layout_packed, but preserves the
    bathroom-adjacent-to-bedroom architectural rule: for
    min(bed_count, bath_count) pairs, bundles one bedroom with one
    bathroom into a single combined "suite" rectangle (bathroom flush
    against the bedroom's side, height matched to the bedroom's own
    height, exactly as the earlier ensuite-matching logic did) before
    packing. Only bedrooms/bathrooms left over after pairing (when the
    two counts differ) are packed as independent, unpaired rectangles.

    Plain shelf-packing (reserve_bed_bath_layout_packed) treats every
    room as an independent rectangle to pack wherever it fits most
    efficiently, which is what makes it space-efficient -- but it also
    means a bathroom can end up nowhere near any bedroom at all.
    Verified directly: 0 of 6 generated floors had any bathroom
    actually adjacent to a bedroom under that approach. Packing
    pre-bundled suites instead guarantees the adjacency by
    construction, at the cost of some packing flexibility (a suite is
    a single, less flexible shape to place than its two rooms packed
    independently).

    avoid: optional set of (row, col) cells this layout must not use --
    e.g. a staircase pinned to a fixed macro position on an upper floor
    to align with the floor below. When a candidate placement would
    overlap one of these, the shelf position is advanced past it (or a
    new shelf is started if the whole row ahead is blocked) rather than
    colliding. Verified directly this was needed: a forced position
    anywhere other than row 0 of the grid collided with this layout's
    placement before this was added.

    Returns (bed_footprints, bath_footprints, total_height, total_width).
    """
    avoid = avoid or set()
    bed_w, bed_h = _room_dims_for_size(bed_size, min_aspect_ratio) if bed_count > 0 else (0, 0)
    bath_w, bath_h = _room_dims_for_size(bath_size, min_aspect_ratio) if bath_count > 0 else (0, 0)

    n_pairs = min(bed_count, bath_count)
    n_extra_bed = bed_count - n_pairs
    n_extra_bath = bath_count - n_pairs

    # Each suite: bathroom's height matched to the bedroom's height
    # (the same ensuite-matching principle as the earlier sequential
    # placement), bathroom flush against the bedroom's right side with
    # NO gap between them -- they are intentionally adjacent, not
    # separated same-type instances.
    suite_bath_w = max(1, math.ceil(bath_size / bed_h)) if n_pairs > 0 else 0
    suite_w = bed_w + suite_bath_w
    suite_h = bed_h

    # Units to pack: each pair is one combined rectangle; each leftover
    # bedroom or bathroom is its own independent rectangle. A buffer
    # applies BETWEEN different units (so suite-to-suite and
    # suite-to-leftover stay separated), never inside a suite.
    units = []
    for _ in range(n_pairs):
        units.append(("SUITE", suite_w, suite_h))
    for _ in range(n_extra_bed):
        units.append(("BED", bed_w, bed_h))
    for _ in range(n_extra_bath):
        units.append(("BATH", bath_w, bath_h))

    if not units:
        return [], [], 0, 0

    total_area = sum(w * h for _, w, h in units)
    target_width = max(w for _, w, h in units) + math.ceil(math.sqrt(total_area))

    units.sort(key=lambda u: -u[2])  # tallest first

    ox, oy = top_left
    shelf_y = ox
    shelf_x = oy
    shelf_height = 0
    bed_footprints, bath_footprints = [], []
    max_x_used = oy

    def collides(y, x, w, h):
        if not avoid:
            return False
        return any((y + i, x + j) in avoid for i in range(h) for j in range(w))

    for kind, w, h in units:
        if shelf_x != oy and shelf_x + w > oy + target_width:
            shelf_y += shelf_height + min_buffer
            shelf_x = oy
            shelf_height = 0

        # Advance past any obstacle in this row; if advancing runs off
        # the target width, start a fresh shelf below instead.
        guard = 0
        while collides(shelf_y, shelf_x, w, h) and guard < 200:
            shelf_x += 1
            if shelf_x + w > oy + target_width:
                shelf_y += max(shelf_height, h) + min_buffer
                shelf_x = oy
                shelf_height = 0
            guard += 1

        if kind == "SUITE":
            bed_cells = [(shelf_y + i, shelf_x + j) for i in range(bed_h) for j in range(bed_w)]
            bath_cells = [(shelf_y + i, shelf_x + bed_w + j) for i in range(suite_h) for j in range(suite_bath_w)]
            bed_footprints.append(bed_cells)
            bath_footprints.append(bath_cells)
        elif kind == "BED":
            bed_footprints.append([(shelf_y + i, shelf_x + j) for i in range(h) for j in range(w)])
        else:
            bath_footprints.append([(shelf_y + i, shelf_x + j) for i in range(h) for j in range(w)])

        shelf_x += w + min_buffer
        shelf_height = max(shelf_height, h)
        max_x_used = max(max_x_used, shelf_x - min_buffer)

    total_height = shelf_y + shelf_height - ox
    total_width = max_x_used - oy

    return bed_footprints, bath_footprints, total_height, total_width


def reserve_bed_bath_layout_packed(bed_count, bath_count, bed_size, bath_size,
                                    min_buffer=2, min_aspect_ratio=0.30, top_left=(0, 0)):
    """Like reserve_bed_bath_layout, but packs bedrooms and bathrooms
    together as one shared pool of rectangles (shelf-packing), instead
    of as two independently-sized stacked blocks. Stacking two blocks
    wastes space whenever one type's natural width doesn't match the
    other's; interleaving lets a row of tall bedrooms sit directly
    beside a row of short bathrooms, sharing the same shelf, which
    verified directly to meaningfully shrink the total bounding area
    for a realistic bed/bath mix.

    Algorithm: classic shelf-packing. All rectangles (both types) are
    sorted tallest-first. Rectangles are placed left-to-right on the
    current shelf (row) until the target width would be exceeded, then
    a new shelf starts below the tallest rectangle placed on the
    previous shelf. The target width is chosen to roughly match a
    reasonable overall aspect ratio for the combined block.

    Returns (bed_footprints, bath_footprints, total_height, total_width).
    """
    bed_w, bed_h = _room_dims_for_size(bed_size, min_aspect_ratio) if bed_count > 0 else (0, 0)
    bath_w, bath_h = _room_dims_for_size(bath_size, min_aspect_ratio) if bath_count > 0 else (0, 0)

    rects = [("BED", bed_w, bed_h) for _ in range(bed_count)] + \
            [("BATH", bath_w, bath_h) for _ in range(bath_count)]
    if not rects:
        return [], [], 0, 0

    total_area = sum(w * h for _, w, h in rects)
    # Target width for a roughly-square overall block, nudged wider
    # since shelf-packing tends to leave more slack in height than
    # width -- verified empirically to land close to a good ratio.
    target_width = max(w for _, w, h in rects) + math.ceil(math.sqrt(total_area))

    rects.sort(key=lambda r: -r[2])  # tallest first

    ox, oy = top_left
    shelf_y = ox
    shelf_x = oy
    shelf_height = 0
    bed_footprints, bath_footprints = [], []
    max_x_used = oy

    for kind, w, h in rects:
        if shelf_x != oy and shelf_x + w > oy + target_width:
            # Start a new shelf below the current one.
            shelf_y += shelf_height + min_buffer
            shelf_x = oy
            shelf_height = 0

        cells = [(shelf_y + i, shelf_x + j) for i in range(h) for j in range(w)]
        if kind == "BED":
            bed_footprints.append(cells)
        else:
            bath_footprints.append(cells)

        shelf_x += w + min_buffer
        shelf_height = max(shelf_height, h)
        max_x_used = max(max_x_used, shelf_x - min_buffer)

    total_height = shelf_y + shelf_height - ox
    total_width = max_x_used - oy

    return bed_footprints, bath_footprints, total_height, total_width


def verify_separation(footprints, min_buffer=1):
    """Check that no two footprints in the list are within min_buffer
    cells (Manhattan distance) of each other."""
    for i, fp_a in enumerate(footprints):
        for j, fp_b in enumerate(footprints):
            if i >= j:
                continue
            for (r1, c1) in fp_a:
                for (r2, c2) in fp_b:
                    if abs(r1 - r2) + abs(c1 - c2) <= min_buffer:
                        return False
    return True