"""Loads the ResPlan dataset (Abouagour et al., 2025 -- 17,000 residential
floor plans, MIT/CC BY 4.0) and converts its room-connectivity graphs into
training examples for heuristic_model.ContextScorer.

Confirmed structure (from direct inspection of ResPlan.pkl):
  - Top level: dict of 17,000 plans, each itself a dict
  - Each plan's "graph" key holds a networkx.Graph
  - Graph nodes: named like "living_0", "bedroom_1", with attributes
    {"geometry": <Shapely Polygon>, "type": "<room type string>", "area": float}
  - Graph edges: (node_a, node_b, {"type": "via_door"}) or similar

We only need node "type" and edge existence -- geometry and area are
ignored entirely (this project's tileset works on typed rooms, not
precise vector shapes). networkx and shapely must still be importable
for pickle.load to succeed, since the pickled objects reference their
classes, but neither library's API is otherwise used here.

ResPlan's room-type taxonomy is broader than this project's 7-type
vocabulary (BED, BATH, KIT, LIV, COR, ENT, EXT). ROOM_TYPE_MAP below is
the explicit, documented mapping -- room types with no reasonable
equivalent (balcony, garden, parking, pool) are dropped rather than
force-mapped, since a wrong mapping would quietly corrupt the training
signal more than a dropped example would.
"""

import pickle
import random

ROOM_TYPE_MAP = {
    "living": "LIV",
    "kitchen": "KIT",
    "bedroom": "BED",
    "bathroom": "BATH",
    "inner": "COR",  # ResPlan's term for interior circulation/hallway space
    # Deliberately dropped (no equivalent in this project's tileset):
    # "balcony", "garden", "parking", "pool"
}


def load_resplan_graphs(pkl_path, max_plans=None):
    """Yields (nodes, edges) tuples in the same shape as
    graph_utils.extract_room_graph's output, one per ResPlan plan, with
    room types already mapped and unmappable rooms dropped. If dropping a
    room type would leave one side of an edge without a node, that edge
    is skipped too.
    """
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    items = data.values() if isinstance(data, dict) else data
    count = 0
    for plan in items:
        if max_plans is not None and count >= max_plans:
            break
        graph = plan.get("graph")
        if graph is None:
            continue

        node_type = {}
        for node, attrs in graph.nodes(data=True):
            raw_type = attrs.get("type")
            mapped = ROOM_TYPE_MAP.get(raw_type)
            if mapped is not None:
                node_type[node] = mapped

        edges = []
        for a, b, attrs in graph.edges(data=True):
            if a not in node_type or b not in node_type:
                continue
            edges.append({"a": a, "b": b, "edge_type": attrs.get("type", "unknown")})

        if not node_type:
            continue

        nodes = [{"id": n, "room_type": t, "size": 1} for n, t in node_type.items()]
        yield {"nodes": nodes, "edges": edges}
        count += 1


def build_examples_from_resplan(pkl_path, room_types, max_plans=None, max_examples=20000,
                                 max_per_pair=300, seed=0):
    """Builds (candidate_room_type, [neighbor_room_types], label) training
    examples directly from ResPlan, without materializing every graph in
    memory at once -- with 17,000 plans this keeps memory bounded.

    Unlike a naive set-based dedup, this preserves frequency information
    (a pairing observed in 4,000 plans contributes more training signal
    than one observed twice) while capping any single pair type at
    max_per_pair examples, so one extremely common pairing (e.g.
    living-bedroom) can't swamp the whole training set.

    Returns (examples, plans_used, pair_counts) where pair_counts is a
    dict {(room_a, room_b): count} for diagnostic reporting.
    """
    from collections import Counter

    pair_counts = Counter()
    plans_used = 0

    for graph in load_resplan_graphs(pkl_path, max_plans=max_plans):
        node_type = {n["id"]: n["room_type"] for n in graph["nodes"]}
        if not graph["edges"]:
            continue
        plans_used += 1
        for e in graph["edges"]:
            a, b = node_type[e["a"]], node_type[e["b"]]
            pair_counts[(a, b)] += 1
            pair_counts[(b, a)] += 1

    examples = []
    for (a, b), count in pair_counts.items():
        n = min(count, max_per_pair)
        examples.extend([(a, [b], 1.0)] * n)

    rng = random.Random(seed)
    all_pairs = [(a, b) for a in room_types for b in room_types]
    negative_pool = [p for p in all_pairs if p not in pair_counts]
    rng.shuffle(negative_pool)
    n_neg = min(len(examples), max(len(negative_pool), 1) * max_per_pair)
    neg_examples = []
    i = 0
    while len(neg_examples) < min(len(examples), n_neg) and negative_pool:
        a, b = negative_pool[i % len(negative_pool)]
        neg_examples.append((a, [b], 0.0))
        i += 1
        if i > len(examples) * 2:
            break
    examples.extend(neg_examples)

    rng.shuffle(examples)
    return examples[:max_examples], plans_used, dict(pair_counts)


def print_pair_frequency_report(pair_counts, top_n=15):
    """Prints the most frequent real room-type adjacencies -- useful to
    sanity-check (or revise) the hand-derived Layer 2 compatibility
    matrix in TILESET_DESIGN.md against actual architectural data."""
    ranked = sorted(pair_counts.items(), key=lambda kv: -kv[1])
    seen = set()
    print(f"\nTop {top_n} most frequent real room-type adjacencies (ResPlan):")
    shown = 0
    for (a, b), count in ranked:
        key = tuple(sorted([a, b]))
        if key in seen:
            continue
        seen.add(key)
        print(f"  {a:6s} -- {b:6s}  ({count} occurrences)")
        shown += 1
        if shown >= top_n:
            break