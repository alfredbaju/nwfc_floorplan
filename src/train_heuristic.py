"""Build a training set for the learned selection heuristic and train it.

Data sources, honestly labeled:
  1. floor_plan_1.json -- one real, hand-annotated reference plan (the
     only annotated ground-truth data currently available in this
     project; see project history -- annotating more plans, or pulling
     ResPlan's graph-annotated dataset, is the next real step to improve
     on this).
  2. A batch of N-WFC-generated plans with Layer 2 enforcement active --
     not real architectural data, but every edge in them is guaranteed
     Layer-2-compatible by construction, so they're a reasonable
     synthetic supplement while (1) is so small.

Positive examples: room-type pairs actually observed adjacent (via a
non-wall edge) in the source data. Negative examples: room-type pairs
randomly sampled that were NOT observed adjacent, standard negative
sampling for link-prediction-style training.
"""

import json
import random
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from tileset import Tileset  # noqa: E402
from nwfc import solve_nwfc  # noqa: E402
from graph_utils import extract_room_graph  # noqa: E402
from heuristic_model import ContextScorer  # noqa: E402
from load_resplan import build_examples_from_resplan, print_pair_frequency_report  # noqa: E402

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "tileset_core.json")
ANNOTATED_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "floor_plan_1.json")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "context_scorer.npz")
RESPLAN_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "resplan", "ResPlan.pkl")

N_GENERATED_PLANS = 40
RESPLAN_MAX_PLANS = None  # None = use all 17,000; set a smaller number to test faster first


def graph_from_annotated(path, tileset_room_types):
    """Parse floor_plan_1.json's custom cell-list format directly into
    the same {"nodes":..., "edges":...} shape extract_room_graph
    produces, without needing a full tile-id grid."""
    with open(path) as f:
        data = json.load(f)

    cells = {(c["row"], c["col"]): c for c in data["cells"]}

    parent = {}

    def find(x):
        while parent.get(x, x) != x:
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for (r, c) in cells:
        parent.setdefault((r, c), (r, c))
    for (r, c), cell in cells.items():
        for dr, dc in ((0, 1), (1, 0)):
            nr, nc = r + dr, c + dc
            if (nr, nc) in cells and cells[(nr, nc)]["room_type"] == cell["room_type"]:
                union((r, c), (nr, nc))

    comp_ids = {}
    for (r, c) in cells:
        comp_ids.setdefault(find((r, c)), len(comp_ids))

    nodes_by_comp = {}
    for (r, c), cell in cells.items():
        cid = comp_ids[find((r, c))]
        nodes_by_comp.setdefault(cid, {"id": cid, "room_type": cell["room_type"], "size": 0})
        nodes_by_comp[cid]["size"] += 1

    edges = []
    seen = set()
    traversable = {"D", "O", "C"}
    for (r, c), cell in cells.items():
        comp_a = comp_ids[find((r, c))]
        for direction, (dr, dc) in (("s", (1, 0)), ("e", (0, 1))):
            nr, nc = r + dr, c + dc
            if (nr, nc) not in cells:
                continue
            edge_val = cell[f"e_{direction}"]
            if edge_val not in traversable:
                continue
            comp_b = comp_ids[find((nr, nc))]
            if comp_a == comp_b:
                continue
            key = (min(comp_a, comp_b), max(comp_a, comp_b))
            if key in seen:
                continue
            seen.add(key)
            edges.append({"a": comp_a, "b": comp_b, "edge_type": edge_val})

    return {"nodes": list(nodes_by_comp.values()), "edges": edges}


def build_examples(graphs, room_types):
    positives = set()
    for g in graphs:
        node_type = {n["id"]: n["room_type"] for n in g["nodes"]}
        for e in g["edges"]:
            a, b = node_type[e["a"]], node_type[e["b"]]
            positives.add((a, b))
            positives.add((b, a))

    examples = []
    for (a, b) in positives:
        examples.append((a, [b], 1.0))

    rng = random.Random(0)
    all_pairs = [(a, b) for a in room_types for b in room_types]
    negatives_needed = len(examples)
    negative_pool = [p for p in all_pairs if p not in positives]
    rng.shuffle(negative_pool)
    for (a, b) in negative_pool[:negatives_needed]:
        examples.append((a, [b], 0.0))

    rng.shuffle(examples)
    return examples


def run():
    tileset = Tileset.load(DATA_PATH)
    room_types = tileset.room_types

    if os.path.exists(RESPLAN_PATH):
        print(f"Found ResPlan dataset at {RESPLAN_PATH} -- using it as the primary training source.\n")
        examples, plans_used, pair_counts = build_examples_from_resplan(
            RESPLAN_PATH, room_types, max_plans=RESPLAN_MAX_PLANS, max_examples=20000
        )
        print(f"ResPlan plans used: {plans_used}")
        print_pair_frequency_report(pair_counts)
    else:
        print(f"ResPlan not found at {RESPLAN_PATH}.")
        print("Falling back to the small bootstrap dataset (1 annotated plan + "
              f"{N_GENERATED_PLANS} generated plans) -- see GNN_Heuristic_Integration_Plan.md "
              "for how to obtain ResPlan and unlock the real dataset.\n")

        graphs = []
        if os.path.exists(ANNOTATED_PATH):
            g = graph_from_annotated(ANNOTATED_PATH, room_types)
            graphs.append(g)
            print(f"Loaded 1 annotated reference plan: {len(g['nodes'])} rooms, {len(g['edges'])} edges")

        print(f"Generating {N_GENERATED_PLANS} Layer-2-valid plans as supplementary training data...")
        n_ok = 0
        for seed in range(N_GENERATED_PLANS):
            random.seed(9000 + seed)
            try:
                result = solve_nwfc(tileset, A=3, B=3, C=4, enforce_layer2=True, max_exterior_backtracks=50)
                g = extract_room_graph(result.global_grid, tileset)
                graphs.append(g)
                n_ok += 1
            except RuntimeError:
                continue
        print(f"  {n_ok}/{N_GENERATED_PLANS} generations succeeded and were added")

        examples = build_examples(graphs, room_types)

    n_pos = sum(1 for _, _, lbl in examples if lbl == 1.0)
    n_neg = sum(1 for _, _, lbl in examples if lbl == 0.0)
    print(f"\nTraining set: {len(examples)} examples ({n_pos} positive, {n_neg} negative)")

    split = int(len(examples) * 0.8)
    train_examples = examples[:split]
    test_examples = examples[split:]

    scorer = ContextScorer(room_types=room_types, hidden_dim=16, seed=1)
    print("\nTraining...")
    scorer.train(train_examples, epochs=300, lr=0.1, verbose=True)

    correct = 0
    for cand, neigh, label in test_examples:
        pred = scorer.predict(cand, neigh)
        if (pred > 0.5) == (label == 1.0):
            correct += 1
    test_acc = correct / len(test_examples) if test_examples else float("nan")
    print(f"\nHeld-out test accuracy: {test_acc:.1%} ({len(test_examples)} examples)")

    scorer.save(MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")

    return scorer, test_acc


if __name__ == "__main__":
    run()