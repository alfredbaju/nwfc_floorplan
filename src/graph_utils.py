"""Extracts room-adjacency graphs from generated or annotated tile grids.

This is the data layer for the learned selection heuristic: it converts a
tile-id grid (from wfc_core.py, nwfc.py, or a hand-annotated reference
plan like floor_plan_1.json) into a graph of room instances and their
connections -- the training data format the heuristic model in
heuristic_model.py consumes.

Reuses the same connected-component logic as visualize.py's room merging
(contiguous same-room_type cells = one room instance), so "what counts as
a room" is consistent between what gets rendered and what gets learned
from.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from wfc_core import DIRS  # noqa: E402


def find_room_components(tile_id_grid, tileset):
    """Group cells into room instances: contiguous cells sharing the same
    room_type. Returns (cell_to_component, components) where components
    maps component_id -> list of (r, c) cells."""
    rows = len(tile_id_grid)
    cols = len(tile_id_grid[0])
    cell_to_component = {}
    components = {}
    next_id = 0

    for r in range(rows):
        for c in range(cols):
            if (r, c) in cell_to_component:
                continue
            room_type = tileset.tiles[tile_id_grid[r][c]].room_type
            stack = [(r, c)]
            cell_to_component[(r, c)] = next_id
            members = [(r, c)]
            while stack:
                cr, cc = stack.pop()
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = cr + dr, cc + dc
                    if not (0 <= nr < rows and 0 <= nc < cols):
                        continue
                    if (nr, nc) in cell_to_component:
                        continue
                    if tileset.tiles[tile_id_grid[nr][nc]].room_type != room_type:
                        continue
                    cell_to_component[(nr, nc)] = next_id
                    members.append((nr, nc))
                    stack.append((nr, nc))
            components[next_id] = members
            next_id += 1

    return cell_to_component, components


def extract_room_graph(tile_id_grid, tileset):
    """Build a room-adjacency graph from a tile-id grid.

    Returns a dict:
      {
        "nodes": [{"id": int, "room_type": str, "size": int}, ...],
        "edges": [{"a": int, "b": int, "edge_type": "D"|"O"|"C"}, ...]
      }

    Nodes are room instances (merged same-type cell components). Edges
    are the *distinct* connections between different room instances,
    labeled with the geometric edge type that connects them. This is
    deliberately generator-agnostic -- it works the same whether the
    grid came from wfc_core.py, nwfc.py, or a hand-annotated JSON like
    floor_plan_1.json, as long as it's expressed as a tile-id grid.
    """
    cell_to_component, components = find_room_components(tile_id_grid, tileset)
    rows = len(tile_id_grid)
    cols = len(tile_id_grid[0])

    nodes = []
    for comp_id, members in components.items():
        r0, c0 = members[0]
        room_type = tileset.tiles[tile_id_grid[r0][c0]].room_type
        nodes.append({"id": comp_id, "room_type": room_type, "size": len(members)})

    seen_pairs = set()
    edges = []
    for r in range(rows):
        for c in range(cols):
            comp_a = cell_to_component[(r, c)]
            tile = tileset.tiles[tile_id_grid[r][c]]
            for direction, (dr, dc) in DIRS.items():
                nr, nc = r + dr, c + dc
                if not (0 <= nr < rows and 0 <= nc < cols):
                    continue
                comp_b = cell_to_component[(nr, nc)]
                if comp_a == comp_b:
                    continue  # same room instance, not a cross-room edge
                edge_type = tile.edge(direction)
                if edge_type == "W":
                    continue  # walls aren't connections
                pair_key = (min(comp_a, comp_b), max(comp_a, comp_b), edge_type)
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                edges.append({"a": comp_a, "b": comp_b, "edge_type": edge_type})

    return {"nodes": nodes, "edges": edges}