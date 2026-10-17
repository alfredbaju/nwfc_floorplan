"""Tileset loading and validation utilities for the N-WFC floor-plan project.

Loads the tile vocabulary defined in docs/TILESET_DESIGN.md and encoded in
data/tileset_core.json, and exposes it in a form the WFC/N-WFC engines can
consume directly.

This module only concerns itself with Layer 1 (geometric) data: tile ids,
room-type labels, and the four edge codes per tile. Layer 2 (the room-type
compatibility matrix) is loaded here too, but is applied later in
validity.py, not by the baseline WFC engine in wfc_core.py.
"""

from dataclasses import dataclass
import json

OPPOSITE = {"n": "s", "s": "n", "w": "e", "e": "w"}


@dataclass(frozen=True)
class Tile:
    id: str
    room_type: str
    e_n: str
    e_s: str
    e_w: str
    e_e: str
    weight: float = 1.0

    def edge(self, direction: str) -> str:
        """Return this tile's edge code facing the given direction
        ('n', 's', 'w', or 'e'). Direct attribute access, not
        getattr(self, f"e_{direction}") -- this is called on the order
        of tens of millions of times in a single generation (WFC's
        constraint propagation checks it for every candidate tile on
        every cell update), and profiling showed the f-string
        formatting + dynamic getattr call was consuming roughly 45% of
        total generation time on its own."""
        if direction == "n":
            return self.e_n
        if direction == "s":
            return self.e_s
        if direction == "w":
            return self.e_w
        if direction == "e":
            return self.e_e
        raise ValueError(f"Unknown direction: {direction!r}")


class Tileset:
    def __init__(self, edge_types, room_types, adjacency_compatibility, tiles):
        self.edge_types = edge_types
        self.room_types = room_types
        self.adjacency_compatibility = adjacency_compatibility
        self.tiles = tiles  # dict: tile id -> Tile

    @classmethod
    def load(cls, path: str) -> "Tileset":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tiles = {t["id"]: Tile(**t) for t in data["tiles"]}
        return cls(
            edge_types=data["edge_types"],
            room_types=data["room_types"],
            adjacency_compatibility=data["adjacency_compatibility"],
            tiles=tiles,
        )

    def is_room_compatible(self, room_a: str, room_b: str, threshold: float = 0.5) -> bool:
        """Layer 2 check: may these two room types connect through a door?

        The compatibility matrix now mixes plain booleans (for room types
        with no reliable training signal -- see compatibility_score's
        docstring) with continuous, data-derived scores in [0, 1] for
        BED/BATH/KIT/LIV (learned from the ResPlan dataset). A naive
        bool(value) would incorrectly treat any nonzero float as
        compatible (e.g. bool(0.037) is True) -- this thresholds
        explicitly instead. bool(True) == 1.0 >= threshold and
        bool(False) == 0.0 < threshold, so plain-boolean entries are
        unaffected by this change.
        """
        value = self.adjacency_compatibility.get(room_a, {}).get(room_b, False)
        return float(value) >= threshold

    def compatibility_score(self, room_a: str, room_b: str) -> float:
        """Raw compatibility value: either the hand-derived 1.0/0.0 for
        room types without real training data (COR, ENT, EXT -- see
        train_heuristic.py / load_resplan.py for why these were excluded
        from the data-driven update), or the continuous, ResPlan-derived
        plausibility score for BED/BATH/KIT/LIV pairs."""
        value = self.adjacency_compatibility.get(room_a, {}).get(room_b, False)
        return float(value)

    def edge_matches(self, tile_a_id: str, direction: str, tile_b_id: str) -> bool:
        """Layer 1 geometric check: does tile_b fit in the neighboring cell
        located in `direction` relative to tile_a? (i.e. tile_a's edge
        facing `direction` must equal tile_b's edge facing the opposite
        direction, per the base paper's accepted-result definition.)"""
        a, b = self.tiles[tile_a_id], self.tiles[tile_b_id]
        return a.edge(direction) == b.edge(OPPOSITE[direction])