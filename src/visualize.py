"""Render a generated tile-id grid as an actual floor-plan diagram using
matplotlib: colored rooms, walls as solid black lines, doors as gapped
brown openings, corridors/open archways as lighter dashed lines.

This turns wfc_core.py / nwfc.py output (a 2-D grid of tile ids) into
something a person can actually look at and evaluate, rather than raw
room-type text -- the E4 qualitative case studies in your experiment plan
depend on this.
"""

import os
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(__file__))

ROOM_COLORS = {
    "BED": "#F6C6C6",
    "BATH": "#AEDFF7",
    "KIT": "#FDE7A8",
    "LIV": "#C9E4CA",
    "COR": "#E3E3E3",
    "ENT": "#FFC98B",
    "EXT": "#FFFFFF",
    "STAIR": "#D5B8E0",
    "BALC": "#BFE3D0",
}
ROOM_LABELS = {
    "BED": "Bedroom",
    "BATH": "Bathroom",
    "KIT": "Kitchen",
    "LIV": "Living",
    "COR": "Corridor",
    "ENT": "Entrance",
    "EXT": "Exterior",
    "STAIR": "Staircase",
    "BALC": "Balcony",
}

WALL_STYLE = dict(color="black", linewidth=3, solid_capstyle="round")
DOOR_STYLE = dict(color="#8B5A2B", linewidth=3)
OPEN_STYLE = dict(color="#999999", linewidth=1, linestyle=(0, (1, 2)))
CORRIDOR_STYLE = dict(color="#999999", linewidth=1.5, linestyle=(0, (4, 2)))

EDGE_STYLES = {"W": WALL_STYLE, "D": DOOR_STYLE, "O": OPEN_STYLE, "C": CORRIDOR_STYLE}
DOOR_GAP_EDGES = {"D"}  # edge types drawn with a visible gap (an opening), not a solid line


def _draw_edge(ax, edge_type, p1, p2, cell_size):
    style = EDGE_STYLES.get(edge_type, dict(color="lightgray", linewidth=0.5))
    if edge_type in DOOR_GAP_EDGES:
        # Draw two short wall stubs (the wall the door is set into) with
        # a visibly colored door-style bar filling the gap between them
        # -- previously the gap was left with only plain wall-colored
        # stubs, so a door was indistinguishable from a broken wall.
        gap = cell_size * 0.35
        (x1, y1), (x2, y2) = p1, p2
        dx, dy = (x2 - x1), (y2 - y1)
        length = (dx**2 + dy**2) ** 0.5
        ux, uy = dx / length, dy / length
        stub_len = (length - gap) / 2
        gap_start = (x1 + ux * stub_len, y1 + uy * stub_len)
        gap_end = (x2 - ux * stub_len, y2 - uy * stub_len)
        ax.plot([x1, gap_start[0]], [y1, gap_start[1]], **WALL_STYLE)
        ax.plot([gap_end[0], x2], [gap_end[1], y2], **WALL_STYLE)
        ax.plot([gap_start[0], gap_end[0]], [gap_start[1], gap_end[1]], **DOOR_STYLE, solid_capstyle="butt")
    else:
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], **style)


def _find_room_components(tile_id_grid, tileset):
    """Group cells into connected 'room instances': contiguous cells
    sharing the same room_type, connected via any traversable edge (O, D,
    or C -- treated loosely here since this is a rendering concern, not a
    Layer 1/2 generation concern). Returns a dict (r,c) -> component_id
    and a dict component_id -> list of (r,c).
    """
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
            # BFS over same-room_type neighbors
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


def render_floor_plan(tile_id_grid, tileset, cell_size=1.0, title=None,
                       show_labels=True, save_path=None, merge_rooms=True):
    """Render a tile-id grid as a floor-plan diagram.

    tile_id_grid: 2-D list of tile ids (as produced by wfc_core.solve or
        nwfc.solve_nwfc's .global_grid).
    tileset: the Tileset instance used to generate the grid.
    merge_rooms: if True (default), adjacent cells of the same room_type
        are rendered as one continuous shape (no internal grid lines,
        one label at the centroid) instead of a visible per-cell grid --
        this is what makes the output read as an actual floor plan rather
        than a lattice of 1x1 rooms.
    save_path: if given, writes a PNG there. Either way, returns the
        matplotlib Figure so it can also be displayed inline.
    """
    rows = len(tile_id_grid)
    cols = len(tile_id_grid[0])

    cell_to_component, components = _find_room_components(tile_id_grid, tileset) if merge_rooms else ({}, {})

    fig, ax = plt.subplots(figsize=(max(4, cols * cell_size * 0.9), max(4, rows * cell_size * 0.9)))

    for r in range(rows):
        for c in range(cols):
            tile = tileset.tiles[tile_id_grid[r][c]]
            x = c * cell_size
            y = (rows - 1 - r) * cell_size  # flip so row 0 renders at the top
            color = ROOM_COLORS.get(tile.room_type, "#DDDDDD")
            ax.add_patch(mpatches.Rectangle((x, y), cell_size, cell_size,
                                             facecolor=color, edgecolor="none"))

    if show_labels:
        if merge_rooms:
            for comp_id, members in components.items():
                room_type = tileset.tiles[tile_id_grid[members[0][0]][members[0][1]]].room_type
                if room_type == "EXT":
                    continue
                rs = [rows - 1 - r for r, _ in members]
                cs = [c for _, c in members]
                cy = (sum(rs) / len(rs) + 0.5) * cell_size
                cx = (sum(cs) / len(cs) + 0.5) * cell_size
                if room_type == "ENT":
                    ax.plot(cx, cy, marker="*", color="#B8860B", markersize=16, zorder=5)
                else:
                    ax.text(cx, cy, ROOM_LABELS.get(room_type, room_type),
                            ha="center", va="center", fontsize=9, color="#333333", weight="medium")
        else:
            for r in range(rows):
                for c in range(cols):
                    tile = tileset.tiles[tile_id_grid[r][c]]
                    x = c * cell_size
                    y = (rows - 1 - r) * cell_size
                    if tile.room_type == "ENT":
                        ax.plot(x + cell_size / 2, y + cell_size / 2, marker="*",
                                color="#B8860B", markersize=14, zorder=5)
                    elif tile.room_type != "EXT" and cell_size >= 0.5:
                        ax.text(x + cell_size / 2, y + cell_size / 2, tile.room_type,
                                ha="center", va="center", fontsize=7, color="#333333")

    for r in range(rows):
        for c in range(cols):
            tile = tileset.tiles[tile_id_grid[r][c]]
            x = c * cell_size
            y = (rows - 1 - r) * cell_size

            def maybe_draw(edge_type, p1, p2, nr, nc):
                if merge_rooms and 0 <= nr < rows and 0 <= nc < cols:
                    if cell_to_component.get((r, c)) == cell_to_component.get((nr, nc)):
                        return  # same room instance -- no internal line
                _draw_edge(ax, edge_type, p1, p2, cell_size)

            maybe_draw(tile.e_n, (x, y + cell_size), (x + cell_size, y + cell_size), r - 1, c)
            maybe_draw(tile.e_s, (x, y), (x + cell_size, y), r + 1, c)
            maybe_draw(tile.e_w, (x, y), (x, y + cell_size), r, c - 1)
            maybe_draw(tile.e_e, (x + cell_size, y), (x + cell_size, y + cell_size), r, c + 1)

    ax.set_xlim(0, cols * cell_size)
    ax.set_ylim(0, rows * cell_size)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=12)

    used_types = {tileset.tiles[tid].room_type for row in tile_id_grid for tid in row}
    legend_handles = [
        mpatches.Patch(facecolor=ROOM_COLORS.get(rt, "#DDDDDD"), edgecolor="black", linewidth=0.5,
                        label=ROOM_LABELS.get(rt, rt))
        for rt in sorted(used_types)
    ]
    legend_handles.append(plt.Line2D([0], [0], color=WALL_STYLE["color"], lw=3, label="Wall"))
    legend_handles.append(plt.Line2D([0], [0], color=DOOR_STYLE["color"], lw=3, label="Door"))
    ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.02, 1),
              fontsize=8, frameon=False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def compute_area_breakdown(tile_id_grid, tileset, exclude_types=("EXT",)):
    """Compute each room type's share of the total floor area, as a
    percentage of all non-excluded cells (EXT/void cells are excluded by
    default, since they aren't real floor area). Returns a list of
    {"room_type", "label", "cells", "percent"} dicts, sorted by area
    descending, largest room type first.
    """
    rows = len(tile_id_grid)
    cols = len(tile_id_grid[0]) if rows else 0
    counts = {}
    total = 0
    for r in range(rows):
        for c in range(cols):
            room_type = tileset.tiles[tile_id_grid[r][c]].room_type
            if room_type in exclude_types:
                continue
            counts[room_type] = counts.get(room_type, 0) + 1
            total += 1

    breakdown = []
    for room_type, cells in counts.items():
        breakdown.append({
            "room_type": room_type,
            "label": ROOM_LABELS.get(room_type, room_type),
            "cells": cells,
            "percent": (cells / total * 100) if total else 0.0,
        })
    breakdown.sort(key=lambda d: d["percent"], reverse=True)
    return breakdown


def export_figures_to_pdf(figures, destination):
    """Combine several matplotlib Figures into a single multi-page PDF,
    one figure per page, in the given order. Used to bundle all floors
    of a generated building into one downloadable document rather than
    a separate image per floor.

    figures: list of matplotlib Figure objects (as returned by
    render_floor_plan). destination: a file path (str) to write the PDF
    to, or a file-like object (e.g. io.BytesIO) to write into -- the
    latter is used to build the PDF entirely in memory for a Streamlit
    download button, without touching disk.
    """
    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(destination) as pdf:
        for fig in figures:
            pdf.savefig(fig, bbox_inches="tight")
    return destination


if __name__ == "__main__":
    import random
    from tileset import Tileset
    from nwfc import solve_nwfc

    DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "tileset_core.json")
    tileset = Tileset.load(DATA_PATH)

    random.seed(1)
    result = solve_nwfc(tileset, A=3, B=3, C=4)

    out_dir = os.path.join(os.path.dirname(__file__), "..", "data", "generated")
    os.makedirs(out_dir, exist_ok=True)

    render_floor_plan(result.global_grid, tileset, cell_size=1.0,
                       title="N-WFC generated floor plan -- unmerged (before)",
                       merge_rooms=False,
                       save_path=os.path.join(out_dir, "demo_floorplan_before.png"))
    render_floor_plan(result.global_grid, tileset, cell_size=1.0,
                       title="N-WFC generated floor plan -- merged rooms (after)",
                       merge_rooms=True,
                       save_path=os.path.join(out_dir, "demo_floorplan_after.png"))
    print(f"Saved to {out_dir}")