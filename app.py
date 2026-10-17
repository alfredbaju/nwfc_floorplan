"""N-WFC Floor Plan Generator -- interactive web app.

Wraps the existing generation pipeline (tileset -> N-WFC -> repair ->
validate -> render) behind a Streamlit interface. Run with:

    streamlit run app.py

No new generation logic lives here -- this is purely a UI layer over
src/nwfc.py, src/repair.py, src/validity.py, and src/visualize.py, all
of which are already tested independently in tests/.
"""

import io
import os
import random
import sys
import time

import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from tileset import Tileset  # noqa: E402
from nwfc import solve_nwfc  # noqa: E402
from validity import validate  # noqa: E402
from visualize import render_floor_plan, export_figures_to_pdf, compute_area_breakdown  # noqa: E402
from repair import repair_connectivity, unify_corridor_network, enforce_internal_room_connectivity  # noqa: E402
from entrance_cap import cap_entrance_count  # noqa: E402
from room_connections import enforce_room_connection_rules  # noqa: E402
from heuristic_model import ContextScorer  # noqa: E402
from wall_cleanup import cleanup_cross_room_edges  # noqa: E402
from multi_floor import generate_multi_floor_building  # noqa: E402
from zoning import (  # noqa: E402
    solve_zoned_with_exact_counts,
    count_actual_room_instances,
    compute_grid_size_for_rooms,
    ensure_entrance_on_border,
)

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "tileset_core.json")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "data", "context_scorer.npz")

st.set_page_config(page_title="N-WFC Floor Plan Generator", layout="wide")


@st.cache_resource
def load_tileset():
    return Tileset.load(DATA_PATH)


@st.cache_resource
def load_scorer():
    if os.path.exists(MODEL_PATH):
        try:
            return ContextScorer.load(MODEL_PATH)
        except Exception:
            return None
    return None


tileset = load_tileset()
scorer = load_scorer()

st.title("N-WFC Floor Plan Generator")
st.caption(
    "Generalizing Nested Wave Function Collapse to architectural floor-plan generation. "
    "Every option below maps directly to a tested module in this project's codebase."
)

with st.sidebar:
    st.header("Generation Settings")

    zoning_enabled = st.checkbox(
        "Control exact room counts", value=False,
        help="Instead of letting room counts emerge randomly, request an exact number of each room type. "
             "Uses macro-zoning: specific subgrids are dedicated to each requested room, the rest become connective corridor.",
    )

    room_counts = {}
    num_floors = 1
    second_entrance_near_kitchen = False
    auto_retry_quality = False
    if zoning_enabled:
        num_floors = st.slider("Number of floors", min_value=1, max_value=3, value=1)

        c1, c2 = st.columns(2)
        with c1:
            room_counts["BED"] = st.number_input("Bedrooms" + (" (total)" if num_floors > 1 else ""),
                                                    min_value=0, max_value=20, value=1, step=1)
            room_counts["KIT"] = st.number_input("Kitchens" + (" (total)" if num_floors > 1 else ""),
                                                    min_value=0, max_value=20, value=1, step=1)
        with c2:
            room_counts["BATH"] = st.number_input("Bathrooms" + (" (total)" if num_floors > 1 else ""),
                                                     min_value=0, max_value=20, value=1, step=1)
            if num_floors > 1:
                st.number_input("Living rooms", min_value=1, max_value=1, value=1, step=1, disabled=True,
                                 help="Fixed at exactly 1 per floor when building multiple floors.")
                room_counts["LIV"] = 1
            else:
                room_counts["LIV"] = st.number_input("Living rooms", min_value=0, max_value=12, value=1, step=1)

        balc_col, ent_col = st.columns(2)
        with balc_col:
            if num_floors > 1:
                balcony_count = st.number_input("Balconies (upper floors only, total)", min_value=0, max_value=6,
                                                  value=0, step=1)
                if balcony_count > 0:
                    room_counts["BALC"] = balcony_count
            else:
                st.number_input("Balconies", min_value=0, max_value=0, value=0, step=1, disabled=True,
                                 help="Balconies only apply to upper floors -- select 2 or more floors to enable this.")
                balcony_count = 0
        with ent_col:
            second_entrance_near_kitchen = st.checkbox(
                "2nd entrance near kitchen",
                help="Adds a service/back door near the kitchen, in addition to the main entrance near the living room. "
                     "Ground floor only.",
            )
        if balcony_count > 0:
            st.caption("Balconies are placed only on upper floors (never the ground floor), on the exterior wall "
                       "with an open (no-door) connection to a living room or bedroom.")
        if second_entrance_near_kitchen:
            st.caption("A second entrance will be placed adjacent to the kitchen when possible (not guaranteed every generation).")

        auto_retry_quality = st.checkbox(
            "Auto-retry for best quality (up to 3 attempts)",
            help="Some placements (corridor unity, second entrance reaching the kitchen, balcony on the "
                 "exterior wall) are best-effort, not guaranteed on the first try. When checked, up to 2 "
                 "extra attempts are tried automatically and the best-scoring one is kept.",
        )

        if num_floors > 1:
            st.caption(
                "Bedroom/bathroom/kitchen counts above are TOTALS across the whole building, split as evenly "
                "as possible across floors. Every floor gets exactly one living room and shares one staircase "
                "kept at the same position on every floor. Only the ground floor has a building entrance -- "
                "upper floors are reached via the staircase."
            )

        st.caption(
            "Living rooms are generated roughly 2x the area of other rooms. Every room gets exactly one "
            "connection to a corridor, never a direct connection to another room: bedrooms/bathrooms get "
            "a real door, while kitchens and living rooms get an open archway (no door). Kitchens match "
            "bedroom size. Bathrooms currently match bedroom size too -- making them smaller isn't "
            "supported yet (see known limitations)."
        )
        total_requested = sum(room_counts.values())
        if total_requested == 0:
            st.warning("Request at least one room.")
        else:
            display_counts = dict(room_counts)
            if num_floors > 1:
                display_counts["STAIR"] = 1
            A, B = compute_grid_size_for_rooms(display_counts)
            st.caption(f"Auto-sized grid: {A}x{B} subgrids per floor (sized to keep corridors a minority of the map).")
        C = 4
    else:
        size_map = {
            "Small (7x7)": (2, 2, 4),
            "Medium (10x10)": (3, 3, 4),
            "Large (13x13)": (4, 4, 4),
            "Extra Large (16x16)": (5, 5, 4),
        }
        size_label = st.select_slider("Plan size", options=list(size_map.keys()), value="Medium (10x10)")
        A, B, C = size_map[size_label]

    st.divider()
    enforce_layer2 = st.checkbox(
        "Enforce room compatibility (Layer 2)", value=True,
        help="Filters door connections to only room-type pairs the compatibility matrix allows.",
    )

    use_scorer = st.checkbox(
        "Use learned selection heuristic", value=scorer is not None, disabled=scorer is None,
        help="Weights candidate tiles using the model trained on real ResPlan floor-plan data.",
    )
    if scorer is None:
        st.caption("No trained model found at data/context_scorer.npz -- run train_heuristic.py first.")

    auto_repair = st.checkbox(
        "Auto-repair connectivity", value=True,
        help="After generation, converts isolating walls into doors so every room is reachable from the entrance.",
    )

    seed_mode = st.radio("Seed", ["Random", "Specific"], horizontal=True)
    seed_value = st.number_input("Seed value", value=42, step=1) if seed_mode == "Specific" else None

    generate_clicked = st.button("Generate Floor Plan", type="primary", width='stretch')

if "history" not in st.session_state:
    st.session_state.history = []


def _generate_single_floor(seed, zoning_enabled, room_counts, A, B, C, enforce_layer2,
                            scorer, use_scorer, second_entrance_near_kitchen, auto_repair, tileset):
    """The full single-floor generate-and-repair pipeline, as one call
    so it can be run more than once (for auto-retry) without duplicating
    this sequence. Returns a dict of every value the report/render code
    needs -- deliberately a dict, not a dataclass, so this stays a
    faithful extraction of what was previously inline code rather than
    a new abstraction with its own behavior to verify.
    """
    random.seed(seed)
    t0 = time.perf_counter()
    exterior_backtracks = None
    zoning_attempts = None
    zoning_matched = None

    if zoning_enabled:
        grid, subgrid_plan, zoning_attempts, zoning_matched = solve_zoned_with_exact_counts(
            tileset, A=A, B=B, C=C, room_counts=room_counts,
            enforce_layer2=enforce_layer2, context_scorer=scorer if use_scorer else None,
            second_entrance_near_kitchen=second_entrance_near_kitchen,
        )
        if grid is None:
            return None
    else:
        try:
            result = solve_nwfc(
                tileset, A=A, B=B, C=C,
                enforce_layer2=enforce_layer2,
                context_scorer=scorer if use_scorer else None,
                max_exterior_backtracks=50,
            )
        except RuntimeError:
            return None
        grid = result.global_grid
        exterior_backtracks = result.exterior_backtracks

    gen_time = time.perf_counter() - t0

    grid, wall_fixes = cleanup_cross_room_edges(grid, tileset)

    if zoning_enabled:
        grid, _ = enforce_internal_room_connectivity(grid, tileset)

    if zoning_enabled:
        max_ent = 2 if second_entrance_near_kitchen else 1
        diverse = ("LIV", "KIT") if second_entrance_near_kitchen else ()
        grid, entrance_conversions = cap_entrance_count(
            grid, tileset, max_entrances=max_ent, preferred_types=("COR",), strict=True,
            prefer_diverse_adjacency=diverse,
        )
    else:
        grid, entrance_conversions = cap_entrance_count(grid, tileset)

    repairs_made = []
    room_connection_results = None
    corridor_unified = None
    if auto_repair:
        grid, repairs_made, _ = repair_connectivity(grid, tileset, prefer_corridor=True, protect_types=("KIT", "BALC"))
        if zoning_enabled:
            grid, room_connection_results = enforce_room_connection_rules(grid, tileset)
            grid, extra_repairs, _ = repair_connectivity(grid, tileset, prefer_corridor=True, protect_types=("KIT", "BALC"))
            repairs_made += extra_repairs
            grid, corridor_unified = unify_corridor_network(grid, tileset)
    if zoning_enabled:
        grid, _ = ensure_entrance_on_border(grid, tileset)

    report = validate(grid, tileset)
    actual_counts = count_actual_room_instances(grid, tileset) if zoning_enabled else None

    return {
        "grid": grid, "gen_time": gen_time, "exterior_backtracks": exterior_backtracks,
        "zoning_attempts": zoning_attempts, "zoning_matched": zoning_matched, "wall_fixes": wall_fixes,
        "entrance_conversions": entrance_conversions, "repairs_made": repairs_made,
        "room_connection_results": room_connection_results, "corridor_unified": corridor_unified,
        "report": report, "actual_counts": actual_counts, "seed": seed,
    }


def _score_quality(result, second_entrance_near_kitchen, has_balcony, tileset):
    """A simple count of how many of the "soft" (best-effort, not
    hard-guaranteed) architectural preferences a generated result
    satisfies -- used to pick the best of several auto-retry attempts.
    Hard requirements (structural validity, reachability, exact counts)
    are assumed already satisfied before this is called; this only
    differentiates among otherwise-valid results."""
    score, max_score = 0, 0
    if result["corridor_unified"] is not None:
        max_score += 1
        score += int(bool(result["corridor_unified"]))
    if second_entrance_near_kitchen and result["room_connection_results"]:
        max_score += 1
        ent_results = [r for r in result["room_connection_results"] if r["room_type"] == "ENT"]
        score += int(any(r["connected_to"] == "KIT" for r in ent_results))
    if has_balcony:
        max_score += 1
        rows, cols = len(result["grid"]), len(result["grid"][0])
        balc_cells = [(r, c) for r in range(rows) for c in range(cols)
                      if tileset.tiles[result["grid"][r][c]].room_type == "BALC"]
        on_border = balc_cells and all(
            r == 0 or r == rows - 1 or c == 0 or c == cols - 1 for (r, c) in balc_cells
        )
        score += int(bool(on_border))
    return score, max_score


if generate_clicked:
    if zoning_enabled and sum(room_counts.values()) == 0:
        st.error("Fix the room count settings in the sidebar before generating.")
        st.stop()

    actual_seed = seed_value if seed_value is not None else random.randint(0, 1_000_000)
    random.seed(actual_seed)

    if zoning_enabled and num_floors > 1:
        # --- Multi-floor path: completely separate from single-floor
        # generation below, since it orchestrates several floors plus
        # staircase alignment rather than one grid.
        with st.spinner(f"Generating {num_floors} floors..."):
            t0 = time.perf_counter()
            try:
                floors = generate_multi_floor_building(
                    tileset, num_floors=num_floors, room_counts=room_counts, C=C,
                    enforce_layer2=enforce_layer2, context_scorer=scorer if use_scorer else None,
                    seed=actual_seed, second_entrance_near_kitchen=second_entrance_near_kitchen,
                )
            except RuntimeError as exc:
                st.error(f"Generation failed: {exc}. Try different counts or a new seed.")
                st.stop()
            gen_time = time.perf_counter() - t0

        st.success(f"Generated {num_floors} floors in {gen_time:.1f}s (seed={actual_seed}).")

        floor_figures = []
        tabs = st.tabs([f"Floor {f.floor_number}" + (" (Ground)" if f.is_ground_floor else "") for f in floors])
        for tab, floor in zip(tabs, floors):
            with tab:
                col1, col2 = st.columns([2, 1])
                with col1:
                    fig = render_floor_plan(
                        floor.grid, tileset, cell_size=1.0,
                        title=f"Floor {floor.floor_number}" + (" (Ground)" if floor.is_ground_floor else ""),
                        merge_rooms=True,
                    )
                    st.pyplot(fig, width='stretch')
                    floor_figures.append(fig)
                with col2:
                    st.subheader("Floor Report")
                    st.metric("Structural validity", "PASS" if floor.structural_valid else "FAIL")
                    st.metric("Reachable" + (" (from entrance)" if floor.is_ground_floor else " (from staircase)"),
                               "PASS" if floor.reachable else "FAIL")
                    if not floor.reachable:
                        st.warning(f"{len(floor.unreachable_cells)} cells unreachable on this floor.")
                    st.json({rt: floor.room_counts_actual.get(rt, 0) for rt in ("BED", "BATH", "KIT", "LIV", "STAIR")})
                    if floor.stair_positions:
                        st.caption(f"Staircase at macro position(s) {floor.stair_positions} -- "
                                   f"kept identical across all floors.")

                    st.divider()
                    st.subheader("Area breakdown")
                    breakdown = compute_area_breakdown(floor.grid, tileset)
                    for entry in breakdown:
                        st.progress(min(entry["percent"] / 100, 1.0), text=f"{entry['label']}: {entry['percent']:.1f}%")

        pdf_buffer = io.BytesIO()
        export_figures_to_pdf(floor_figures, pdf_buffer)
        pdf_buffer.seek(0)
        st.download_button(
            "Download all floors as PDF", data=pdf_buffer, file_name=f"floor_plan_seed{actual_seed}.pdf",
            mime="application/pdf",
        )
        st.stop()

    # --- Single-floor path (original behavior, unchanged) ---
    has_balcony = zoning_enabled and room_counts.get("BALC", 0) > 0
    with st.spinner("Generating..."):
        attempts_tried = 1
        best = _generate_single_floor(
            actual_seed, zoning_enabled, room_counts, A, B, C, enforce_layer2,
            scorer, use_scorer, second_entrance_near_kitchen, auto_repair, tileset,
        )
        if best is None:
            if zoning_enabled:
                st.error("Generation failed entirely for this room-count request. Try different counts or a larger grid.")
            else:
                st.error("Generation failed with this seed. Try a different seed or size.")
            st.stop()
        best_score, best_max = _score_quality(best, second_entrance_near_kitchen, has_balcony, tileset)

        if auto_retry_quality and best_max > 0 and best_score < best_max:
            for retry_i in range(2):  # up to 2 additional attempts, 3 total
                candidate_seed = actual_seed + retry_i + 1
                candidate = _generate_single_floor(
                    candidate_seed, zoning_enabled, room_counts, A, B, C, enforce_layer2,
                    scorer, use_scorer, second_entrance_near_kitchen, auto_repair, tileset,
                )
                if candidate is None:
                    continue
                attempts_tried += 1
                cand_score, cand_max = _score_quality(candidate, second_entrance_near_kitchen, has_balcony, tileset)
                if cand_score > best_score:
                    best, best_score, best_max = candidate, cand_score, cand_max
                    actual_seed = candidate_seed
                if best_score >= best_max:
                    break

    grid = best["grid"]
    gen_time = best["gen_time"]
    exterior_backtracks = best["exterior_backtracks"]
    zoning_attempts = best["zoning_attempts"]
    zoning_matched = best["zoning_matched"]
    wall_fixes = best["wall_fixes"]
    entrance_conversions = best["entrance_conversions"]
    repairs_made = best["repairs_made"]
    room_connection_results = best["room_connection_results"]
    corridor_unified = best["corridor_unified"]
    report = best["report"]
    actual_counts = best["actual_counts"]

    if auto_retry_quality and best_max > 0:
        if best_score >= best_max:
            st.success(f"Auto-retry: all {best_max} soft quality checks satisfied after {attempts_tried} attempt(s).")
        else:
            st.info(f"Auto-retry: best result satisfies {best_score}/{best_max} soft quality checks "
                    f"after {attempts_tried} attempt(s) (seed={actual_seed}).")

    col1, col2 = st.columns([2, 1])

    with col1:
        fig = render_floor_plan(
            grid, tileset, cell_size=1.0,
            title=f"Generated Floor Plan (seed={actual_seed})",
            merge_rooms=True,
        )
        st.pyplot(fig, width='stretch')

        pdf_buffer = io.BytesIO()
        export_figures_to_pdf([fig], pdf_buffer)
        pdf_buffer.seek(0)
        st.download_button(
            "Download as PDF", data=pdf_buffer, file_name=f"floor_plan_seed{actual_seed}.pdf",
            mime="application/pdf",
        )

    with col2:
        st.subheader("Generation Report")
        st.metric("Seed used", actual_seed)
        st.metric("Generation time", f"{gen_time * 1000:.0f} ms")
        if exterior_backtracks is not None:
            st.metric("Exterior backtracks", exterior_backtracks)

        if zoning_enabled:
            st.divider()
            st.metric("Zoning attempts needed", zoning_attempts)
            if zoning_matched:
                st.success("Exact room counts achieved.")
            else:
                st.warning("Could not achieve exact room counts within the attempt budget -- showing closest result.")
            requested_vs_actual = {
                rt: f"{room_counts.get(rt, 0)} requested / {actual_counts.get(rt, 0)} actual"
                for rt in ("BED", "BATH", "KIT", "LIV")
            }
            st.json(requested_vs_actual)
            if room_connection_results:
                fixed_count = sum(1 for r in room_connection_results if r["fixed"])
                total_count = len(room_connection_results)
                if fixed_count == total_count:
                    st.success(f"Every room has exactly one connection to a corridor ({fixed_count}/{total_count}).")
                else:
                    st.warning(
                        f"{fixed_count}/{total_count} rooms achieved exactly one corridor connection -- "
                        f"the rest needed an extra connection to stay fully reachable (connectivity is "
                        f"prioritized over this rule when the two genuinely conflict)."
                    )
                with st.expander("Per-room connection details"):
                    st.json([
                        {"room": r["room_type"], "connections": r["connections_after"],
                         "type": r["edge_type"], "connected_to": r["connected_to"],
                         "single_connection_achieved": r["fixed"]}
                        for r in room_connection_results
                    ])
            if corridor_unified is not None:
                if corridor_unified:
                    st.success("Corridor network is fully connected (one continuous hallway system).")
                else:
                    st.warning("Corridor network has disconnected fragments (rare -- rooms are still all reachable overall).")

        st.divider()
        st.metric("Structural validity", "PASS" if report.structural_valid else "FAIL")
        st.metric("Functional solvability", "PASS" if report.functional_solvable else "FAIL")
        st.metric("Constraint satisfaction", f"{report.constraint_satisfaction_rate:.0%}")
        st.metric("Entrance cells found", len(report.entrance_cells))
        if entrance_conversions:
            st.caption(f"Capped to a single entrance ({entrance_conversions} surplus cell(s) converted to ordinary rooms).")

        st.divider()
        st.subheader("Area breakdown")
        breakdown = compute_area_breakdown(grid, tileset)
        for entry in breakdown:
            st.progress(min(entry["percent"] / 100, 1.0), text=f"{entry['label']}: {entry['percent']:.1f}%")

        if auto_repair:
            st.divider()
            st.metric("Repairs applied", len(repairs_made))
            if not report.functional_solvable:
                st.warning(
                    f"{len(report.unreachable_cells)} cells still unreachable after repair -- "
                    "this happens in a small fraction of cases where no valid door "
                    "substitution exists anywhere along the disconnected boundary."
                )
            elif repairs_made:
                st.success(f"Fully connected after {len(repairs_made)} repair(s).")

    st.session_state.history.insert(0, {
        "seed": actual_seed, "size": f"{A}x{B} (zoned)" if zoning_enabled else size_label,
        "solvable": report.functional_solvable,
        "satisfaction": report.constraint_satisfaction_rate,
        "repairs": len(repairs_made),
    })
    st.session_state.history = st.session_state.history[:10]

else:
    st.info("Configure settings in the sidebar and click **Generate Floor Plan** to begin.")

if st.session_state.history:
    st.divider()
    st.subheader("Recent generations (this session)")
    st.dataframe(st.session_state.history, width='stretch', hide_index=True)