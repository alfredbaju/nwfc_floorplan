# PROJECT HANDOVER — N-WFC Architectural Floor-Plan Generator

Written at the end of a very long chat so a fresh conversation can continue without it.
No project source file was modified while preparing this document. Two new, additive items were
created: this file and the `PENDING_CANDIDATE/` folder (see §8). Nothing existing was overwritten.

## 0. How to read this document

Every factual claim carries one tag:

| Tag | Meaning |
|---|---|
| **[V]** | Verified by running code during the handover preparation. Commands are in §9. |
| **[C]** | Read directly from current source/data files. |
| **[P]** | Claimed in earlier chat or in an old doc. **Not re-verified.** Treat as a lead, not a fact. |
| **[A]** | Assumption or inference. Not verified. |

**Rule for the next conversation:** where this document and an old chat/doc disagree, trust the current code, then this document, then old docs. §6 lists known conflicts.

**Most important single fact:** there are two code states. The **shipped** state (`outputs/src`, what the user's app currently runs) has the problems in the user's screenshots. A **candidate** state (`outputs/PENDING_CANDIDATE/`) fixes them in testing but is **not shipped** and has had limited testing. See §8.

---

## 1. Purpose, requirements, current objectives

**Purpose [C: README]:** Final-year project generalizing Nested Wave Function Collapse (N-WFC; Nie, Zheng, Zhuang, Togelius, *IEEE Trans. Games* 16(4), 2024) from game-tile content to **architectural floor-plan generation**. Output is a **schematic** 2-D layout only. It is explicitly **not** building-code compliant.

**Project reference files (Claude project, not in code) [C]:** the N-WFC paper PDF and five Pala Tech floor-plan PDFs (`Floor_Plan_1,2,3,4,9`). Whether the floor plans feed any code is **[A] no** — nothing in `src/` loads them.

**Layout requirements stated by the user (cumulative; status = current *shipped* behaviour unless noted):**

| # | Requirement (user's words condensed) | Shipped | Candidate |
|---|---|---|---|
| R1 | Corridor ≈ 10% of floor; a thin connecting path, one connected network linking every room and the stairs, extending across floors | Not met: 26–40% seen in screenshots | Floor 1 ≈ 17%, floor 2 ≈ 12% **[V]**; "thin path / extends across floors" not implemented |
| R2 | Bathrooms always adjacent to a bedroom | Mostly (a floor-2 violation appears in the screenshot) | Floor 1 6/6, floor 2 5/6 **[V]** |
| R3 | Kitchen: one clean zone, never adjacent to bedroom/bathroom, in a corner/grid edge, opposite living room, joined to living room by corridor, size ≈ living (living slightly larger) | Not met (kitchen 1–12 cells of 16, touching bed/bath) | 0/6 touching bed/bath, ≈17% area **[V]**; "opposite living" and "exactly equal size" only partly checked |
| R4 | Upper floors with balconies: bedrooms + balconies take most space, living ≤ 30%, balcony always on an edge | Not met (living ~27–50%, bedrooms truncated) | Floor 2: bed 37%, balcony 20%, living 20% **[V]**; balcony-on-edge only via planner scoring, not independently verified at tile level |
| R5 | Every room has exactly one door, to the corridor | **Not implemented** (see `living_room_door.py` note in §3) | Not implemented |
| R6 | "Schematic concept only" disclaimer in app output | **Not implemented** (no match in `app.py`) **[V]** | Not implemented |
| R7 | Fast generation; app must not hang on "loading" | Shipped slow/hangs on hard cases | Candidate: 1.7–4.3 s per 2-floor building in the tested scenario **[V]** |
| R8 | Apply a long building-code-style rule list | See below | – |

**R8 detail (user's rule list, my classification, agreed in chat):** *Directly actionable, not yet done:* horizontal zoning (public/private wings), vertical stacking of bathrooms across floors, wet-wall clustering. *Partially actionable (user said defer):* room area minimums and corridor widths (need a metres-per-subgrid scale that does not exist), centralized stair core. *Out of scope (no 3-D/structural/plumbing model exists):* ceiling height, drainage slope, waterproofing, glazing %, ventilation shafts, stair riser/tread, structural continuity, foundations, roof trusses.

**Current objective:** validate and ship the candidate (§10, task 1), then the connectivity rules R1/R5, then disclaimer + docs.

---

## 2. Technology stack, architecture, dependencies

**Stack [C/V]:** Pure Python. UI = Streamlit. Third-party imports found by scanning all `src/`, `app.py`, `tests/`, `experiments/`: `numpy`, `matplotlib`, `streamlit`, `pytest`. **No `networkx`, torch, scipy, sklearn, pandas** (earlier chat summaries wrongly listed networkx). **No database, no HTTP API, no ORM [V: grep]**. No `requirements.txt` / `pyproject.toml` exists in the shipped files **[V]**.

**Versions seen:** sandbox = Python 3.12.3, numpy 2.4.4, matplotlib 3.10.8, streamlit 1.65.0, pytest 9.1.1 **[V]**. The user's own venv traceback shows **Python 3.14** at `/home/alfredbaju/Desktop/Code/MainProject/nwfc-floorplan/venv` **[C: screenshot]**; compatibility with 3.14 beyond "the app ran" is **[A]**.

**Architecture (bottom → top):**

1. **Tileset** `tileset_core.json`: 1968 tiles; each tile has 4 edge types (`e_n,e_s,e_w,e_e`), a `room_type`, a `weight`. Edge types `W D O C W_EXT`. Room types `BED BATH KIT LIV COR ENT EXT STAIR BALC`. `adjacency_compatibility` is the "Layer 2" room-type door-compatibility table **[V]**.
2. **`wfc_core.WFCGrid`** — baseline WFC (Layer 1 = edge matching). Optional `enforce_layer2` removes door tiles between incompatible room types.
3. **`nwfc.solve_nwfc`** — N-WFC: grid of A×B subgrids, each C×C (C=4), solved in diagonal order, adjacent subgrids share a row/column. Full grid = `A*(C-1)+1` by `B*(C-1)+1` tiles. Accepts `room_plan` (macro plan restricting each subgrid to one room type).
4. **`zoning.py`** — the project's own macro layer (not in the paper): picks the room type of each A×B subgrid ("macro plan"), sizes rooms, then drives `solve_nwfc` with retries until exact room counts match.
5. **Post-processing** (`wall_cleanup`, `repair`, `entrance_cap`, `room_connections`) fixes walls, connectivity, entrances, doors.
6. **`multi_floor.py`** — splits rooms across floors, shares one grid size and one staircase macro position across floors.
7. **`app.py`** (Streamlit) + `visualize.py` render floors and area breakdown.

**Pipeline order for one floor (multi-floor path) [C: multi_floor.py]:** `solve_zoned_with_exact_counts` → `cleanup_cross_room_edges` → `enforce_internal_room_connectivity` → `cap_entrance_count` (ground floor) → `repair_connectivity` (stairs as entrance on upper floors) → `ensure_entrance_on_border` → `enforce_room_connection_rules` → `repair_connectivity` → `unify_corridor_network` → `validate`.

---

## 3. Directory and file structure

Expected run layout (the code hard-codes `data/tileset_core.json` relative to the project root **[C]**):

```
nwfc-floorplan/
├── app.py                     Streamlit UI (24 KB)
├── data/tileset_core.json     REQUIRED. Shipped copy sits at outputs/tileset_core.json, NOT in outputs/data/  [V]
├── data/context_scorer.npz    OPTIONAL learned scorer; absent in shipped files; app falls back to None [C]
├── src/  (see table)
├── tests/
└── experiments/               e1..e5 benchmark scripts + common.py (results docs dated Aug, pre-date zoning work)
```

| File | Purpose (from module docstring/code) |
|---|---|
| `src/zoning.py` (109 KB shipped) | Macro planner and sizing. Key functions: `compute_room_sizes_and_grid`, `_recheck_corridor_for_actual_grid`, `plan_subgrid_assignment`, `solve_zoned_with_exact_counts`, footprint claimers (`_claim_footprint_regular`, `_claim_footprint_best_of`, `_claim_border_strip_footprint`), `count_actual_room_instances`, `ensure_entrance_on_border` |
| `src/dense_placement.py` | Reserved-slot placement of paired bedroom+bathroom "suites" (`reserve_bed_bath_layout_paired`). **Imported by `zoning.py`; must be copied with it** (a missing copy caused `ModuleNotFoundError` on the user's machine) |
| `src/multi_floor.py` | `distribute_rooms_across_floors`, `generate_multi_floor_building`, `FloorResult` (fields: `floor_number, grid, is_ground_floor, structural_valid, reachable, unreachable_cells, room_counts_actual, stair_positions`) |
| `src/nwfc.py`, `src/wfc_core.py` | N-WFC solver and baseline WFC engine (paper algorithm). Shipped versions untouched by zoning work; candidate modifies both (§8) |
| `src/tileset.py` | Load/validate tileset, `is_room_compatible` |
| `src/validity.py` | Layer-2 validation + reachability (`validate`) |
| `src/repair.py` | Connectivity repair, `unify_corridor_network`, `enforce_internal_room_connectivity` |
| `src/room_connections.py` | Room-to-corridor connection enforcement |
| `src/living_room_door.py` | Docstring: "Enforces exactly one door for a given room instance". **Possibly relevant to R5; where/whether it is called was not checked [A]** |
| `src/entrance_cap.py`, `src/wall_cleanup.py`, `src/graph_utils.py` | Entrance-count cap; wall cleanup; room-adjacency graph / `find_room_components` |
| `src/visualize.py` | Matplotlib floor-plan rendering, `compute_area_breakdown`, PDF export |
| `src/heuristic_model.py`, `src/train_heuristic.py`, `src/load_resplan.py` | Optional learned selection heuristic (numpy only); ResPlan dataset loader (dataset availability not verified **[A]**) |
| `src/demo_*.py` | Small demos |
| `tests/test_nwfc.py`, `test_validity.py`, `test_wfc_core.py` | "Core" suite: 70 pass, 2 skip **[V]**, ~7 s |
| `tests/test_new_features.py` | 25 tests in 5 classes (`TestBalcony`, `TestSecondEntranceNearKitchen`, `TestKitchenIsolation`, `TestRoomSizingRules`, `TestTimeBudget`); slow |
| `outputs/*.md, *.pptx, *.docx` | Proposal/review/status docs. **Several are stale (§6)** |
| `outputs/PENDING_CANDIDATE/` | Candidate files, patch, eval scripts (§8) |

---

## 4. Database schema, APIs, important data flows

**Database: none. Network API: none [V].** The only persistent data are JSON files.

**Data formats [C]:**
- `tileset_core.json`: `{edge_types[5], room_types[9], adjacency_compatibility{room→{room→float|bool}}, tiles[1968]}`; tile = `{id, room_type, e_n, e_s, e_w, e_e, weight}`.
- `room_counts`: `{"BED":int,"BATH":int,"KIT":int,"LIV":int,"BALC":int,"STAIR":int}`. App passes totals; `distribute_rooms_across_floors` splits them (LIV is forced to 1 on **every** floor; BALC only on upper floors; every other key (e.g. BED, BATH, KIT) is split evenly with the remainder on lower floors).
- Macro plan: `dict[(row,col)] → room-type string | "CORRIDOR" | "CORRIDOR_ENTRANCE"`.
- Floor grid: 2-D list of tile-id strings.

**Main data flow [C]:** UI inputs → `generate_multi_floor_building` → per floor `compute_room_sizes_and_grid` (only to size the shared grid; shared A,B = max over floors) → `solve_zoned_with_exact_counts(A,B,…)` → **re-derives room sizes internally** and rechecks them against the shared grid (`_recheck_corridor_for_actual_grid`) → loop: plan macro layout → `solve_nwfc(room_plan)` → count room instances → accept when exact counts and quality checks pass or budget expires → post-processing → `FloorResult`.

---

## 5. Features and verified status

**Verified this session [V] (shipped baseline = `outputs/src`):**

| Item | Result |
|---|---|
| Core suite (70 tests) | 70 passed, 2 skipped |
| `test_new_features.py` on shipped | Run twice: a 4-process parallel run (**1-CPU sandbox, so ~4x slowdown; generation/timing results from it are unreliable**) and a serial rerun of the unfinished/timing tests. **Confirmed in the serial run:** `test_balcony_never_on_ground_floor` FAILED (`RuntimeError: Failed to generate floor 2 after 30 attempts`); `test_time_budget_never_sacrifices_a_valid_result_for_speed` FAILED (4/5 succeeded); `test_multi_floor_hard_scenario_stays_bounded` PASSED (it had failed only under contention). **Deterministic failures (pure sizing functions, stale assertions):** `test_kitchen_at_least_15_percent_when_requested`, `test_multiple_balconies_total_area_is_twice_living_room`. **Seen failing only in the contended run, not re-checked serially:** `test_balcony_never_directly_connects_to_bath_kit_stair`, `test_balcony_on_exterior_wall`, `test_solvability_and_counts_unaffected_by_second_entrance`. Generation tests are seed/time dependent, so treat them as flaky. |
| Screenshot scenario (5 bed, 5 bath, 1 kit, 1 liv, 2 balc, 2 floors), macro level, shipped planner | Kitchen planned 16 cells; delivered avg 9.8, and 1 cell in the worst seed; touching bed/bath in 17/20 plans; floor-2 corridor avg 28.8%, max 59.4% |

**Candidate state [V]** (same scenario unless stated):

| Item | Result |
|---|---|
| Core suite | 70 passed, 2 skipped |
| Full pipeline, 6 seeds, 2 floors | 6/6 succeed, 1.7–4.3 s each; exact counts floor 1 (3 bed, 3 bath) and floor 2 (2 bed, 2 bath, 2 balc) in every seed |
| Floor 1 | corridor 16.6%, living 19.2%, bed 30.9%, bath 13.9%, kitchen 17.3%; bath adjacent to bed 6/6; kitchen touching bed/bath 0/6 |
| Floor 2 | corridor 12.3%, living 19.9%, bed 36.6%, bath 9.2%, balcony 20.1%; bath adjacent to bed 5/6 |
| Tile-level solve with layer-2 on, same planner, floor 2 | **shipped solver 0/14, candidate solver 14/14** (0.5–0.8 s each); floor 1 14/14 both |
| `test_new_features.py` | 23 pass, 2 fail (the two stale sizing tests), 59 s serial |

**Prior-session claims, NOT re-run this session [P]:** high-density 6 bed + 7 bath → 6/6 exact counts; typical 3 bed/2 bath 0/6 failures; `AppTest` clean in both app modes; entrance-to-living adjacency; second-entrance-to-kitchen rates; E1–E5 experiment numbers (dated Aug 25, predate zoning rewrites). Re-run them.

**Feature inventory in shipped code [C]:** reserved-slot bedroom/bathroom suites (`bed+bath ≥ 6`); sparse-floor suite reservation (`0 < bath ≤ bed ≤ 2`); 50%-bedroom / equal-kitchen-living sizing; corridor target 10% and recheck against shared grid; living cap 30% on floors with balconies (three places); balcony border-strip grower; balcony size cap when living room present; kitchen corner/zero-bed-bath seed preference plus post-growth selection by touching ratio and fill ratio; second entrance near kitchen; staircase forced to identical macro position across floors; `density_multiplier = 2` when `BED+BATH ≥ 8`; per-attempt tile-solve cap 7.0 s.

---

## 6. Bugs, errors, unresolved issues, and conflicting information

**A. Problems in the shipped state (the user's screenshots) [V]**
1. Kitchen tiny (2.9%) and touching rooms; corridor 26% (floor 1) / 40% (floor 2); floor-2 bedrooms/bathrooms small and bathroom not beside bedroom. Root causes: greedy planner silently accepts truncated rooms; kitchen squeezed into leftover pockets; shared grid set by the dense floor leaves floor 2 oversized; bedrooms capped at 6 cells; sizing assumed kitchen 16 + living 17 fit where ~half fits.
2. Generation hard-failures / long waits. Messages seen: `RuntimeError: Failed to generate floor 2 after 30 attempts.`, `N-WFC exceeded 110 exterior backtracks attempting to resolve a conflict at subgrid (1,1)`, `N-WFC exceeded its 7.0s time budget (after N exterior backtracks)`. Mechanism: with `enforce_layer2=True` many macro plans are unsolvable; solves are either <1 s or run to the cap, so a ~36–48 s floor budget allows only ~6 attempts.
3. User-reported: app sometimes "keeps loading" (consistent with item 2).

**B. Unresolved in both states**
- R1 "corridor is a thin connecting path / single network linking every room and stairs / extends across floors": not implemented. Dense floors keep ≈15–18% corridor because of suite gaps plus the kitchen's corridor ring; 10% is **not** achieved there **[V]**.
- R5 one-door-per-room; R6 disclaimer; vertical bathroom stacking; horizontal zoning; wet-wall clustering: not implemented.
- Floor-2 bath adjacent to bedroom 5/6 in candidate **[V]**.
- Two stale tests (assert 15% kitchen minimum; balcony total = 2× living) fail deterministically **[V]**: kitchen share 12.0% for `{BED2,BATH2,KIT1,LIV1}`; balcony ratio 0.33 vs 0.5. They encode rules replaced by the later sizing rewrite. Update them only after confirming the intended rule with the user.
- Candidate tested on **one scenario only** (see §8).

**C. Conflicting / outdated information**
| Old statement | Reality |
|---|---|
| README: status boxes all unchecked; lists `requirements.txt`, `docs/`, `paper/`, `data/reference_plans/`; path `~/projects/nwfc-floorplan` | Those files/dirs are not in the shipped files; user path differs **[V]**. README is stale |
| `PROJECT_STATUS.md`: "~72% complete"; kitchen-living adjacency "6 of 8 (75%)" shipped as a rule | Adjacency rule was **reversed** by the user (kitchen must NOT touch living). Percent is a judgement, not a measurement. Stale |
| Earlier chat: stack includes networkx | Not imported anywhere **[V]** |
| Earlier chat: "balcony + high density reliability gap pre-dates my work, deep core-solver issue, unfixable" | True only of the last shipped code, which already contained the prior session's tight-packing sizing. The candidate solver change fixes it in tests (0/14→14/14), so it was not necessarily "deep" |
| Earlier chat: kitchen never-adjacent is "a genuine geometric limit" | Limit of the *greedy* approach only; a reserved rectangle + corridor ring achieves 0/6 **[V]** |
| Earlier chat: sandbox `/home/claude/nwfc` as the project | It is a disposable restore; file timestamps are unreliable |

**D. Cannot be verified from available information (explicit list)**
1. **Provenance of the candidate's edits to `nwfc.py` and `wfc_core.py`** (neighbour-type door look-ahead). They were present in the sandbox with a modification time of this session, but I did not make them in the visible chat history and cannot say who/what did. Their effect is measured (§5); their origin is not.
2. Contents of the user's own local project (is `data/tileset_core.json` current? is `dense_placement.py` present? any local edits?).
3. Whether the candidate works in the Streamlit UI (no `AppTest` run on it).
4. Whether `load_resplan.py`'s dataset and `data/context_scorer.npz` exist anywhere.
5. Where `living_room_door.py` is called, and its effect.
6. Behaviour of the candidate on scenarios other than the one tested (1 or 3 floors, other room counts, multiple kitchens, no kitchen, 4+ bedrooms on one floor).
7. Whether the 4-process parallel test run results are meaningful (the sandbox has **1 CPU**; contention made the `TestTimeBudget` results unreliable until rerun serially).

---

## 7. Decisions and constraints that must not be changed (without the user's say-so)

1. **Scope = schematic only.** Never claim building-code compliance.
2. **Paper fidelity:** `nwfc.py`/`wfc_core.py` implement the paper's algorithm ("clean-room"). The zoning layer is the project's own contribution. Any solver change (including the candidate's look-ahead) must be flagged as a modification to the paper's algorithm and justified.
3. **Staircase macro position identical on all floors**; all floors share one A×B grid = max of per-floor needs.
4. **`enforce_layer2` stays default True** (quality). Note: `solve_nwfc` defaults to **False** but the pipeline passes True — isolated tests that omit it are misleading.
5. **Reserved-slot suites are required at high bed/bath density** (planner threshold `bed+bath ≥ 6`; `multi_floor` multiplier threshold is separately `≥ 8`). They are separate thresholds in separate files **[C]**; if either is changed, re-run the full verification in §7-9.
6. **Reliability ceiling `max_single_room_size = 6`** per bedroom/bathroom instance when there are many instances (large instances caused placement failures). The candidate allows 12 only when a floor has ≤ 2 bedrooms.
7. **User-fixed layout rules** (R1–R4 in §1). Specifically the kitchen/living rule was reversed once: kitchen must **not** touch living room. Do not re-introduce open-plan.
8. **Ship files together.** `zoning.py` imports `dense_placement.py`; the candidate needs `zoning.py`, `nwfc.py`, `wfc_core.py` together.
9. **Verification discipline (learned the hard way):** isolated checks repeatedly gave false confidence. Always confirm with (a) core suite, (b) the full `generate_multi_floor_building` path with multiple seeds, (c) the slow suite run **alone**, (d) `AppTest` for both app modes, before shipping or claiming a fix.
10. **Dead ends — do not repeat:** reordering kitchen before bedrooms (no effect); biasing bathroom placement toward a prior floor's position via `preferred_positions` (regressed ensuite adjacency 100%→62.5%; the plumbing still exists, unused); ground-floor-only kitchen/living redistribution (caused 7 test failures, reverted); growing living room without a cap (absurd 30–52-cell rooms); shrinking the corridor spine for ensuite space (no reliable effect); raising `density_multiplier` budgets (just fails slowly).

---

## 8. Recent changes and known regressions

**Shipped (outputs, by file mtime) [C]:** `zoning.py` and `multi_floor.py` 24 Sep; `dense_placement.py` 14 Sep. Contains the §5 feature inventory. `multi_floor.py` contains unused `bath_positions`/`preferred_positions` plumbing.

**Candidate (unshipped), `outputs/PENDING_CANDIDATE/` [V that files parse; behaviour per §5]:**
- `zoning.py` (136 KB) adds: `score_macro_plan` (penalties: truncated rooms, kitchen contact/shape/border/corner/distance to living, bath not beside bed, balcony off edge, living > 30% with balcony, corridor share and connectivity, bedroom aspect ratio, bedroom/kitchen not touching corridor); `plan_subgrid_assignment_best` (best of 40 candidate plans per attempt); `_reserve_kitchen_with_corridor_ring` (corner rectangle plus corridor ring, reserved before the spine; entrance/living steered to the far side; second-entrance logic preserved via `kitchen_reserved`); `_adapt_sizes_to_fit` (probes planner, shrinks kitchen/living/balcony so planned sizes are deliverable, keeps kitchen ≤ living/1.1; cached in `_ADAPTED_SIZE_CACHE`); in `_recheck_corridor_for_actual_grid` a bedroom ceiling of 12 for ≤ 2 bedrooms and balcony-before-living growth order on upper floors.
- `nwfc.py` + `wfc_core.py`: `south_neighbor_types`/`east_neighbor_types` look-ahead that removes door tiles in the shared last row/column when the not-yet-generated neighbour subgrid's planned type cannot legally take a door (active only with `enforce_layer2`). **Provenance unverified (§6-D1).**
- `candidate_vs_shipped.patch` (unified diff of those three files) and `eval_scripts/` (`full_eval.py`, `macro_eval.py`, `timing.py`; run from project root, they `sys.path.insert(0,'src')`).

**Known regressions / risks in the candidate [A unless noted]:** only one scenario verified; best-of-40 planning adds ~0.1–0.3 s per attempt; the look-ahead could over-constrain unusual layouts; first call per room-count signature runs the size-adaptation probe. The earlier session's regressions (balcony/high-density generation failures; time spikes of 22–46 s) are real in the shipped state **[V]**.

---

## 9. Install, run, test (exact steps)

```bash
# 1. layout
mkdir nwfc-floorplan && cd nwfc-floorplan
cp -r <outputs>/src <outputs>/tests <outputs>/app.py .
mkdir data && cp <outputs>/tileset_core.json data/tileset_core.json   # REQUIRED location

# 2. environment (no requirements.txt exists; these are the only third-party imports)
python3 -m venv venv && . venv/bin/activate
pip install numpy matplotlib streamlit pytest

# 3. run the app
streamlit run app.py        # tick "Control exact room counts" for the zoned mode

# 4. fast tests (~7 s)
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -p no:cacheprovider \
    tests/test_nwfc.py tests/test_validity.py tests/test_wfc_core.py -q
# expected: 70 passed, 2 skipped

# 5. slow suite — run ALONE (never alongside other CPU work)
python3 -m pytest -p no:cacheprovider -v --tb=line tests/test_new_features.py
# candidate: ~1 min, 23 pass / 2 stale fails. Shipped baseline: several minutes, flaky generation failures

# 6. reproduce the candidate numbers
python3 <outputs>/PENDING_CANDIDATE/eval_scripts/full_eval.py 0 1 2 3 4 5   # full pipeline, per-floor metrics
python3 <outputs>/PENDING_CANDIDATE/eval_scripts/timing.py                   # tile-solve success/time, floors 1 and 2
python3 <outputs>/PENDING_CANDIDATE/eval_scripts/macro_eval.py               # macro-plan metrics
```

To trial the candidate: copy `PENDING_CANDIDATE/{zoning,nwfc,wfc_core}.py` over the same names in `src/` **together**, then repeat steps 4–6 and run `AppTest`:

```python
from streamlit.testing.v1 import AppTest
at = AppTest.from_file('app.py'); at.run(timeout=30); at.button[0].click().run(timeout=60); print(at.exception)
```

Pitfalls: the sandbox has 1 CPU (don't parallelize timing-sensitive tests); `ModuleNotFoundError: dense_placement` means `src/dense_placement.py` was not copied; a missing `data/tileset_core.json` breaks the app and every test.

---

## 10. Next three tasks, in priority order

**1. Validate and promote the candidate.**
   (a) Decide whether the `nwfc.py`/`wfc_core.py` look-ahead is acceptable and document its origin (§6-D1).
   (b) Run a regression matrix: 1/2/3 floors; several room-count mixes (include no kitchen, 4–6 bedrooms, 0/1/2/3 balconies, high-density 6 bed + 7 bath); ≥ 8 seeds each; record success, time, exact counts, and the §5 metrics.
   (c) `AppTest` both modes; slow suite alone; update the two stale tests once the user confirms the intended kitchen/balcony rules.
   (d) Ship the three files together; then refresh README/`PROJECT_STATUS.md`.
   *Done when:* matrix shows no hard failures and no count mismatches, and the user confirms the new screenshots.

**2. Corridor connectivity rules (R1 + R5), the user's most repeated asks.**
   Make the corridor a single connected network of thin paths touching every room and the staircase, one door per room into it (check `living_room_door.py`, `room_connections.py`, `repair.py` first), and make it visibly continuous between floors. Accept and state honestly that dense floors will stay near 15–18% corridor unless suite packing in `dense_placement.py` is improved.

**3. Robustness and remaining quality gaps.**
   Raise floor-2 bath-beside-bed from 5/6 to 6/6 (scoring weight or planner fix); add the "schematic concept only" disclaimer to `app.py` (R6); add `requirements.txt`; then, one at a time, the directly-actionable rules (vertical bathroom stacking — previous attempt regressed, horizontal zoning, wet-wall clustering), each gated by the §7-9 verification discipline.
