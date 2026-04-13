# Orbital Rollout Collection Stats

**Job:** `38621910` | **Date:** 2026-04-08  
**Tasks:** 44/54 succeeded | **Total episodes saved:** 1,467

---

## Failed Tasks (0 episodes saved)

| Task | Group |
|------|-------|
| `push_buttons` | G3, G4, G5 — **all groups missing, no data at all** |
| `place_wine_at_rack_location` | G4 |

## Short Groups (partial saves)

| Task | Group | Episodes |
|------|-------|----------|
| `open_drawer` | G1 | 24/30 |
| `place_shape_in_shape_sorter` | G2 | 25/30 |
| `stack_blocks` | G4 | 25/30 |
| `put_money_in_safe` | G1 | 28/30 |
| `reach_and_drag` | G2 | 28/30 |
| `stack_cups` | G4 | 28/30 |
| `turn_tap` | G2 | 24/30 |
| `turn_tap` | G6 | 28/30 |
| `place_wine_at_rack_location` | G2 | 29/30 |
| `put_groceries_in_cupboard` | G4 | 29/30 |
| `meat_off_grill` | G1 | 29/30 |

## Per-Task Episode Counts

| Task | Groups | Episodes |
|------|--------|----------|
| `close_jar` | G2, G3, G4 | 90 ✓ |
| `insert_onto_square_peg` | G3, G4, G5 | 90 ✓ |
| `light_bulb_in` | G4, G5, G6 | 90 ✓ |
| `place_cups` | G1, G2, G3 | 90 ✓ |
| `put_item_in_drawer` | G1, G5, G6 | 90 ✓ |
| `slide_block_to_color_target` | G2, G3, G4 | 90 ✓ |
| `sweep_to_dustpan_of_size` | G1, G5, G6 | 90 ✓ |
| `put_groceries_in_cupboard` | G4, G5, G6 | 89 |
| `meat_off_grill` | G1, G5, G6 | 89 |
| `stack_cups` | G4, G5, G6 | 88 |
| `put_money_in_safe` | G1, G2, G6 | 88 |
| `reach_and_drag` | G1, G2, G3 | 88 |
| `stack_blocks` | G3, G4, G5 | 85 |
| `place_shape_in_shape_sorter` | G1, G2, G3 | 85 |
| `open_drawer` | G1, G2, G6 | 84 |
| `turn_tap` | G1, G2, G6 | 82 |
| `place_wine_at_rack_location` | G2, G3 | 59 |
| `push_buttons` | — | 0 |

## Failure Cause

All failures caused by: `The X11 connection broke (error 1). Did the X11 server die?`  
Xvfb virtual display server crashed during CoppeliaSim simulation.

> Note: 9 tasks reported exit code 1 but successfully saved all 30 episodes — the error was an xvfb-run cleanup glitch after collection finished.
