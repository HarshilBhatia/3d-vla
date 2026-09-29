# FullPatch depth 8

**Status:** stopped at 281k of 300k (24 Sept, to free capacity; `last.pth` kept)
**Week introduced:** [22–29 Sept](../../weekly_logs/2026-09-29.md) (TODO 1b)

## Question

Does doubling the history stack's depth (4 → 8) help the full-patch variant?

## Setup

| | |
|---|---|
| Config | `config/experiment/peract2_orbital_vid_deltam_fullpatch_d8.yaml` |
| Start | warm start from VideoPooled 84k, 300k steps, batch 64 |
| Job | 3966489 (shubhamlong, 4 GPU) |
| Ladder | automatic, from 50k |

## Results

SBRS best **79.2 at 140k**, 60–72 at later steps. Early ladder (6 SBRS tasks at
e3/1, K=5):

| Step | push_box | pick_laptop | lift_ball | straighten_rope | pick_plate | handover_item_easy | Mean |
|---|---|---|---|---|---|---|---|
| 50k | 95 | 45 | 100 | 25 | 45 | 95 | 67.5 |
| 60k | 100 | 15 | 100 | 35 | 15 | 95 | 60.0 |

## Reading

**Not better than depth 4.** `pick_plate` and `pick_laptop` are the weak tasks.
Decision on 22 Sept was to run it to completion rather than gate it early; it
was then stopped at 281k for capacity, not for performance.

## Caveats

- The logged **total** training loss is `nan` while the component losses are
  fine. The parent FullPatch run shows the same thing, so it is a logging
  quirk, not a divergence.
- The older FullPatch ladder mixed K=1 and K=5 evals, so the "76.7 best
  FullPatch" gate from that ladder is unreliable.
