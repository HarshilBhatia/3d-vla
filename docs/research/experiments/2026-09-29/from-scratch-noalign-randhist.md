# From-scratch control, and the history-register test

**Status:** active — A at 178k of 200k (26 Sept); C finished 150k
**Week introduced:** [22–29 Sept](../../weekly_logs/2026-09-29.md)

## Question

Two questions on one pair of runs.

1. **From-scratch control.** The warm-started ablations
   ([no view align](pooled-no-view-align.md),
   [random history](random-history-no-time.md)) inherit weights trained with
   delta_M and ordered history. Does the result survive random init?
2. **Register test.** The **history register** is one vector per sample. It
   cross-attends over all pooled image tokens and is added to every trajectory
   query. It has no 3D position, so it is invariant to extrinsics — it is the
   only position-free route from external-camera appearance to the actions.
   Is it the channel behind the robustness?

## Setup

| Run | Start | Register | Config | Job |
|---|---|---|---|---|
| A | from scratch, 200k | on | `peract2_orbital_vid_pooled_noalign_randhist_scratch` | 3971512 |
| C | warm-started from A at 50k, then 150k | **off** | `peract2_orbital_vid_pooled_noalign_randhist_noreg_from_a50k` | 3972595 |

- Both: no delta_M, random unordered history, no time embedding. Batch 128.
- A runs on grogu-2-25 (shubhamlong, 4 GPUs, 400 GB), automatic ladder from 50k.
- The flag is `video_deltam_history_register: false` (commit `0f62a5f`); it
  never builds the readout, so no parameters go unused.
- **Step numbering:** C's ladder counts from its own step 0, so C's 50k is 100k
  of total training.
- To make room, [TODO 1](pooled-no-view-align.md) and
  [1b](fullpatch-d8.md) were stopped at about 281k of 300k.

## How to read it

If C collapses toward Base at IDLT 60/60 while A holds about 71, the register
is the channel behind the robustness.

## Results so far (26 Sept, 00:45)

- SBRS ladders: **A best 80 at 160k; C best 79 at its 100k** (150k total).
  There is no in-distribution difference.
- Robustness evals submitted on A 160k and C 100k, using the same protocols and
  campaign ids as the TODO 1/5 reruns (plans `*_regtest.json`, 430 cells):
  G7 arrays 3974081–3974083, IDLT arrays 3974084–3974103.
- EMA fine-tunes put both about 7 points below the warm-started models:
  A 64.0 → 71.5, C 63.0 → 72.5 ([ema-finetunes.md](ema-finetunes.md)).

**Open:** the robustness verdict waits on those 430 cells.
