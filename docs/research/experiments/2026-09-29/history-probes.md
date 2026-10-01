# What does VideoPooled's robustness depend on?

**Status:** done (eval-only probes, submitted and read out 24 Sept)
**Week introduced:** [22–29 Sept](../../weekly_logs/2026-09-29.md)

## Question

Three eval-only interventions, no retraining, to find which input channel the
robustness actually rides on. All run at G7 opposing translation 200 cm
(6 SBRS tasks × 20 rollouts), where VideoPooled scores 78.3, random history
73.3 and Base 42.5.

| Test | Models | What changes at eval | Array |
|---|---|---|---|
| `hist_current_only` | VideoPooled 84k, random history 50k | All 4 past visual slots hold the current frame | 3971246 |
| `hist_other_episode` | VideoPooled 84k, random history 50k | Past slots come from the previous episode (a different scene). The first episode per task falls back to current-only | 3971247 |
| `external_blackout` | VideoPooled 84k, Base 160k | Orbital cameras get black RGB and their points are parked 10 m below the table; wrist cameras untouched | 3971248 |

- **Code:** eval-owned keys `eval_history_probe` and `eval_blackout_cameras`
  (commit `fea9968`).
- **Plans:** `instructions/eval_plans/g7_opposite200_probe_*_v01.json`.

## Results

6-task mean at 20 rollouts per task.

| Test | VideoPooled 84k | Random history 50k | Base 160k |
|---|---|---|---|
| Normal | 78.3 | 73.3 | 42.5 |
| `hist_current_only` | 52.5 | 70.8 | — |
| `hist_other_episode` | 80.8 | 65.0 | — |
| `external_blackout` | 16.7 | — | 5.8 |

Current-frame-only across the G7 extremes (each cell reads normal →
current-only; 6 tasks × 20 rollouts):

| Model | tr 100 | tr 200 | tr 500 | rot 20 | rot 90 | rot 180 | rot 270 |
|---|---|---|---|---|---|---|---|
| VideoPooled 84k | 72→53 | 78→58 | 68→52 | 78→58 | 78→53 | 72→49 | 79→52 |
| No delta_M 140k | 78→55 | 78→54 | 66→47 | 80→51 | 74→49 | 75→53 | 70→53 |
| Random history 50k | 76→71 | 73→71 | 66→71 | 78→82 | 77→72 | 80→72 | 76→75 |

The 200 cm cell was measured twice: VideoPooled scored 52.5 and 58 in the two
campaigns, which is within noise.

## Reading

- **The external cameras are essential** (blackout → 16.7). The robustness is
  not wrist reliance.
- **VideoPooled needs distinct past frames, but not this episode's**
  (other-episode 80.8 ≥ normal): history acts as a task-progress cue.
- **Random history barely uses history** yet matches the robustness. So the
  robustness likely comes from the video training and architecture, not from
  history content or delta_M.
- Models trained with ordered history lose 15–29 points everywhere when the
  history is flattened; random history is unaffected (71–82).

## Current-frame-only on SBRS, EMA checkpoints (submitted 1 Oct)

Random history EMA 15k and A (no delta_M + random history) EMA 20k, SBRS (trained bias + 3/1, 6 tasks × 20),
`eval_history_probe: current_only`. Pairs with the EMA ladder cells at the same steps. Plan
`instructions/eval_plans/sbrs_hist_current_only_ema_v01.json` from `scripts/eval/create_sbrs_current_only_ema_plan.py`; array 3984479.
