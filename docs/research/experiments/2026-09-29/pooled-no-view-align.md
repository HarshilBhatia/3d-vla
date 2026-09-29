# History with no delta_M refinement

**Status:** stopped at 281k of 300k (24 Sept, to free capacity; `last.pth` kept, resumable)
**Week introduced:** [22–29 Sept](../../weekly_logs/2026-09-29.md) (TODO 1)

## Question

Does temporal history carry the robustness, or does view alignment? Pooled
video history still refines the camera-summary tokens and the register, but
`view_align_mode: none`, so no delta_M is predicted anywhere.

## Setup

| | |
|---|---|
| Config | `config/experiment/peract2_orbital_vid_deltam_pooled_no_view_align.yaml` |
| ARM | `video_deltam_pooled_no_view_align` |
| Start | warm start from VideoPooled 84k, 300k steps |
| Job | 3966423 (shubhamlong, 4 GPU) |
| Robustness plans | `instructions/eval_plans/*_todo15.json` |

This config has no automatic `checkpoint_evals`; ladders were run by hand.

## Results

**Matches VideoPooled everywhere.**

| Suite | No delta_M | VideoPooled 84k | Base 160k |
|---|---:|---:|---:|
| IDLT held-out 60°/60 cm | 71.4 | 71.5 | 27.1 |
| G7 extremes | within ±5 of VideoPooled | — | — |
| SBRS best | 83.3 at 140k | — | — |

EMA fine-tune (see [ema-finetunes.md](ema-finetunes.md)): raw 69.3 → EMA 77.5.

Current-frame-only probe at the G7 extremes (normal → current-only, 6 tasks ×
20 rollouts), from [history-probes.md](history-probes.md):

| tr 100 | tr 200 | tr 500 | rot 20 | rot 90 | rot 180 | rot 270 |
|---|---|---|---|---|---|---|
| 78→55 | 78→54 | 66→47 | 80→51 | 74→49 | 75→53 | 70→53 |

## Reading

View alignment is not what buys the robustness — removing it costs nothing on
any robustness suite. Like VideoPooled, this model loses 15–29 points when its
history is replaced by copies of the current frame, so it does use history.

## Caveats

- It inherits a backbone trained **with** view alignment on. The from-scratch
  control is [from-scratch-noalign-randhist.md](from-scratch-noalign-randhist.md).
- The pooled history module sees appearance only
  (`video_deltam_patch_rope3d: false`), so it cannot detect miscalibration
  geometrically.
