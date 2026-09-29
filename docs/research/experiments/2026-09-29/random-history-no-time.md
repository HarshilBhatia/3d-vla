# Random history, no time embedding

**Status:** active (warm-start run complete to 100k; EMA fine-tune done)
**Week introduced:** [22–29 Sept](../../weekly_logs/2026-09-29.md) (TODO 5)

## Question

Does the history need to be *ordered and recent*, or is any set of frames from
the demo enough?

Same config as the VideoPooled 84k model, except:

- `visual_history_sampling: random` — the history is the current keyframe
  (always last) plus 4 distinct, randomly chosen earlier keyframes from the
  same demo, in random order. Short histories are padded with the demo's first
  frame.
- `video_deltam_time_embedding: false` — no time embedding and no causal mask;
  one learned vector marks the current frame.
- Eval samples history the same way, with a fixed seed per episode.

## Setup

| | |
|---|---|
| Config | `config/experiment/peract2_orbital_vid_deltam_pooled_randhist_notime.yaml` |
| ARM | `video_deltam_pooled_randhist_notime` |
| Start | warm start from VideoPooled 84k, 100k steps, batch 128 |
| Ladder | automatic, from 50k |
| Jobs | 3966716 (4 nodes × 1 GPU on `all`, `scripts/multinode/config/vid_deltam_pooled_randhist_notime.yaml`), then 3966755 → 3966903 |

A supervisor loop resubmits after each 24 h round and training resumes from
`last.pth`. A 4-GPU single-node job (3966643) started and was cancelled by
mistake; its ~4 minutes of progress are in `last.pth`.

## Results

**Matches VideoPooled's robustness.** IDLT held-out 60/60: **71.9** against
VideoPooled 71.5 and Base 27.1. G7 extremes within about ±5. It is 3–4 points
lower at low noise.

EMA fine-tune is the largest gain of any arm: raw 67.8 → **EMA 79.0**
(see [ema-finetunes.md](ema-finetunes.md)).

Current-frame-only probe (normal → current-only, 6 tasks × 20 rollouts):

| tr 100 | tr 200 | tr 500 | rot 20 | rot 90 | rot 180 | rot 270 |
|---|---|---|---|---|---|---|
| 76→71 | 73→71 | 66→71 | 78→82 | 77→72 | 80→72 | 76→75 |

## Reading

**Random history barely uses history** — replacing it with copies of the
current frame costs nothing, while ordered-history models lose 15–29 points —
**yet it matches their robustness.** So the robustness likely comes from the
video training and architecture, not from history content or delta_M.

## Caveats and open questions

- Demos are short (validation median is 7 keyframes), so the random set often
  contains nearly the same frames consecutive history would. This mostly tests
  "no order information".
- Warm-started from VideoPooled, which was trained with delta_M and ordered
  history. From-scratch control:
  [from-scratch-noalign-randhist.md](from-scratch-noalign-randhist.md).
- **Open: make history sampling harder?** With replacement, drop the previous
  keyframe, or dense timesteps.
