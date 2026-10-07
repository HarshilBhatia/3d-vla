# Direct delta_M on random history (Exp 2)

**Status:** done (6 Oct)
**Week introduced:** [30 Sept – 6 Oct](../../weekly_logs/2026-10-06.md)

## Main idea

The Direct role (`video_deltam_role: predict_delta_m`), now trained on random
history. The history extractor takes each camera's current C-token after it has
mixed with the other cameras and with that camera's own past frames, and
predicts a 6×6 delta_M from it. delta_M rotates that camera's RoPE in every
head layer.

The head receives nothing else from history: no refined C-tokens and no
register. Its scene tokens are exactly those of a model without history.

```
 5×4 pooled grid ─► History extractor ─► C_j (current frame) ─► MLP ─► delta_M_j (6×6, orbital cams only)
 Head: plain encoder C-tokens and scene tokens; RoPE of camera j's tokens rotated by delta_M_j,
       one delta_M for every layer and every denoising step
```

## Setup

| | |
|---|---|
| Config | `config/experiment/peract2_orbital_vid_direct_randhist_scratch.yaml` |
| ARM | `exp2_direct_randhist` |
| Training | from scratch, 200k, batch 128, EMA on |
| History | random, K=5, no time embedding |
| delta_M | `rope_6d` on cameras [0, 1]; `layerwise_view_align: false` (it is predicted once) |
| Ladders | SBRS every 10k from 50k, EMA weights only, K=5 |

## What we are checking

- **Is delta_M from the history stack enough on its own?** It gets no register
  and no refined tokens. Compare with exp 1a–1c, which use the same C-tokens
  for adaLN instead of delta_M, and with control A.
- **The old Direct runs** used ordered history. Their ladder recipe also lacked
  `visual_num_history: 5`, so they were evaluated with K=1. This run is laddered
  at K=5 from the start.

## Results (6 Oct)

At 70k IDLT 60/60 56.6, **68.8 at 200k** (calibrated 78.8). Originally reported: IDLT 60/60 **56.6** (A 62.8, C 61.6); IDLT calibrated 68.8; G7 holds 64–78 including 500 cm (70.0). Below the no-delta_M controls under heavy miscalibration. Full table: [camera-adaln](camera-adaln.md#results-6-oct).
