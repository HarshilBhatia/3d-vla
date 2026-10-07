# Layer-wise delta_M on random history, from scratch (Exp 3)

**Status:** done (6 Oct)
**Week introduced:** [30 Sept – 6 Oct](../../weekly_logs/2026-10-06.md)

## Main idea

This is the delta_M counterpart of the [C-token adaLN arms](camera-adaln.md). The
recipe is the same: from scratch, random pooled history, no time embedding, no
register, EMA. The C-tokens predict a 6×6 delta_M per orbital camera, re-predicted
before every self-attention layer, which rotates that camera's RoPE. That is the
VideoPooled mechanism.

The only earlier random-history delta_M run
([random-history-no-time](../2026-09-29/random-history-no-time.md)) was
warm-started from VideoPooled 84k, had the register on, used no EMA, and ran
100k steps. This is the first from-scratch version.

## Setup

| | |
|---|---|
| Config | `config/experiment/peract2_orbital_vid_pooled_randhist_deltam_scratch.yaml` |
| ARM | `exp3_randhist_deltam` |
| Training | from scratch, 200k, batch 128, EMA on, register off |
| delta_M | `rope_6d`, layer-wise, cameras [0, 1] |
| Ladders | SBRS every 10k from 50k, EMA weights, K=5 |
| Jobs | chain 3985398 → 3985399 → 3985400 (the single-node job won the race; twins cancelled) |

## What we are checking

- **Geometry vs features (exp 3 vs 1a).** The same C-tokens are used either to
  rotate positions (delta_M) or to modulate features (adaLN, per camera,
  layer-wise). Which carries more robustness?
- **Does delta_M help on random history from scratch?** Compare with control
  A/C, which have no delta_M. The warm-started random-history model matched
  VideoPooled; this run removes the inherited weights.

## Results (6 Oct)

At 110k IDLT 60/60 55.9, **but 71.6 at 200k** (50k→200k: 48.4, 53.8, 55.9, 62.5, 65.7, 71.6). Originally reported: IDLT 60/60 **55.9** (A 62.8, C 61.6; 1b 70.4); IDLT calibrated 71.8; G7 opposite 500 cm collapses to 46.7. From scratch, layer-wise delta_M does not help on random history. Full table: [camera-adaln](camera-adaln.md#results-6-oct).
