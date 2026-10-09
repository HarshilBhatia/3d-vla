# Proprio encoder without miscalibrated scene context (exp 5a, 5b)

**Status:** submitted 9 Oct
**Week introduced:** [7–13 Oct](../../weekly_logs/2026-10-13.md)

## Why

Activation patching ([TODO 7](../2026-10-06/TODOs.md)) found that most of the 60/60 damage in 1b and control A reaches the action through the **proprio features**: 79% in 1b and 88% in A, and 95–100% of the rotation damage in most models. The encoder builds them (`encode_proprio`, `encoder3d.py`) with 6 learned queries placed by RoPE at the true gripper xyz, attending over **all four cameras'** dense scene tokens at their unprojected xyz. Under miscalibration the two external cameras' tokens sit at wrong positions, the lookup reshuffles, and the corrupted summary conditions every head layer through adaLN.

## Arms

Both use exp 1b's recipe: pooled layer-wise C-token adaLN, from scratch, random pooled history, no time embedding, no register, EMA, 200k. The only change is the new `proprio_context` option (commit `6b22fa4`).

| Exp | `proprio_context` | What the proprio encoder sees | Config | ARM |
|---|---|---|---|---|
| **5a** | `wrist` | only the wrist cameras' tokens (cameras 2, 3, never miscalibrated) | `peract2_orbital_vid_pooled_randhist_adaln_pooled_proprio_wrist_scratch` | `exp5a_proprio_wrist` |
| **5b** | `none` | no scene at all; an MLP on the gripper state (xyz, quaternion, open) | `peract2_orbital_vid_pooled_randhist_adaln_pooled_proprio_none_scratch` | `exp5b_proprio_none` |

## Jobs

| | Training chain | 4×1 twin | Full robustness suite at 200k (on training dependency) |
|---|---|---|---|
| 5a | 4000087 → 88 → 89 | 4000094 | 4000096–4000102 (215 cells) |
| 5b | 4000090 → 91 → 92 | 4000095 | 4000103–4000109 (215 cells) |

Robustness ladder (IDLT calibrated and 60/60, 13 × 20, EMA) every 20k from 100k; SBRS sanity check every 50k.

## What we're checking

- **5a vs 1b** (70.4 at IDLT 60/60, 80.8 calibrated): does cutting the miscalibrated cameras out of the proprio lookup improve robustness, without costing calibrated performance?
- **5b vs 5a:** is the gripper-vs-scene context needed at all? If 5b matches 5a, the simpler encoder wins. If 5b drops when calibrated, the scene lookup carries real information and the wrist-only version is the right fix.

## Results

None yet.
