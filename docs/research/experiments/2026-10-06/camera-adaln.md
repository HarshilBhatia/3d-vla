# C-token adaLN conditioning (Exp 1a, 1b, 1c)

**Status:** done. Robustness evals complete (6 Oct)
**Week introduced:** [30 Sept – 6 Oct](../../weekly_logs/2026-10-06.md)

## Main idea

Each camera has a **C-token**: a single vector summarising what that camera
currently sees. Until now, the C-tokens were used to predict delta_M, a 6×6
rotation of that camera's RoPE. That is a *geometric* correction applied to
attention positions.

These runs keep the C-tokens but never predict a delta_M from them. Instead, the
C-tokens set the **adaLN** scale and shift of the policy head's layers. The
camera information now reaches the actions as **feature modulation**, not as a
change to the 3D positions.

The three variants differ along two questions:

- does each camera condition its own tokens, or is there one global signal?
- is the signal updated as the C-tokens change through the head's layers?

## What a C-token is here

- **Encoder:** each (frame, camera) image is mean-pooled over its patches into
  one appearance vector, giving a 5 frames × 4 cameras grid. No xyz is mixed in.
- **History extractor** (pooled, random history, no time embedding): 4 rounds of
  same-time attention across the 4 cameras, then attention across each camera's
  5 frames. The current row comes out as the 4 history-refined C-tokens.
- **Policy head:** the C-tokens join the self-attention sequence as tokens at
  each camera's mean point-cloud xyz, so the head's layers keep updating them.
- The extractor sees only pooled tokens. The head's 3D reasoning uses only the
  **current frame's** patches.

## Shared backbone

```
 4 cams × 5 random frames ─► Encoder ─► current-frame patches ─► head cross-attn keys + FPS scene tokens
                                    └─► 5×4 pooled grid ─► History extractor ─► C1..C4
 Head:  traj queries ─► 2 × cross-attn (3D RoPE, no delta_M)             [adaLN]
        SA = [traj | FPS tokens | C1..C4 | 4 registers | camera token]
        4 × self-attn + FFN (3D RoPE)                                    [adaLN]
        position head, rotation head (2 SA layers each), gripper MLP     [adaLN]

 every adaLN:  scale, shift = Linear_layer(SiLU(signal));  x ← x·(1+scale) + shift
 before:       signal = time_embs                  (denoising timestep + proprio)
 now:          signal = time_embs + MLP_c(C)       (MLP_c = new Linear–ReLU–Linear shared by all layers)
```

## The arms

| Exp | `camera_adaln` | Signal | Changes per SA layer? |
|---|---|---|---|
| **1a** | `per_camera` | camera j's tokens (its FPS tokens and C_j) get `time_embs + MLP_c(C_j)`; traj, register and camera tokens get `time_embs` | yes: layer i uses the C-tokens as layer i−1 left them; output heads use the final ones |
| **1b** | `pooled` | `time_embs + MLP_c(mean_j C_j)` for every token | yes, same schedule as 1a |
| **1c** | `static_pooled` | `time_embs + MLP_c(mean_j C_j)` with the C-tokens **entering** the head, for every token | no: one signal for cross-attn, all SA layers and both output heads |

In 1a, cross-attention gets `time_embs` only, because its adaLN modulates only
the trajectory queries, which belong to no camera. In 1b, cross-attention uses
the mean of the C-tokens entering the head.

## Everything else (identical across arms)

- From scratch, **200k steps**, batch 128, lr 3e-4. The same recipe as control A
  ([from-scratch-noalign-randhist](../2026-09-29/from-scratch-noalign-randhist.md)).
- Random unordered history (K=5), no time embedding, proprio K=3.
- `view_align_mode: none`: delta_M is never predicted.
- **History register off** (the new default; see the weekly log).
- **EMA on** (the new default).
- Miscalibration: medium fixed group bias on the orbital cameras, plus 3°/1 cm
  sampled perturbation.
- adaLN uses all 4 C-tokens, wrists included. delta_M only ever corrected the 2
  orbital cameras.
- Ladders: SBRS every 10k from 50k, **EMA weights only**
  (`video_deltam_checkpoint_ladder_ema_v01`, K=5). Raw ladders are skipped to
  save compute.

| Exp | Config | ARM | Run dir |
|---|---|---|---|
| 1a | `peract2_orbital_vid_pooled_randhist_adaln_percam_scratch` | `exp1a_adaln_percam` | same as config |
| 1b | `peract2_orbital_vid_pooled_randhist_adaln_pooled_scratch` | `exp1b_adaln_pooled` | same as config |
| 1c | `peract2_orbital_vid_pooled_randhist_adaln_static_scratch` | `exp1c_adaln_static` | same as config |

## What we are checking

1. **Can adaLN replace delta_M and the register?** These runs have neither.
   Compare them with the no-adaLN controls. A has the register, from scratch.
   C has no register and was warm-started from A at 50k.
   - If 1a/1b/1c hold IDLT 60/60 robustness while C drops, the C-tokens can
     carry the robustness as feature modulation.
2. **Per-camera vs global (1a vs 1b).** Does each camera need its own
   conditioning, as delta_M was per camera, or is one global summary enough?
3. **Updated vs fixed (1b vs 1c).** Does the signal need to change as the head
   refines the C-tokens, as layer-wise delta_M did, or is one global signal
   computed up front enough?
4. **In distribution first:** SBRS ladders on EMA weights. Then the
   robustness suites (IDLT task-held-out up to 60/60, G7 extremes) on the best
   EMA checkpoint of each arm.

## Caveats

- **Controls.** The register makes no difference (A ≈ C on all 430 robustness
  cells), so A and C both serve as the no-adaLN control. Neither was trained
  with EMA, and these runs are laddered on EMA weights only, so part of any
  gain may come from EMA. The A/C EMA fine-tunes (71.5 / 72.5 SBRS) are the
  nearest EMA reference. **Bar:** IDLT 60/60
  about 62 from scratch; the warm-started models reach about 72.
- Random history barely uses its history (see
  [random-history-no-time](../2026-09-29/random-history-no-time.md)), so the
  C-tokens are probably close to "the current view, mixed across cameras".
- Each adaLN Linear is zero-initialised, so every arm starts out identical to
  having no C-token conditioning.

## Results (6 Oct)

| Condition | A | C | 1a | **1b** | 1c | 2 | 3 |
|---|---|---|---|---|---|---|---|
| IDLT calibrated | 75.3 | 74.1 | 65.5 | **80.8** | 74.6 | 68.8 | 71.8 |
| IDLT 5/5 | 75.2 | 76.2 | 65.3 | **81.0** | 72.5 | 71.8 | 69.8 |
| IDLT 10/10 | 73.8 | 75.2 | 64.8 | **81.3** | 72.8 | 69.7 | 71.8 |
| IDLT 20/20 | 67.1 | 72.0 | 63.7 | **78.2** | 64.7* | 64.3 | 67.8 |
| IDLT 30/30 | 68.1 | 69.1 | 61.0 | **74.0** | 65.3 | 62.0 | 67.4 |
| IDLT 40/40 | 64.8 | 66.2 | 59.8 | **71.5** | 61.2 | 60.9 | 65.5 |
| IDLT 50/50 | 63.2 | 64.9 | 58.8 | **70.7** | 55.1 | 58.8 | 62.2 |
| **IDLT 60/60** | 62.8 | 61.6 | 55.5 | **70.4** | 48.2 | 56.6 | 55.9 |
| G7 opposite 100 / 200 / 500 cm | 71.7 / 73.3 / 66.7 | 75.0 / 80.8 / 66.7 | 64.2 / 68.3 / 66.7 | 69.2 / 70.8 / 70.8 | 54.2 / 55.0 / 53.3 | 72.5 / 75.8 / 70.0 | 70.8 / 70.0 / 46.7 |
| G7 rotation 20 / 90 / 180 / 270° | 78.3 / 75.0 / 71.7 / 74.2 | 80.0 / 72.5 / 65.8 / 70.8 | 65.8 / 65.8 / 63.3 / 71.7 | 72.5 / 67.5 / 75.0 / 75.0 | 75.8 / 67.5 / 60.8 / 59.2 | 78.3 / 74.2 / 65.8 / 64.2 | 69.2 / 61.7 / 60.8 / 65.8 |

IDLT cells are 13 tasks × 100 rollouts (about ±1.3 points); G7 cells are 6 tasks × 20 (about ±4). A and C are raw weights at their best ladder points (160k, 100k); the new models are EMA weights at their best EMA ladder points (1a 80k, 1b 200k, 1c 120k, exp 2 70k, exp 3 110k). \* = one of 13 tasks missing (timed-out cell, resubmitted as 3990347).

## Reading (corrected 7 Oct: the first reading was a checkpoint-step artifact)

The table above compares checkpoints picked by best SBRS, which spanned 70k to 200k. The step sweep shows robustness keeps climbing with training while SBRS stays flat, so that table mostly ranks training length. At equal training (200k, EMA):

| Model (200k) | IDLT calibrated | IDLT 60/60 |
|---|---|---|
| exp 3, layer-wise delta_M | 77.5 | **71.6** |
| 1b, pooled adaLN | **80.8** | 70.4 |
| exp 2, Direct delta_M | 78.8 | 68.8 |
| 1a, per-camera adaLN | 74.3 | 68.5 |
| 1c, static adaLN | 77.3 | 56.6 |

exp 3 by step (calibrated / 60/60): 50k 64.3/48.4, 80k 65.0/53.8, 110k 71.8/55.9, 140k 73.2/62.5, 170k 72.7/65.7, 200k 77.5/71.6. SBRS at the same steps: 69, 68, 74, 63, 65, 72.

- **3, 1b, 2 and 1a are within about 3 points of each other at 60/60.** Only 1c (static signal) is clearly worse.
- **Fair control pending.** A (62.8) and C (61.6) were also evaluated at SBRS-picked steps (160k, 100k); A at 200k is being evaluated.
- **The SBRS ladder is retired for checkpoint selection** (see the weekly log).

**delta_M diagnostic** (`scripts/eval/diagnose_delta_m.py`, `results/diagnose_delta_m/wk1006.json`): in every pooled model — VideoPooled, the warm-started random-history models and exp 2/3 — delta_M does not respond to miscalibration. It is predicted from appearance-only C-tokens, so the delta_M before cross-attention is identical under clean and 60/60 cameras, and later layers barely move it. It acts as a learned fixed re-mixing of the RoPE basis: about 0.35–0.38 (Frobenius distance from identity) in the warm models, 0.77 → 0.89 in exp 3.
