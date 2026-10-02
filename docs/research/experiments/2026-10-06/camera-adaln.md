# C-token adaLN conditioning (Exp 1a, 1b, 1c)

**Status:** code, configs and ARMs ready (1 Oct); not yet submitted
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

## Results

None yet.
