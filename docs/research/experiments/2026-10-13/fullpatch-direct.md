# Full-patch 3D-RoPE history with Direct δM (exp 6)

**Status:** training since 12 Oct
**Week introduced:** [7–13 Oct](../../weekly_logs/2026-10-13.md)

## Why

Every pooled history extractor we trained collapsed to a constant ([TODOs](../2026-10-06/TODOs.md)), and the pooled C-tokens carry no geometry: they are mean-pooled appearance features, so the extractor cannot see where a miscalibrated camera's points landed. δM therefore never responded to miscalibration. The full-patch rope3d extractor does see geometry (every patch is rotated by its unprojected xyz inside attention), but it also collapsed and was trained with consecutive history only. This run gives the geometry-aware extractor the two fixes from exp 4 and this week's history recipe.

## Architecture

- **Input:** every patch of every camera at K = 5 **randomly sampled** history frames, with each patch's world xyz from its (miscalibrated) extrinsics.
- **Extractor:** 4 × (same-time cross-camera attention, then per-camera temporal attention), q/k rotated by 3D RoPE of patch xyz. No time embedding and no causal mask (order-free history); only the current frame is marked by a learned embedding. A per-image token sits at its patches' centroid and is read out per camera.
- **Pre-LN** blocks plus a final LayerNorm (exp 4a's fix), so the residual stream is never renormalised and the stack cannot collapse to a constant output.
- **Input skip** (exp 4b's fix): a Linear of the raw current-frame patch mean is added to each camera's output token. It carries appearance only, no xyz.
- **Direct δM:** the current frame's image token per camera → MLP → 6×6 orthogonal δM (zero-init, so it starts at identity), applied once to cameras 0/1's scene-token RoPE in the head. No head-side δM predictor, no register.

Code: `video_deltam_prenorm`, `video_deltam_input_skip` and `video_deltam_time_embedding: false` now work on the `patch_rope3d` full-image path (`modeling/policy/video_deltam.py`, `_forward_patch_rope3d`); tests in `tests/test_video_deltam.py`.

| Config | ARM | Twin |
|---|---|---|
| `peract2_orbital_vid_fullpatch_rope3d_randhist_direct_prenorm_inputskip_scratch` | `exp6_fullpatch_direct` | `scripts/multinode/config/exp6_fullpatch_direct.yaml` |

From scratch, EMA, 200k, batch 128. Robustness ladder (IDLT calibrated and 60/60, 13 × 20, EMA) every 20k from 100k; SBRS sanity check every 50k.

## Jobs

| Training chain | 4×1 twin | Full robustness suite at 200k (on training dependency) |
|---|---|---|
| 4027033 → 34 → 35 (shubhamlong, grogu-2-25) | 4027036, cancelled while pending (chain started first) | 4027037–4027043 (215 cells), from `scripts/eval/create_exp6_robustness_plans.py` |

## What we're checking

- **Does δM respond to miscalibration** now that the extractor sees per-patch geometry and stays alive? Measure δM spread across samples and its correlation with the injected error (TODO 1/3 style, at 100k and 200k).
- **IDLT 60/60 vs exp 2** (68.8; Direct δM on pooled random history) and exp 3 (71.6), without losing calibrated performance (exp 2 78.8).
- **Is the skip used instead of the stack?** Probe the stack output and the skip separately for task and camera group (TODO 15 style).
