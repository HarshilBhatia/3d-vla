# Architecture

## Model Pipeline

```
RGB-D images (3 cams) + language + proprioception
    → BaseEncoder
        ├── Visual backbone (CLIP / SigLIP2 / DINOv2) + optional FPN
        │     → rgb3d_feats (B, N, F)  + pcd (B, N, 3)
        ├── Text encoder → instr_feats (B, L, F)
        ├── FPS subsampling of scene → fps_scene_feats (B, M+ncam, F)
        │     last ncam tokens = per-image avg features (one per camera)
        └── Proprioception cross-attn (3D RoPE) → proprio_feats
    → Iterative denoising (default 5 steps)
        TransformerHead per step:
            1. Traj cross-attn → language
            2. Traj cross-attn → scene (3D RoPE)
            3. Shared self-attn over [traj | fps_scene | 4 register tokens | 1 camera token]
            4. Position / Rotation / Openness heads
Output: trajectory (B, T, nhand, 3+4+1)  xyz + quat_xyzw + gripper
```

## Key Modules

| File | Role |
|---|---|
| `main.py` | Entry point: Hydra → DDP → dataset/model/trainer |
| `modeling/policy/denoise_actor_3d.py` | Main 3D policy; `TransformerHead` extends base |
| `modeling/policy/base_denoise_actor.py` | `compute_loss`, `conditional_sample`, `forward` |
| `modeling/policy/head_strategies.py` | `ExtrinsicsPredictor` strategy classes |
| `modeling/encoder/multimodal/encoder3d.py` | 3D encoder: backbone + FPS + proprioception RoPE cross-attn |
| `modeling/noise_scheduler/rectified_flow.py` | Default noise scheduler |
| `modeling/utils/position_encodings.py` | `RotaryPositionEncoding3D` with optional `delta_M` |
| `modeling/utils/layers.py` | `AttentionModule` / `ComRoPEAttentionModule` with AdaLN |
| `datasets/rlbench.py` | All dataset classes |
| `datasets/base.py` | Zarr loading, LRU caching, chunked indexing |
| `utils/trainers/base.py` | Full training loop, eval, checkpointing, W&B |
| `utils/data_preprocessors/rlbench.py` | On-GPU preprocessing: depth → cloud, augmentation |
| `utils/depth2cloud/rlbench.py` | Batched depth unprojection to world coords |
| `online_evaluation_rlbench/evaluate_policy.py` | Online eval entry point |
| `paths.py` | Per-user path config (toggled via `USER_NAME` env var) |

## Non-Obvious Design Details

**3D RoPE + $\Delta M$**: `RotaryPositionEncoding3D` encodes xyz coordinates as RoPE. A learned 6×6 (or full D×D) orthogonal matrix performs per-camera view alignment by correcting representation-space sin/cos bases. It is not an estimate of a physical extrinsic.

**Global context token**: A learnable `nn.Parameter` appended as the last token in the shared self-attention sequence (`features[:, -1, :]`). It is one global register/context token, not one token per camera. Camera-summary features directly predict $\Delta M_j$; the global token affects them only indirectly through shared attention.

**Rotation representation**: Training converts quaternion → 6D rotation for both target and prediction; loss is computed in 6D space. Output is converted back to quaternion.

**Loss weights**: `30 * L1(pos) + 10 * L1(rot) + BCE(openness)`. Target for rectified flow = `noise − gt_trajectory` (velocity field).

**Val metric for `best.pth`**: `traj_pos_acc_001` — fraction of trajectory points within 1 mm of ground truth.

**Workspace normalizer**: On first run (no checkpoint), the trainer scans all training data to compute action min/max for normalization. Saved in the checkpoint and reused on resume.

**Checkpoint format**: `{"weight": ..., "ema_weight": ..., "optimizer": ..., "iter": int, "best_loss": float}`. Loading is non-strict (`strict=False`) to support architectural changes.

**AMP dtype**: Uses `bfloat16` by default; falls back to `float32` on Quadro RTX 6000 (detected at runtime).

**Checkpoint format**: `{"weight": ..., "ema_weight": ..., "optimizer": ..., "iter": int, "best_loss": float}`. Loading is non-strict (`strict=False`) to support architectural changes.
