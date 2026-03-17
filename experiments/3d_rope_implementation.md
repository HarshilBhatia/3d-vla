# 3D RoPE Implementation for Spatially-Grounded Vision Tokens

## Overview

We extend GR00T N1D6's DiT action head with **3D Rotary Position Embeddings (RoPE)** applied to the cross-attention over vision tokens. Each image patch token is assigned a 3D world-frame position (from depth unprojection), and this position is encoded into the Keys of cross-attention — allowing the action head to reason about *where in 3D space* each visual feature is located.

---

## Motivation

The standard GR00T DiT cross-attention treats all vision tokens as spatially anonymous: a patch at the top-left of the image and a patch in the center carry no positional signal in cross-attention. With 3D RoPE:

- Each image token carries its 3D position in the robot's base frame
- The dot product `Q · K` implicitly measures "how relevant is this 3D location to this action token"
- The model can learn geometry-aware attention: preferring patches of objects it's about to interact with at the right depth

---

## Architecture

### Where RoPE is Applied

The DiT (`AlternateVLDiT`, 32 blocks) alternates cross-attention targets:
- **Even blocks**: action tokens attend to **text** tokens → no 3D RoPE
- **Odd blocks**: action tokens attend to **image** tokens → 3D RoPE applied to Keys

RoPE is applied **only to Keys** (not Queries or Values). Queries come from action tokens which have no natural 3D position. The Key rotation encodes "this visual feature is at position (x, y, z)" so that the attention score `Q · K_rot` is geometry-aware.

### RoPE Formula

`head_dim = 48` is split into 3 equal groups of 16, one per spatial axis:

```
freqs = 1 / (10000 ^ (2 * [0..7] / 16))    # 8 frequency bands per axis

For position (x, y, z):
  cos_x = cos(x * freqs),  sin_x = sin(x * freqs)   → [8]
  cos_y = cos(y * freqs),  sin_y = sin(y * freqs)
  cos_z = cos(z * freqs),  sin_z = sin(z * freqs)

K_rot:
  K_x_rot = K_x * cos_x - rotate_half(K_x) * sin_x   # dims  0–15
  K_y_rot = K_y * cos_y - rotate_half(K_y) * sin_y   # dims 16–31
  K_z_rot = K_z * cos_z - rotate_half(K_z) * sin_z   # dims 32–47
  K_rot = cat([K_x_rot, K_y_rot, K_z_rot])
```

Position `(0, 0, 0)` → identity rotation (K unchanged). This is used for wrist camera tokens and text tokens — they participate in attention without any spatial bias.

---

## Depth Pipeline: Depth → 3D Position → RoPE

### Step 1: Image Patching Geometry

Eagle resizes all 180×320 DROID images to **168×308** (verified empirically). With `pixels_per_token = 784 = 28×28`, each image is tiled into a **6×11 grid** of 28×28-pixel patches, giving 66 tokens per camera (64 are marked as `image_mask=True` after excluding 2 framing tokens).

Camera token order in backbone sequence (from embodiment config):
1. **Exterior camera** (`exterior_image_1_left`) — first block of image tokens
2. **Wrist camera** (`wrist_image_left`) — second block

### Step 2: Depth Data Format

Depth is extracted from ZED SVO files via `data_processing/extract_svo_depth.py` and stored at:

```
/work/nvme/bgkz/droid_rail_depths/{canonical_id}/{serial}/
    depth.blosc       # (T, 180, 320) float32, blosc-zstd compressed, meters, NaN=invalid
    intrinsics.npy    # [fx, fy, cx, cy] already scaled to 180×320
    shape.npy         # [T, 180, 320]
```

The serial number (ZED camera ID) is mapped to camera role (ext1/wrist) via:
```
/work/nvme/bgkz/droid_rail_depths/serial_map.json
  {canonical_id: {"ext1": "20521388", "wrist": "13062452"}, ...}
```

### Step 3: Unprojection (Camera Frame)

For each exterior camera token at grid position `(row, col)`:

```python
# Pixel region in 168×308 resized image
y0, y1 = row * 28, (row + 1) * 28
x0, x1 = col * 28, (col + 1) * 28

# Depth values (resize from 180×320 → 168×308 via INTER_NEAREST)
D = depth_resized[y0:y1, x0:x1]        # (28, 28)
valid = isfinite(D) & (D > 0.05) & (D < 5.0)

# Intrinsics scaled to 168×308
fx = fx_180 * 308/320;  fy = fy_180 * 168/180
cx = cx_180 * 308/320;  cy = cy_180 * 168/180

# Unproject valid pixels to 3D camera frame
X_cam = (u - cx) * D / fx
Y_cam = (v - cy) * D / fy
Z_cam = D

# Average over valid pixels → one 3D point per patch
```

### Step 4: Camera → World Frame (Extrinsics)

Each RAIL episode's `metadata_*.json` contains `ext1_cam_extrinsics`: a 6-DOF vector `[tx, ty, tz, rx, ry, rz]` where `[tx, ty, tz]` is the camera translation in robot base frame and `[rx, ry, rz]` is an axis-angle rotation vector.

```python
from scipy.spatial.transform import Rotation
T_cam2base = np.eye(4)
T_cam2base[:3, :3] = Rotation.from_rotvec([rx, ry, rz]).as_matrix()
T_cam2base[:3, 3]  = [tx, ty, tz]

pts_world = pts_cam @ T_cam2base.T    # → robot base frame
```

### Step 5: Token Position Assignment

```
token_positions_3d  [seq_len, 3]:
  - Exterior cam tokens (first ~64 image tokens):  real (x, y, z) in base frame
                                                    static extrinsics from metadata_*.json
  - Wrist cam tokens   (next  ~64 image tokens):   real (x, y, z) in base frame
                                                    per-timestep extrinsics from trajectory.h5
  - Text / padding tokens:                          (0, 0, 0)
```

Wrist extrinsics live in `trajectory.h5` at `observation/camera_extrinsics/{wrist_serial}_left`, shape `(T, 6)`. The full array is cached per episode to avoid repeated h5 reads; indexed by `frame_idx` at runtime.

Position `(0, 0, 0)` → identity RoPE → K unchanged → no spatial bias (used for text tokens and episodes where depth is unavailable).

---

## Data Infrastructure

### Preprocessing (one-time, CPU)

**`data_processing/build_episode_frame_index.py`**

Reconstructs the same `ShardedSingleStepDataset` used for backbone caching (seed=42, labs=RAIL, shard_size=10000, episode_sampling_rate=0.1) and walks `sharded_episodes` to build:

- `episode_frame_index.pkl` — list of `{canonical_id, frame_idx}` indexed by `cache_global_idx` (360,736 entries for RAIL)
- `serial_map.json` — `{canonical_id: {ext1: serial, wrist: serial}}` from raw metadata JSONs

Both saved to `/work/nvme/bgkz/droid_rail_depths/`.

### Runtime (on-the-fly in collator)

`Gr00tN1d6DataCollator` loads depth per sample when `depth_dir` is set:

1. Look up `canonical_id, frame_idx` from `episode_frame_index.pkl[global_idx]`
2. Load depth episode from `depth.blosc` (LRU cache keyed by `(canonical_id, serial)`)
3. Clamp `frame_idx` to depth array bounds (SVO may be shorter than trajectory)
4. Load intrinsics from `intrinsics.npy` (cached per episode)
5. Load extrinsics from raw `metadata_*.json` (cached per episode via `_extrinsics_cache`)
6. Unproject + transform to base frame
7. Return `token_positions_3d [seq_len, 3]` in batch

---

## Code Structure

| File | Role |
|------|------|
| `gr00t/model/gr00t_n1d6/rope_3d.py` | `compute_rope_cos_sin`, `apply_3d_rope_to_keys`, `RoPE3DCrossAttnProcessor` |
| `gr00t/model/modules/dit.py` | `BasicTransformerBlock` computes RoPE on-the-fly; `AlternateVLDiT` passes positions only to image-attending blocks |
| `gr00t/model/gr00t_n1d6/gr00t_n1d6.py` | `token_positions_3d` flows: batch → action head → DiT |
| `gr00t/model/gr00t_n1d6/processing_gr00t_n1d6.py` | Collator loads depth on-the-fly, builds `token_positions_3d` |
| `gr00t/model/gr00t_n1d6/setup.py` | Loads `episode_frame_index.pkl` + `serial_map.json` at training startup |
| `gr00t/configs/finetune_config.py` | `--depth-dir`, `--episode-index-path` CLI args |
| `data_processing/build_episode_frame_index.py` | Builds the two index files (run once) |

---

## Training Usage

```bash
python -m gr00t.experiment.launch_finetune \
    --base-model-path /path/to/checkpoint \
    --dataset-path /work/nvme/bgkz/droid_raw_large_superset \
    --embodiment-tag OXE_DROID \
    --cached-backbone-dir /work/nvme/bgkz/droid_rail_cache \
    --depth-dir /work/nvme/bgkz/droid_rail_depths \
    --labs RAIL \
    ...
```

`episode_frame_index.pkl` and `serial_map.json` are loaded automatically from `--depth-dir`.

---

## Open Questions / Future Work

1. **Wrist camera depth**: Implemented. Per-timestep extrinsics from `trajectory.h5` (`observation/camera_extrinsics/{serial}_left`). Falls back to `(0,0,0)` if depth or trajectory file is missing for an episode.

2. **Action token positions**: The spec notes the question of whether to assign positions to action tokens for self-attention RoPE. Currently action tokens have no positional encoding in self-attention (they use a learned 1D position embedding). Noisy 3D positions (e.g., from the robot end-effector trajectory) could be explored.

3. **RoPE scale**: The coordinate scale (meters, robot base frame) interacts with the RoPE frequency bands. Coordinates typically range ~0–3m. The base frequency `1/10000` is appropriate for this scale, but tuning may help.

4. **Wrist camera frame rate mismatch**: Depth arrays can be shorter than the trajectory length (SVO recording may stop early). Currently handled by clamping `frame_idx` to `T-1`.
