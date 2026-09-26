# Data

## Format

All training data is **zarr** archives. Each zarr group has equal-length arrays:

| Array | Shape | dtype | Description |
|---|---|---|---|
| `rgb` | `(N, ncam, 3, H, W)` | uint8 | Camera images |
| `depth` | `(N, ncam, H, W)` | float16 | Depth (mm) |
| `action` | `(N, T, nhand*8)` | float32 | xyz + quat_xyzw + gripper |
| `extrinsics` | `(N, ncam, 4, 4)` | float32 | Camera-to-world transforms |
| `intrinsics` | `(N, ncam, 3, 3)` | float32 | Camera intrinsics |
| `task_id` | `(N,)` | uint8 | Index into task list |
| `instr_id` | `(N,)` | int | Index into instruction list |

Dataset class names (the `dataset` config key) map to classes in `datasets/__init__.py`.

## Dataset Classes

| Config name | Class |
|---|---|
| `Peract2_3dfront_3dwrist` | `Peract2Dataset` |
| `Peract2_3dfront` | `Peract2SingleCamDataset` |
| `Peract` | `PeractDataset` |
| `PeractCollected` | `PeractCollectedDataset` |
| `OrbitalWrist` | `OrbitalWristDataset` |

## Preprocessing (on-GPU, per batch)

Handled by `utils/data_preprocessors/rlbench.py`:
1. Normalize actions/proprio to [-1, 1] workspace; convert quaternion → 6D rotation
2. Unproject depth → world-frame point cloud (via `utils/depth2cloud/rlbench.py`)
3. Optional: rotate point cloud to front-camera frame (`use_front_camera_frame`)
4. Optional: add miscalibration noise to extrinsics
5. Image augmentation: random affine + random resized crop (via `kornia`)

## Data Generation

Raw PerAct2 data comes from RLBench (customized fork). Raw format is per-episode PKL files.
Convert with `data/processing/convert_to_zarr/peract2_to_zarr.py`.

For Orbital, camera groups G1–G6 are defined in `instructions/orbital_cameras_grouped.json`.
Generation uses `data/generation/orbital/collection.py`.

Language instructions are JSON files in `instructions/peract/` and `instructions/peract2/`.
