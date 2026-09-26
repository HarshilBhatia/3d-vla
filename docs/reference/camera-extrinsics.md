# PerAct2 camera extrinsics

## Source

Extrinsics come from **RLBench (CoppeliaSim/PyRep)**. For each observation, the scene stores per-camera matrices via `VisionSensor.get_matrix()` in `rlbench/backend/scene.py` and exposes them as `{camera_name}_camera_extrinsics` in the demo `misc` dict.

## Format

- **Shape**: `(4, 4)` per camera per timestep.
- **Meaning**: **Camera-to-world** transform (object pose of the camera in the simulation world frame).
  - `R = E[:3, :3]`: rotation from camera frame to world frame.
  - `t = E[:3, 3]`: camera origin in world coordinates.
- **Convention**: Same as PyRep `Object.get_matrix()` — standard 4×4 homogeneous transform with bottom row `[0, 0, 0, 1]`.

## Cameras

PerAct2 uses three cameras (order in zarr):

1. `front`
2. `wrist_left`
3. `wrist_right`

They are fixed in the scene (or attached to the wrist); extrinsics can vary per timestep for wrist cameras.

## In zarr

- **Dataset**: `extrinsics`
- **Shape**: `(n_rollouts, T, NCAM, 4, 4)` with `NCAM=3`, stored as float16.
- **Indexing**: `extrinsics[i]` is shape `(T, 3, 4, 4)` for rollout `i` (T keyframes, 3 cams).

## Transforming extrinsics (rotation and/or translation)

To simulate a globally transformed camera rig:

- **Rotation**: `E_new = R_global @ E` (e.g. +10° around world Z).
- **Translation**: same 4×4 transform with `t` in meters: `T_global = [R|t; 0 0 0 1]`, `E_new = T_global @ E`.

The script `peract2_to_zarr.py` supports:
- `--rotate-extrinsics-deg` (e.g. `10` for +10° around Z)
- `--translate-extrinsics "dx,dy,dz"` (meters, e.g. `"0.01,0,0"`)

When **rotation** is used, RGB and depth images are rotated by the same angle (in the image plane) so the stored view matches the rotated extrinsics. Translation does not change images.

## PerAct2 demonstration count

- **Tasks**: 13 (see `DEFAULT_TASKS` in `peract2_to_zarr.py`).
- **Count episodes** from raw data:  
  `python -m data_processing.count_peract2_demos --root peract2_raw`  
  This prints per-task/split episode counts and total. Exact numbers depend on your raw dataset.
