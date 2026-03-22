# DROID trajectory.h5 — Field Reference

Per-episode file at:
```
/work/nvme/bgkz/droid_multilab_raw/{canonical_id}/trajectory.h5
```
Shape `(T, ...)` where T = number of timesteps in the rollout (e.g. 113).

---

## observation/robot_state/

| Key | Shape | Notes |
|-----|-------|-------|
| `cartesian_position` | (T, 6) | EEF pose in robot base frame: `[x, y, z, rx, ry, rz]`. First 3 = position (meters), last 3 = axis-angle rotation (radians). **Authoritative EEF position — same frame as depth-unprojected image token positions.** |
| `joint_positions` | (T, 7) | Full 7-DOF Franka joint angles (radians). Note: lerobot parquet only stores 6. |
| `joint_velocities` | (T, 7) | |
| `joint_torques_computed` | (T, 7) | |
| `motor_torques_measured` | (T, 7) | |
| `gripper_position` | (T,) | |
| `prev_command_successful` | (T,) bool | |
| `prev_controller_latency_ms` | (T,) | |
| `prev_joint_torques_computed` | (T, 7) | |
| `prev_joint_torques_computed_safened` | (T, 7) | |

---

## observation/camera_extrinsics/

One entry per camera serial × {left, right}. Format: `[tx, ty, tz, rx, ry, rz]`.

**Convention: cam→base (T_base_cam)** — a point in camera frame multiplied by this matrix lands in robot base frame.

**Rotation: Euler XYZ** (intrinsic rotations about X then Y then Z, in radians) — NOT axis-angle/rotvec.
Use `Rotation.from_euler("xyz", dof[3:]).as_matrix()`, not `from_rotvec`.

Note: `cartesian_position` (robot state) uses axis-angle for its rotation component — different convention.

| Camera type | Varies over time? | Source used in pipeline |
|-------------|-------------------|------------------------|
| Wrist (type=0) | **Yes** — moves with arm | `trajectory.h5` per-timestep |
| Exterior (type=1) | **No** — static per rollout, but differs across rollouts | `metadata_*.json` (`ext1_cam_extrinsics`) |

Some serials also have `_gripper_offset` variants — camera-to-gripper-tip transform.

### observation/camera_type/
Maps serial → `0` (wrist) or `1` (exterior). Only 3 cameras listed here (the active cameras for this episode); trajectory.h5 may contain extrinsics for additional serials not used.

---

## action/

| Key | Shape | Notes |
|-----|-------|-------|
| `cartesian_position` | (T, 6) | Next-step EEF target pose `[x,y,z,rx,ry,rz]` |
| `cartesian_velocity` | (T, 6) | |
| `joint_position` | (T, 7) | |
| `joint_velocity` | (T, 7) | |
| `gripper_position` | (T,) | |
| `gripper_velocity` | (T,) | |
| `target_cartesian_position` | (T, 6) | Commanded target (slightly different from `cartesian_position`) |
| `target_gripper_position` | (T,) | |
| `robot_state/*` | same as observation/robot_state | Robot state at action time |

---

## observation/controller_info/

`controller_on`, `failure`, `movement_enabled`, `success` — all `(T,) bool`.

---

## observation/timestamp/

Per-camera capture/receive/read timestamps (int64, milliseconds epoch) and control loop timing. Useful for synchronisation but not used in current pipeline.

---

## Key facts for EEF position in RoPE

- **Best source**: `observation/robot_state/cartesian_position[:, :3]` — EEF xyz in robot base frame, computed by Franka controller (essentially high-accuracy FK).
- **Same frame** as depth-unprojected image token positions `p_k`.
- **Relative vector** `p_k - p_eef` = spatial offset of image token from arm → enables true relative RoPE: `score = R(p_eef)q · R(p_k)k = q · R(p_k - p_eef)k`.
- **Not in lerobot parquet** — parquet only has joint positions (6-DOF, not 7). To use EEF position at training time, options are:
  1. Pre-cache EEF xyz alongside depth shards (cleanest)
  2. Compute FK from 6 joint angles at training time (self-contained, ~0.1ms/batch)
  3. Read from trajectory.h5 at training time (slow — raw files, not memory-mapped)

---

## pipeline/cache_depth_features.py usage

- Wrist extrinsics: reads `observation/camera_extrinsics/{wrist_serial}_left` per-timestep from `trajectory.h5`
- Exterior extrinsics: reads `ext1_cam_extrinsics` from `metadata_*.json` (static per rollout)
- EEF position: **not currently used** — candidate for query-side RoPE
