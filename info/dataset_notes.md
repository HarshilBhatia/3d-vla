# Dataset: droid_raw_large_superset

## Location
`/work/nvme/bgkz/droid_raw_large_superset`

## Basic stats
- 23,700 episodes
- 6,723,022 total frames
- 15 FPS, avg episode length: 283.7 frames (~18.9s)
- Min/Max episode length: 1 / 2287 frames
- 12,826 unique language tasks (free-form text, no structured categories)

## Filter
`cam2base_superset_both_cameras` — episodes with camera-to-base extrinsics available AND both cameras present.
Extrinsics source: `/ocean/projects/cis240058p/hbhatia1/data/droid_annotations/cam2base_extrinsic_superset.json`

## Source labs (by episode count)
| Lab | Episodes |
|-----|----------|
| TRI (Toyota Research) | 5,667 |
| AUTOLab (Berkeley) | 4,555 |
| IPRL (Stanford) | 2,285 |
| REAL | 1,673 |
| CLVR | 1,658 |
| ILIAD | 1,511 |
| RAIL (Berkeley) | 1,470 |
| IRIS | 1,301 |
| PennPAL (Penn) | 1,259 |
| RPL | 989 |
| WEIRD | 810 |
| GuptaLab | 518 |

Episode IDs encoded as `{LAB}+{robot_id}+{timestamp}`.

## Modality
- **Cameras**: `exterior_image_1_left`, `wrist_image_left` (180×320, AV1 codec)
  - `exterior_image_2_left` exists in features schema but is NOT in modality.json (not used)
- **State**: 7-dim — joint_position[0:6] + gripper_position[6:7]
- **Action**: 7-dim — joint_position[0:6] + gripper_position[6:7]
- **Annotation**: language_instruction (mapped via task_index)

## Extrinsics — Findings (2026-03-20)

### Processing convention (confirmed correct)
Matches the official KarlP CalibrationExample notebook exactly:
- DOF format: `[tx, ty, tz, rx, ry, rz]`
- Rotation encoding: `Rotation.from_euler("xyz", dof[3:])` — Euler XYZ intrinsic
- Direction: **cam→base** (T_cam2base), no inversion needed
- To unproject depth: `pts_base = T_cam2base @ pts_cam`
- To project into image: `T_base2cam = inv(T_cam2base)`

### Data source issue (per-episode metadata is WRONG)
The extrinsics stored in per-episode `metadata_*.json` (`ext1_cam_extrinsics`, `ext2_cam_extrinsics`) are **unreliable / uncalibrated**. Confirmed via epipolar geometry:

| F source | Mean epipolar error | % < 3px |
|---|---|---|
| Our extrinsics (metadata_*.json) | 75.87 px | 0.8% |
| RANSAC F (data-driven baseline) | 0.20 px | 100% |

Tried all 13 rotation conventions × 4 T/inv(T) combinations — best result was 48px mean / 24% < 3px. None came close to RANSAC. This rules out a convention mismatch — the values themselves are wrong.

Decomposing the RANSAC F → (R, t) and comparing to our extrinsics-implied relative pose:
- **Rotation error: 86.8°**
- **Translation direction agreement: 55%**

### Correct data source
The authoritative calibrated extrinsics are in `cam2base_extrinsics.json` (DROID annotations), which is the file the KarlP notebook reads from. This file is **not yet available** locally. Once available, the depth pipeline just needs to read DOFs from there instead of `metadata_*.json` — no other code changes needed.

Path expected at: `/ocean/projects/cis240058p/hbhatia1/data/droid_annotations/cam2base_extrinsic_superset.json`

### Debug scripts
- `data_processing/debug_extrinsics.py` — Z-distribution alignment test (too weak, not reliable)
- `data_processing/visualise_eef_projection.py` — EEF projection + cross-camera depth overlay + color error heatmap
- `data_processing/visualise_epipolar.py` — epipolar line visualization with RANSAC vs extrinsics F comparison

---

## ShardedSingleStepDataset behavior
- `episode_sampling_rate=0.1` (default): splits each episode into 10 sub-sequences, uses all timesteps from sampled sub-sequences
- With shard_size=10000, seed=42: produces 599 shards, ~5.99M samples
- With shard_size=1024 (old default): produced ~5850 shards
