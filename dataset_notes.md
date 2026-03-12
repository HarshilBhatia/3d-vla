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

## ShardedSingleStepDataset behavior
- `episode_sampling_rate=0.1` (default): splits each episode into 10 sub-sequences, uses all timesteps from sampled sub-sequences
- With shard_size=10000, seed=42: produces 599 shards, ~5.99M samples
- With shard_size=1024 (old default): produced ~5850 shards
