# Dense standard PerAct2 bimanual_lift_ball (chunk8, keypose_segment50)

**Status:** built 13 Oct. chunk8: 13,280 train / 13,273 val rows. keypose_segment50: 403 train / 398 val rows. Every keyframe row is bit-identical to the standard `Peract2_zarr`.
**Week introduced:** [7–13 Oct](../../weekly_logs/2026-10-13.md)

## Why

The orbital dense data ([dense 4-task](dense-4task-dataset.md), [handover targets](handover-dense-targets.md)) mixes two changes: dense targets and orbital cameras. This dataset is the dense-target counterpart of the **standard** PerAct2 setup (front + two wrists, the standard scene) on one easy task, so dense targets can be tested against the standard keypose baseline on the same demos.

## Raw data: the original PerAct2 demos, not a re-collection

The standard zarr `/grogu/datasets/hbhatia/peract2_standard/Peract2_zarr` was made by the upstream 3DFA `data_processing/peract2_to_zarr.py` from the public PerAct2 raw release (`https://dataset.cs.washington.edu/fox/bimanual/image_size_256/{task}.{split}.squashfs`; see [`download_peract2.py`](../../../../scripts/data_generation/download_peract2.py)). That release keeps **every frame** (rgb, depth, mask for all 6 cameras plus `low_dim_obs.pkl`). So instead of re-collecting with new seeds, the original `bimanual_lift_ball` train and val demos were downloaded (SHA256 checked against the server's `SHA256SUM`). These are exactly the episodes behind the standard zarr: 100 train and 100 val, variation 0 (lift_ball has one variation). This matches the standard zarr's lift_ball count (task_id 1: 403 train / 398 val keyframe rows, 100 episode starts each). Re-collecting in the sim would only reproduce the scene distribution, not these scenes.

Conventions copied from the reference converter, and checked bit for bit:

- cameras `[front, wrist_left, wrist_right]`, 256 px
- depth: 24-bit RGB PNG / (2^24 − 1), then `near + d·(far − near)` from `misc`, stored as float16 metres
- extrinsics and intrinsics from `misc`, float16
- state per arm `[gripper_pose (xyz, quat xyzw), gripper_open]`, left arm first, float32
- `keypoint_discovery(demo, bimanual=True)` with frame 0 inserted
- episodes in `sorted(os.listdir)` order (`episode0, episode1, episode10, …`), so `demo_id` order equals the standard zarr's

## Format

Both zarrs keep the standard keys with their dtypes, plus `frame_id`, `is_keyframe`, `keypose_idx`, `episode_idx` (N in `episodeN`), `demo_id`, `next_keypose (1,2,8)` (the state at the next keyframe, i.e. the standard `action`), and `proprioception_keypose (3,2,8)` (the standard keypose history `[k_{i-2}, k_{i-1}, s_t]`). `action_joints` holds joints at the next keyframe, and `proprioception_joints` holds joints at t.

| zarr | rows | `action` | `proprioception` |
|---|---|---|---|
| `chunk8/` | every t in `[0, T−1)` (frames 0..T−2; the last frame is the last keyframe) | `(8,2,8)`: states t+1 … t+8, held at the last frame past the episode end | states `[t−16, t−8, t]`, clamped at 0 |
| `keypose_segment50/` | keyframes only (`kf_0 … kf_{n−1}`) | `(50,2,8)`: reference `hiveformer_to_zarr._interpolate(states[kf_i : kf_{i+1}+1], 50)` per arm (quat → euler `xyz`, CubicSpline on xyz+euler over `linspace(0,1,T)`, nearest for gripper, back to quat) | keypose history `[k_{i−2}, k_{i−1}, k_i]`, padded with the first (= standard) |
| `keypose_segment50_unwrap/` | as above | as above, but the euler angles are `np.unwrap`ped before the spline (**not** in the reference, see below) | as above |

chunk8 is the existing orbital converter's `--target chunk --proprio timestep --proprio-stride 8` semantics.

### The reference spline is broken for this task's rotations

The PerAct2 grippers point down, so the `xyz` euler roll sits near ±π and wraps inside **304 of 806 arm-segments** (train). The reference spline does not unwrap, so it interpolates straight across the 2π jump. In **216 of 403 train rows (214 of 398 val)** the 50-point trajectory spins through a large rotation mid-segment: the largest step between consecutive points is 3.12 rad, against ≤ 0.38 rad with unwrapping. The endpoints are still right. `keypose_segment50/` is the exact reference, as requested. `keypose_segment50_unwrap/` differs only in `action`: all other keys are identical, a large step remains in 1 row per split, and there are no quaternion sign flips between steps. Train on the unwrapped one unless the goal is to reproduce the reference recipe; the handover exp 8e page also unwraps. Endpoint quaternions can carry the opposite sign from the recorded state (euler round trip): first point flipped in 133 arm-rows, last in 114 (train).

## Paths

| what | path |
|---|---|
| raw (original PerAct2, all frames) | `/grogu/datasets/hbhatia/3dfa_dense/peract2_lift_ball/raw/{train,val}/bimanual_lift_ball/all_variations/episodes/episodeN/` (25 GB) |
| chunk8 | `/grogu/datasets/hbhatia/3dfa_dense/peract2_lift_ball/chunk8/{train,val}.zarr` + `build_summary.json` (15 GB) |
| keypose_segment50 | `/grogu/datasets/hbhatia/3dfa_dense/peract2_lift_ball/keypose_segment50/{train,val}.zarr` (413 MB) |
| keypose_segment50_unwrap | `/grogu/datasets/hbhatia/3dfa_dense/peract2_lift_ball/keypose_segment50_unwrap/{train,val}.zarr` (413 MB) |
| smoke test (3 train episodes) | `/grogu/datasets/hbhatia/3dfa_dense/_test/peract2_lift_ball/` |

Code: [`download_lift_ball_raw.sbatch`](../../../../scripts/peract2_dense/download_lift_ball_raw.sbatch) (download, checksum, unsquashfs; skips a split that exists), [`peract2_dense_to_zarr.py`](../../../../data/processing/convert_to_zarr/peract2_dense_to_zarr.py) (`--target chunk|keypose_segment`, `--horizon`, `--chunk-stride`, `--proprio timestep|keypose`, `--proprio-stride`, `--unwrap-euler`; refuses to overwrite without `--overwrite`; one batch of `workers` episodes in memory), [`convert_lift_ball.sbatch`](../../../../scripts/peract2_dense/convert_lift_ball.sbatch) (runs it in the sim container `/home/harshilb/my_eval_env`, where `low_dim_obs.pkl` unpickles into real RLBench observations). Rebuild commands:

```bash
O=/grogu/datasets/hbhatia/3dfa_dense/peract2_lift_ball
sbatch scripts/peract2_dense/convert_lift_ball.sbatch --target chunk --horizon 8 --proprio timestep --proprio-stride 8 --out $O/chunk8
sbatch scripts/peract2_dense/convert_lift_ball.sbatch --target keypose_segment --horizon 50 --proprio keypose --out $O/keypose_segment50
sbatch scripts/peract2_dense/convert_lift_ball.sbatch --target keypose_segment --horizon 50 --proprio keypose --unwrap-euler --out $O/keypose_segment50_unwrap
```

## Counts

| | demos | frames/demo (min / median / max) | chunk8 rows | keyframe rows | segment length (frames, min / median / max) |
|---|---|---|---|---|---|
| train | 100 | 117 / 134 / 165 | 13,280 | 403 | 1 / 25 / 86 |
| val | 100 | 117 / 134 / 160 | 13,273 | 398 | 6 / 25 / 111 |

## Validation (full train and val)

- **Keyframe rows equal the standard `Peract2_zarr` lift_ball rows bit for bit**, in order, for both zarrs: `rgb`, `depth`, `extrinsics`, `intrinsics`, `proprioception_joints`, `action_joints`, `task_id`, `variation`. The segment50 `proprioception` and the chunk8 `proprioception_keypose` equal the standard `proprioception`, and `next_keypose` equals the standard `action`.
- segment50: the first point equals the current state and the last point equals `next_keypose`. xyz error is 0 and gripper is exact. Quaternion error is ≤ 1.2e-7 up to sign. All quaternions are unit.
- chunk8: on every row, `action` = recorded states t+1..t+8 (held at the end) and `proprioception` = states `[t−16, t−8, t]`. 0 mismatches out of 26,553 rows.
- `Peract2_3dfront_3dwrist` (`Peract2Dataset`) loads all three zarrs with `instructions_full.json`: `action (1, 8, 2, 8)` or `(1, 50, 2, 8)`, `proprioception (1, 3, 2, 8)`, `rgb (1, 3, 3, 256, 256)`.

## How to load / caveats for training

`config/data/peract2_lift_ball_chunk8.yaml` and `config/data/peract2_lift_ball_keypose_segment50.yaml` point at the paths above (switch the segment50 one to `keypose_segment50_unwrap` if wanted); use `dataset=Peract2_3dfront_3dwrist bimanual=true keypose_only=false`.

- `task_id` is 1 (PerAct2 index), so no `train_tasks` filter is needed. **Do not set `train_tasks`**: on a single-task zarr `_filter_by_tasks` copies the whole zarr into RAM (15 GB for chunk8).
- Eval: chunk8 needs `prediction_len: 8` and execute/re-query every 8 steps with proprio at stride 8 (`eval_execute_steps`, `eval_proprio_stride`), as in exp 8d. segment50 needs `prediction_len: 50`.
- `train_iters` defaults were set for keypose zarrs; chunk8 has 33× the rows of the keypose data.
- With `visual_num_history > 1`, chunk8 history comes from neighbouring timesteps (demo_id is present), not keyframes.

## Jobs

| job | what |
|---|---|
| 4028443 | download + checksum + unsquashfs of train and val (done) |
| 4028452, 4028453 | smoke conversions, 3 train episodes (passed) |
| **4028457** | chunk8 conversion (done) |
| **4028458** | keypose_segment50 conversion (done) |
| **4028460** | keypose_segment50_unwrap conversion (done) |

Logs: `logs/data/p2dense_{download,convert}-<job>.{out,err}`.
