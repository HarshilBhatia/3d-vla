# Dense (per-timestep) orbital PerAct2, 4-task subset

**Status:** built 12 Oct: 67,782 train rows (324 demos, 52 GB) and 7,577 val rows (36 demos, 5.7 GB)
**Week introduced:** [7–13 Oct](../../weekly_logs/2026-10-13.md)

## Why

The orbital PerAct2 zarr is keypose-only: one record per keyframe, `action` = next keypose. To train dense prediction we need a record for every demo timestep, with the action being the trajectory to the next keypose. Subset: `bimanual_handover_item`, `bimanual_take_tray_out_of_oven`, `bimanual_dual_push_buttons`, `bimanual_pick_laptop` (the [exp 7 subset](4task-flow-and-actions.md)).

## Raw frames had to be re-collected

The Aug 2026 collection ran on a docker swarm (raw episodes at `/out/raw/...` inside the container) and deleted each raw episode once its shard was converted ([`merge_orbital_shards.py`](../../../../data/processing/convert_to_zarr/merge_orbital_shards.py) header). The egress bundle's `shards/` are keyframe zarrs, not raw frames, and no raw orbital episode exists on grogu. The original collection did not seed `np.random` and the zarr does not keep `demo.random_seed`, so the **exact** original scenes cannot be regenerated.

What is kept identical: tasks and `task_id`, the train camera groups per task (from `instructions/peract2_orbital_task_group_mapping.json`, identical to the egress copy), camera poses (orbital extrinsics/intrinsics are bit-identical to the existing zarr for the same group), 30 episodes per (task, group), variation cycling `variation = episode % n_variations`, the split rule (lexicographic episode order, first 27 to train, so `episode_7/8/9` go to val), the shard-merge order and therefore `demo_id` order, and 256 px / 60° FOV. What differs: the scene initialisations (new random seeds). Episodes are now seeded deterministically (`crc32("task/group/episode/retry")`, stored in `seed.txt`), so the raw data is reproducible — re-collecting `pick_laptop/G2/episode_0` gave the same 122-frame demo twice.

Raw episodes are **kept** this time, so other dense formats can be re-derived without the simulator.

## Format

One record per timestep `t` in `[kf_i, kf_{i+1})` for every keypose segment, `kf_0 = 0` (same `keypoint_discovery` as the keypose converter). The last keyframe has no record, as before. Unchanged keys keep their dtype and per-row shape (`rgb`, `depth`, `extrinsics`, `intrinsics` for cameras `[orbital_left, orbital_right, wrist_left, wrist_right]`, `proprioception_joints`, `action_joints`, `task_id`, `variation`, `camera_group`, `demo_id`).

| key | shape per row | meaning |
|---|---|---|
| `action` | `(8, 2, 8)` | remaining segment `t → kf_{i+1}` resampled to H=8 at `t + j/8·(kf_{i+1}−t)`, `j=1..8`; xyz linear, quat slerp, gripper from the later frame. **`action[-1]` is bit-identical to `next_keypose`** |
| `next_keypose` | `(1, 2, 8)` | state at `kf_{i+1}` (the old `action`) |
| `proprioception` | `(3, 2, 8)` | dense history `[s_{t-2}, s_{t-1}, s_t]`, clamped at demo start |
| `proprioception_keypose` | `(3, 2, 8)` | keypose history: last two keyframes strictly before `t`, then `s_t` (equals the old `proprioception` on keyframe rows) |
| `action_joints` | `(1, 2, 8)` | joints at `kf_{i+1}` (as before) |
| `frame_id`, `keypose_idx`, `is_keyframe`, `episode_idx` | scalar | timestep, segment `i`, `t == kf_i`, raw episode number |

Why: H=8 resampled to the keypose keeps every target ending exactly where the keypose models aim, so `keypose_only=true` (which takes `actions[:, [-1]]` in `DataPreprocessor.process_actions`) trains a keypose model from every timestep with no loader change, and `keypose_only=false` gets an 8-step trajectory. A fixed-time chunk would cross keyposes and lose that property. Segment lengths are 1 to ~60 frames, so 8 points subsample long segments and upsample short ones. `is_keyframe` rows with `next_keypose` / `proprioception_keypose` reproduce the old keypose record exactly (validated below).

## Paths

| what | path |
|---|---|
| raw episodes | `/grogu/datasets/hbhatia/3dfa_dense/orbital_peract2_4task/raw/{task}/{group}/episode_{N}/` |
| dense zarr | `/grogu/datasets/hbhatia/3dfa_dense/orbital_peract2_4task/{train,val}.zarr` + `build_summary.json` |
| smoke test (2 episodes, pick_laptop/G2) | `/grogu/datasets/hbhatia/3dfa_dense/_test/{raw,zarr}` |

Code: [`collect_dense.py`](../../../../scripts/orbital_cameras/collect_dense.py) (seeded collection, reuses `data/generation/orbital/collection.py`), [`collect_dense_4task.sbatch`](../../../../scripts/orbital_cameras/collect_dense_4task.sbatch) (array element = task × group × 10-episode block), [`orbital_dense_to_zarr.py`](../../../../data/processing/convert_to_zarr/orbital_dense_to_zarr.py) and [`convert_dense_4task.sbatch`](../../../../scripts/orbital_cameras/convert_dense_4task.sbatch). The converter must run inside the sim container (`/home/harshilb/my_eval_env`): on host, `CustomUnpickler` turns `Observation` into a `Stub`, which hides the PerAct2 wrist depth. It refuses to build if any shard has fewer than 30 episodes, since that would shift the split.

## How to load

```bash
python main.py data=orbital_peract2_dense_4task dataset=OrbitalPeract2 bimanual=true ...
```

[`config/data/orbital_peract2_dense_4task.yaml`](../../../../config/data/orbital_peract2_dense_4task.yaml) points at the paths above; `OrbitalPeract2Dataset` loads it unchanged (extra keys are ignored). Caveats: visual history (`visual_num_history>1`) now draws from neighbouring **timesteps**, not keyframes; a loader chunk (`chunk_size`) is consecutive timesteps of one demo; `train_iters` defaults were set for the keypose zarr.

## Jobs

| job | what | partition |
|---|---|---|
| 4027070, 4027074 | smoke tests (first wrote to `/home`, not bound in the container; second passed) | shubhamlong |
| **4027075** `[0-35]` | collection, 12 shards × 3 blocks of 10 episodes | shubhamlong,all |
| **4027112** | conversion, `afterany:4027075`; killed by the 64 GB memory limit after 35 episodes (finished episodes queued behind the writer) | shubhamlong,all |
| **4027161** | conversion rerun with batched episodes (one batch of `workers` in memory), 160 GB, `--overwrite`; done | shubhamlong |

Check: `squeue -u harshilb -n dense4_collect,dense4_convert`, logs in `logs/data/dense4_{collect,convert}-*`. A failed element is re-run with `sbatch --array=<idx> scripts/orbital_cameras/collect_dense_4task.sbatch` (finished episodes are skipped), then the conversion is resubmitted.

## Record counts

Smoke test: pick_laptop episode 0 → 121 rows (7 keyframe rows, as in the old zarr), episode 1 → 119. Expected from the first collected episodes (frames/episode ≈ rows): handover ~260, take_tray ~320, dual_push_buttons ~105, pick_laptop ~140 → **~74k rows** total (~66k train / ~7.4k val), ~0.8 MB per row compressed (~60 GB), raw ~90 GB. Old keypose zarr for these tasks: 2915 train / 329 val rows. Final (`build_summary.json`):

| Task | Rows per demo (train) | Keyframes per demo |
|---|---|---|
| dual_push_buttons | 106 | 4.0 |
| handover_item | 252 | 10.9 |
| pick_laptop | 150 | 7.0 |
| take_tray_out_of_oven | 329 | 14.2 |

Keyframes per demo match the old keypose zarr. On 500 random rows per split the last action step equals `next_keypose` exactly, and `OrbitalPeract2Dataset` loads the val split with K = 5 history unchanged.

## Validation (smoke test)

- shapes as above; `OrbitalPeract2Dataset` returns `action (4, 8, 2, 8)`, `proprioception (4, 3, 2, 8)`; with `visual_num_history=3`, `rgb (1, 3, 4, 3, 256, 256)`
- `action[:, -1] == next_keypose` exactly for every row
- keyframe rows equal, bit for bit, what the existing keypose converter (`orbital_utils.process_episode`) writes for the same raw episode: `next_keypose`, `proprioception_keypose`, rgb, depth, extrinsics, `action_joints`, variation
- orbital extrinsics/intrinsics for G2 match the existing zarr exactly; unit quaternions; first trajectory point within 5.4 cm of the current state
