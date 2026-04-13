# Orbital Datagen Pipeline

All core orbital logic lives in the `orbital/` package at the repo root.
CLI entry points in `scripts/rlbench/` are thin wrappers.
Old file locations (`data_generation/orbital_rlbench.py`, `data_processing/orbital_to_zarr.py`)
are backward-compat re-exports.

---

## Package Layout

```
orbital/
  constants.py     # PERACT_TASKS, DEPTH_SCALE, NCAM, NHAND, num2id()
  scene.py         # OrbitalScene, OrbitalEnvironment
  collection.py    # sensor helpers, obs config, episode saving, debug video/zarr,
                   # collect_one_episode()
  to_zarr.py       # raw episode → train.zarr conversion + CLI
  task_mapping.py  # build_mapping(), verify_mapping(), main()
```

---

## Data Files (must exist before running)

| File | Created by | Purpose |
|------|-----------|---------|
| `task_group_mapping.json` | `orbital/task_mapping.py` | Maps 18 PerAct tasks → 3 camera groups each |
| `orbital_cameras_grouped.json` | (pre-existing) | Per-group left/right orbital camera poses (pos + R) |

---

## Entry Point Scripts

```
scripts/rlbench/run_orbital_datagen.sh     # full collection: 18 tasks × 3 groups × 30 eps
scripts/rlbench/run_orbital_debug.sh       # debug: 1 episode/group → MP4 + zarr
  └─ both call: scripts/rlbench/collect_orbital_rollouts.py
                  └─ imports from orbital.collection + orbital.scene

scripts/rlbench/create_task_group_mapping.py
  └─ thin wrapper around orbital.task_mapping.main()
```

---

## `orbital/constants.py`

Single source of truth for shared values:
- `PERACT_TASKS` — 18 task names (previously duplicated in `create_task_group_mapping.py` and `orbital_to_zarr.py`)
- `DEPTH_SCALE` — RGB depth encoding scale (previously duplicated)
- `NCAM`, `NHAND`
- `num2id(i)` — zero-pad frame index (previously duplicated)

---

## `orbital/scene.py`

| Class | Inherits from | Role |
|-------|--------------|------|
| `OrbitalScene` | `CustomizedScene` | Overrides `get_observation()` to capture orbital sensors; wraps `step()` with timers |
| `OrbitalEnvironment` | `CustomizedEnvironment` | Replaces `_scene` with `OrbitalScene` after `Environment.launch()` |

---

## `orbital/collection.py`

All logic needed during a collection session:

| Function | Role |
|----------|------|
| `load_group_cameras(cameras_file, group)` | Parse orbital camera poses from JSON |
| `create_orbital_sensor(pos, R_mat, image_size, fov_deg)` | Spawn PyRep VisionSensor |
| `capture_orbital_extrinsics(left, right)` | Capture 4×4 E + 3×3 K for both sensors |
| `make_obs_config(image_size)` | Build RLBench ObservationConfig |
| `save_orbital_episode(demo, ep_path, group, extrinsics)` | Save RGB/depth PNGs + pkl |
| `collect_one_episode(task_env, scene, ...)` | Main per-episode driver (reset → collect → cleanup) |
| `save_debug_video(demo, video_out, image_size)` | 3-panel MP4 for visual inspection |
| `save_debug_zarr(demo, zarr_path, group, ...)` | Single-episode zarr for pipeline validation |

---

## `orbital/to_zarr.py`

Converts saved raw episodes to a single `train.zarr`:

```
python -m orbital.to_zarr \
    --root data/orbital_rollouts \
    --out  data/orbital_train.zarr
```

| Function | Role |
|----------|------|
| `process_episode(ep_path, task_id, group_str, zarr_file)` | Load one episode, extract keyframes, append to zarr |
| `load_rgb / load_depth_metres` | Load PNG frames from disk |
| `load_orbital_extrinsics(ep_path)` | Load pre-saved extrinsics pkl |

---

## `orbital/task_mapping.py`

| Function | Role |
|----------|------|
| `build_mapping(tasks)` | Cyclic group assignment: task i → [G_{i%6+1}, G_{(i+1)%6+1}, G_{(i+2)%6+1}] |
| `verify_mapping(mapping)` | Assert each group appears in exactly 9 tasks |
| `main()` | Write `task_group_mapping.json` to repo root |

---

## Dependency Graph

```
run_orbital_{datagen,debug}.sh
├── [prereq] orbital/task_mapping.py  →  task_group_mapping.json
├── [prereq] orbital_cameras_grouped.json
└── scripts/rlbench/collect_orbital_rollouts.py   (thin CLI)
    ├── orbital/collection.py
    │   └── orbital/constants.py
    └── orbital/scene.py
        └── data_generation/customized_rlbench.py
            └── rlbench + pyrep (external libs)

data_processing/ (post-collection)
└── orbital/to_zarr.py
    ├── orbital/constants.py
    └── data_processing/rlbench_utils.py  (keypoint_discovery, image_to_float_array)
```
