# Guide: Generating Raw Demos for PerAct with Orbital Cameras

## Architecture overview

The pipeline has two branches that feed different uses:

```
CoppeliaSim (via xvfb-run)
    ↓
collect_orbital_rollouts.py          # produces raw episodes
    ↓
data/orbital_rollouts/
  {task}/{group}/episode_{N}/
    orbital_left_rgb/   *.png
    orbital_right_rgb/  *.png
    wrist_rgb/          *.png
    orbital_left_depth/ *.png        # RGB-encoded float (DEPTH_SCALE = 2^24-1)
    orbital_right_depth/*.png
    wrist_depth/        *.png
    low_dim_obs.pkl                  # full Demo object (gripper_pose, joint_positions, ...)
    orbital_extrinsics.pkl           # {left,right}_{extrinsics,intrinsics}
    camera_group.txt                 # "G2\n"
    ↓
data_generation/orbital/to_zarr.py  # → data/orbital_train.zarr  (for training)
```

Online eval needs `data_dir` to point at **standard RLBench demo format**
(`task/variationN/episodes/episodeN/low_dim_obs.pkl`) — separate from the
orbital rollout data — because `get_stored_demos` and `reset_to_demo()` expect
that layout. The orbital cameras are spawned dynamically on top.

---

## Prerequisites

### 1. CoppeliaSim env vars
```bash
export COPPELIASIM_ROOT="$(pwd)/PyRep/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$COPPELIASIM_ROOT"
export QT_QPA_PLATFORM_PLUGIN_PATH="$COPPELIASIM_ROOT"
```
The Ubuntu20 build ships in `PyRep/` in the repo. Use `Ubuntu18_04` on older systems.

### 2. Python environment
```bash
conda activate 3dfa
# If torch.from_numpy fails (numpy >= 2 + torch 2.0 incompatibility):
pip install "numpy<2"
```

### 3. `close_jar` success-condition fix
The upstream PerAct RLBench has a broken success condition for `close_jar`.
Before collecting, apply the fix from
[RLBench #255](https://github.com/stepjam/RLBench/issues/255) in
`RLBench/rlbench/tasks/close_jar.py`, then `pip install -e RLBench/`.

---

## Step 0: Camera group setup

`orbital_cameras_grouped.json` defines 6 camera groups (G1–G6), each with a
`left` and `right` orbital camera position. This file already exists in the
repo root and has been used for training. **Skip this step if you are reusing
the existing camera groups.**

To generate or extend it, use the spherical sampling script and the rerun
visualizer:
```bash
python scripts/rlbench/sample_camera_positions.py \
    --zarr <any_peract2_val.zarr> \
    --out camera_samples/ \
    --rotate_z "-30,-15,0,15,30" \
    --translate_x "-0.2,0,0.2"
# Opens camera_samples/index.html — pick positions visually

# Visualize chosen positions against a live task scene:
xvfb-run -a bash scripts/rlbench/gen_orbital_cameras.sh
# writes orbital_viz/<task>.rrd files (view in rerun)
```

Each entry in the JSON has the form:
```json
{
  "group": "G1",
  "left":  { "name": "L_el10_0", "pos": [x,y,z], "R": [[...],[...],[...]] },
  "right": { "name": "R_el10_0", "pos": [x,y,z], "R": [[...],[...],[...]] }
}
```
`R` is a 3×3 rotation matrix (camera-to-world). `create_orbital_sensor()`
converts it to a PyRep quaternion pose.

---

## Step 1: Task-to-group mapping

`task_group_mapping.json` maps each task to 3 camera groups. The assignment is
cyclic: task `i` → `[G_{i%6+1}, G_{(i+1)%6+1}, G_{(i+2)%6+1}]`, so each
group appears in exactly 9 of the 18 tasks.

```bash
python scripts/rlbench/create_task_group_mapping.py
# writes task_group_mapping.json to repo root
```

This file already exists for the standard 18 PerAct tasks. Regenerate only if
you add/remove tasks or groups.

---

## Step 2: Collect raw episodes

### Debug run (1 episode, saves video + single-episode zarr)
```bash
xvfb-run -a python scripts/rlbench/collect_orbital_rollouts.py \
    --task close_jar --groups G2 --video-only \
    --cameras-file orbital_cameras_grouped.json \
    --image-size 256 --fov-deg 60.0 \
    --video-dir debug_videos/
# outputs: debug_videos/close_jar_G2.mp4
#          debug_videos/close_jar_G2.mp4.zarr/
```

### Full collection (one task, all its groups, 30 eps each)
```bash
xvfb-run -a python scripts/rlbench/collect_orbital_rollouts.py \
    --task close_jar --groups G2 G3 G4 \
    --n-episodes 30 \
    --save-path data/orbital_rollouts \
    --cameras-file orbital_cameras_grouped.json \
    --image-size 256 --fov-deg 60.0
# outputs: data/orbital_rollouts/close_jar/{G2,G3,G4}/episode_{0..29}/
```

All 3 groups run in a single CoppeliaSim launch (avoids 3× startup overhead).
The script auto-resumes if interrupted: it counts existing `episode_*` dirs and
starts from where it left off.

### All 18 tasks locally (sequential, ~1620 episodes)
```bash
xvfb-run -a bash scripts/rlbench/run_orbital_datagen.sh
# Optionally override episode count:
N_EPISODES=5 xvfb-run -a bash scripts/rlbench/run_orbital_datagen.sh
```

### All 18 tasks on a Slurm cluster (54-way parallel)
```bash
# One (task, group) pair per array element — 54 elements total
bash scripts/rlbench/submit_orbital_collection.sh

# Dry run first to inspect sbatch commands:
DRY_RUN=1 bash scripts/rlbench/submit_orbital_collection.sh

# Resubmit specific failed indices (check SLURM logs to identify):
sbatch --array=7,23,41 scripts/rlbench/collect_orbital_slurm.sbatch
```

The sbatch script expects:
- Repo at `/ocean/projects/cis240058p/hbhatia1/3d-vla` (update `REPO_DIR` for your cluster)
- Apptainer container at `containers/3dfa-sandbox.sif`
- `COPPELIASIM_ROOT` set inside the container env

---

## Step 3: Convert to training zarr

```bash
python data_generation/orbital/to_zarr.py \
    --root data/orbital_rollouts \
    --out  data/orbital_train.zarr \
    --image-size 256

# Include only specific tasks:
python data_generation/orbital/to_zarr.py \
    --root data/orbital_rollouts \
    --out  data/orbital_train.zarr \
    --tasks "close_jar,open_drawer,turn_tap"

# Include only specific camera groups:
python data_generation/orbital/to_zarr.py \
    --root data/orbital_rollouts \
    --out  data/orbital_train.zarr \
    --groups "G2,G3"

# Overwrite an existing zarr:
python data_generation/orbital/to_zarr.py \
    --root data/orbital_rollouts \
    --out  data/orbital_train.zarr \
    --overwrite
```

Verify the output:
```bash
python3 -c "
import zarr
z = zarr.open('data/orbital_train.zarr')
print({k: z[k].shape for k in z})
"
# Expected keys: rgb, depth, extrinsics, intrinsics,
#   proprioception, action, proprioception_joints, action_joints,
#   task_id, variation, camera_group
```

---

## For online eval: standard RLBench demos

The orbital rollout format (`{task}/{group}/episode_{N}/`) **does not match**
what `get_stored_demos` expects (`{task}/variation{N}/episodes/episode{N}/low_dim_obs.pkl`).
For online eval, you need separately generated standard RLBench demos.

Generate them with the RLBench dataset generator:
```bash
xvfb-run -a python RLBench/tools/dataset_generator.py \
    --save_path data/peract_raw \
    --tasks close_jar open_drawer turn_tap \
    --episodes_per_task 5 \
    --processes 1 \
    --all_variations True
```

Then pass this to the eval script via `data_dir`:
```bash
xvfb-run -a bash scripts/rlbench/eval_orbital_grogu_best.sh \
    data_dir=$(pwd)/data/peract_raw
```

> **Note:** The eval script currently passes `eval_data_dir` (the zarr training
> data path, unused by `evaluate_policy.py`). The code reads `args.data_dir`.
> Either override it on the command line as shown above, or fix the eval script
> to set `data_dir=...` instead of `eval_data_dir=...`.

---

## Episode format reference

Each saved episode directory contains:

| File / Folder | Contents |
|---|---|
| `orbital_left_rgb/*.png` | RGB frames (uint8, HxWx3) |
| `orbital_left_depth/*.png` | Depth encoded as RGB; decode with `image_to_float_array(img, DEPTH_SCALE=2^24-1)` to get absolute metres |
| `orbital_right_rgb/`, `orbital_right_depth/` | Same for right orbital camera |
| `wrist_rgb/`, `wrist_depth/` | Wrist camera; depth uses RLBench near/far convention stored in `obs.misc["wrist_camera_near/far"]` |
| `low_dim_obs.pkl` | List of RLBench `Observation` objects (full Demo) — has `gripper_pose`, `joint_positions`, `misc["wrist_camera_extrinsics"]`, etc. |
| `orbital_extrinsics.pkl` | `{left_extrinsics, right_extrinsics, left_intrinsics, right_intrinsics}` — 4×4 and 3×3 float32 arrays captured while sensors are alive |
| `camera_group.txt` | Group name string, e.g. `"G2"` |
