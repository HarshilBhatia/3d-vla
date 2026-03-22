# How to collect rot10 data for PerAct

**Base PerAct data** comes from 3DDA (HuggingFace). You run:

```bash
bash scripts/rlbench/peract_datagen.sh
```

That downloads `Peract_packaged.zip`, unpacks it to `peract_raw/Peract_packaged/` (with `train/`, `val/`, and `task+var/` folders containing `.dat` files), then builds the zarr. There is **no rot10 version** of that download; 3DDA only provides the default camera setup.

To get **rot10 data** (clean 10° rotated views, no post-hoc image rotation), you must **collect demos yourself** with RLBench with the camera rig rotated at scene init, then convert that output into the same layout as Peract_packaged so the existing zarr pipeline can run.

---

## Step 1: Collect demos with RLBench (camera rig rotated 10°)

Use the RLBench dataset generator with `--camera_rig_rotation_deg 10`. From the **3d-vla repo root** (and with RLBench on your path, e.g. `pip install -e RLBench` or `RLBench_PerAct`):

```bash
# Example: collect a subset of tasks to a dedicated rot10 path
python RLBench/tools/dataset_generator.py \
  --save_path peract_raw_rot10 \
  --tasks place_cups close_jar open_drawer \
  --episodes_per_task 10 \
  --image_size 256,256 \
  --camera_rig_rotation_deg 10
```

- Use `--save_path peract_raw_rot10` (or another path you will pass to the converter).
- Add more `--tasks` or use the default (all tasks) if you want full coverage.
- `--image_size 256,256` matches PerAct’s expected resolution.
- **No post-processing rotation** is applied later; the simulator renders from the rotated cameras.

This creates:

- `peract_raw_rot10/<task_name>/all_variations/episodes/episode_0/`, `episode_1/`, …
- In each episode: `low_dim_obs.pkl`, `variation_number.pkl`, and image folders (`left_shoulder_rgb/`, `left_shoulder_depth/`, etc.) with `0.png`, `1.png`, …

---

## Step 2: Convert RLBench output → Peract_packaged layout

The zarr script `peract_to_zarr.py` expects the **Peract_packaged** layout:

- `train/` and `val/`
- Under each: `task+var/` (e.g. `place_cups+0`, `close_jar+3`) containing `.dat` files.

Run the converter (from repo root):

```bash
python scripts/rlbench/rlbench_to_peract_packaged.py \
  --rlbench_save_path peract_raw_rot10 \
  --out peract_raw_rot10/Peract_packaged
```

- `--rlbench_save_path`: same path you passed to `--save_path` in Step 1.
- `--out`: directory that will contain `train/` and `val/` with `task+var/*.dat`. Use `peract_raw_rot10/Peract_packaged` so the rot10 sbatch can use it as `PERACT_ROOT_ROT10` without moving files.

The converter assigns each episode to train or val (e.g. by variation) and writes one `.dat` per episode in the correct `task+var` folder.

---

## Step 3: Build rot10 zarr and (optional) video

Point the rot10 sbatch at the converted data (default is already `peract_raw_rot10/Peract_packaged/`):

```bash
sbatch sbatch_experiments/peract_zarr_rot10_transform.sbatch
```

To override:

```bash
PERACT_ROOT_ROT10=/path/to/peract_raw_rot10/Peract_packaged sbatch sbatch_experiments/peract_zarr_rot10_transform.sbatch
```

Then render the verification video:

```bash
sbatch sbatch_experiments/peract_zarr_videos.sbatch
```

You should see a clean 10° rotated view in `Peract_zarr_rot10deg/verify_peract_rot10.mp4` (no black corners or post-hoc blur).

---

## One-shot: full rot10 sbatch

Run collect, convert, and zarr in one job:

```bash
sbatch sbatch_experiments/peract_rot10_full.sbatch
```

Optional: `EPISODES_PER_TASK=10 TASKS="close_jar place_cups" sbatch ...`  
Requires RLBench (PyRep/CoppeliaSim) and conda env 3dfa. Default 48h limit.

---

## Summary

| Step | Command / action |
|------|-------------------|
| One-shot | `sbatch sbatch_experiments/peract_rot10_full.sbatch` |
| 1. Collect rot10 demos | `python RLBench/tools/dataset_generator.py --save_path peract_raw_rot10 ... --camera_rig_rotation_deg 10` |
| 2. Convert to Peract_packaged | `python scripts/rlbench/rlbench_to_peract_packaged.py --rlbench_save_path peract_raw_rot10 --out peract_raw_rot10/Peract_packaged` |
| 3. Build rot10 zarr | `sbatch sbatch_experiments/peract_zarr_rot10_transform.sbatch` |
| 4. (Optional) Render video | `sbatch sbatch_experiments/peract_zarr_videos.sbatch` |

Base PerAct (no rotation) stays as today: run `peract_datagen.sh` to download 3DDA data and build the base zarr; rot10 is a separate pipeline (collect with rotated cameras → convert → build rot10 zarr).
 `sbatch sbatch_experiments/peract_zarr_rot10_transform.sbatch` |
| 4. (Optional) Render video | `sbatch sbatch_experiments/peract_zarr_videos.sbatch` |

Base PerAct (no rotation) stays as today: run `peract_datagen.sh` to download 3DDA data and build the base zarr; rot10 is a separate pipeline (collect with rotated cameras → convert → build rot10 zarr).
