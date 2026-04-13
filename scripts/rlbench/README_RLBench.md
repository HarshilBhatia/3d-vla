# RLBench setup for PerAct and PerAct2

## CoppeliaSim / COPPELIASIM_ROOT

RLBench (via PyRep) needs **CoppeliaSim** and the env var **COPPELIASIM_ROOT** set to its install directory.

This repo already includes CoppeliaSim under `PyRep/`. Use one of these as `COPPELIASIM_ROOT` (match your OS):

- **Ubuntu 20.04:** `REPO/PyRep/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04`
- **Ubuntu 18.04:** `REPO/PyRep/CoppeliaSim_Edu_V4_1_0_Ubuntu18_04`

From the **3d-vla repo root**:

```bash
export COPPELIASIM_ROOT="$(pwd)/PyRep/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$COPPELIASIM_ROOT"
export QT_QPA_PLATFORM_PLUGIN_PATH="$COPPELIASIM_ROOT"
```

(Use `Ubuntu18_04` if your system is 18.04.) Add these to `~/.bashrc` if you want them in every shell. Then install PyRep (see main README or below).

## Install

- **PerAct2** (bimanual, 13 tasks):
  ```bash
  ./scripts/rlbench/install_rlbench_peract2.sh
  ```
  Uses [markusgrotz/RLBench](https://github.com/markusgrotz/RLBench).

- **PerAct** (unimanual, 18 tasks):
  ```bash
  ./scripts/rlbench/install_rlbench_peract.sh
  ```
  Uses [MohitShridhar/RLBench](https://github.com/MohitShridhar/RLBench), branch `peract`.

  Install goes to `RLBench_PerAct/` by default so it does not overwrite a PerAct2 RLBench clone. Set `RLBENCH_PERACT_DIR` to use another path.

## close_jar success condition (PerAct)

The original `close_jar` task in RLBench has an incorrect success condition; the sim can keep running or mis-report success. Fix it before generating data or evaluating.

1. See [RLBench issue #255](https://github.com/stepjam/RLBench/issues/255) (and any PR that fixes it in the fork you use).
2. In your PerAct RLBench clone, open `rlbench/tasks/close_jar.py` and adjust the registered success conditions so that “jar lid is closed” is detected correctly (e.g. correct proximity sensor or joint state check).

After editing the task, reinstall if needed: `pip install -e .` from the RLBench root.

## Camera rig rotation (PerAct)

Post-processing rotation in `peract_to_zarr.py` (rotating RGB/PCD after loading .dat) often looks bad: interpolation, cropped corners, and possible coordinate mismatches. For **clean rotated views**, rotate the cameras at **data collection** time instead.

The RLBench scene supports an optional rotation of the entire camera rig around the workspace Z axis at scene init, using `camera.set_pose(position=..., quaternion=...)` for each camera.

- **ObservationConfig**: set `camera_rig_rotation_deg=10` (or any degrees).
- **Dataset generator**: pass `--camera_rig_rotation_deg=10` when collecting demos.

Example (collect with 10° rotated cameras):

```bash
python RLBench/tools/dataset_generator.py \
  --save_path /path/to/peract_raw_rot10 \
  --tasks close_jar \
  --episodes_per_task 5 \
  --camera_rig_rotation_deg 10
```

Then build zarr from that save path **without** any `--rotate_*` flags in `peract_to_zarr.py`; the images are already from the rotated cameras.

**Rot10 zarr sbatch:** `sbatch_experiments/peract_zarr_rot10_transform.sbatch` builds the rot10 zarr from raw data that was collected with the camera rig rotated. It does **not** apply post-hoc image rotation. You must have rotated data in the same layout as Peract_packaged (`train/`, `val/`, `task+var/` with `.dat` files). Set `PERACT_ROOT_ROT10` to that directory (default: `peract_raw_rot10/Peract_packaged/`). If your RLBench dataset generator outputs a different layout, convert or symlink it to that structure before running the sbatch.
