# Setup

Covers the simulator stack (CoppeliaSim / PyRep / RLBench) and the per-user
path config. For what to run once this works, see [commands.md](commands.md).

## Environment variables

CoppeliaSim ships in `PyRep/` in this repo. Point `COPPELIASIM_ROOT` at the
build matching your OS:

```bash
export COPPELIASIM_ROOT="$(pwd)/PyRep/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"   # or Ubuntu18_04
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$COPPELIASIM_ROOT"
export QT_QPA_PLATFORM_PLUGIN_PATH="$COPPELIASIM_ROOT"
export PYTHONPATH="$(pwd):$PYTHONPATH"
unset QT_QPA_PLATFORM   # must be unset for headless
```

Set `USER_NAME=HB` to use Harshil's data paths from `paths.py` (default is
`LUQMAN`). `paths.py` also adds `./RLBench` and `./PyRep` to `sys.path` — these
are local checkouts at the repo root.

## Installing RLBench

- **PerAct2** (bimanual, 13 tasks) — [markusgrotz/RLBench](https://github.com/markusgrotz/RLBench):
  ```bash
  ./scripts/rlbench/install_rlbench_peract2.sh
  ```
- **PerAct** (unimanual, 18 tasks) — [MohitShridhar/RLBench](https://github.com/MohitShridhar/RLBench), branch `peract`:
  ```bash
  ./scripts/rlbench/install_rlbench_peract.sh
  ```
  Installs to `RLBench_PerAct/` so it does not overwrite a PerAct2 clone. Set
  `RLBENCH_PERACT_DIR` to use another path.

If `torch.from_numpy` fails, it is the numpy>=2 / torch 2.0 incompatibility:
`pip install "numpy<2"`.

## `close_jar` success condition (PerAct)

The upstream `close_jar` task has an incorrect success condition — the sim can
keep running or mis-report success. Fix it before generating data or
evaluating.

1. See [RLBench issue #255](https://github.com/stepjam/RLBench/issues/255) and
   any PR fixing it in the fork you use.
2. In your PerAct RLBench clone, edit `rlbench/tasks/close_jar.py` so that "jar
   lid is closed" is detected correctly (correct proximity sensor or joint
   state check).
3. Reinstall if needed: `pip install -e .` from the RLBench root.

## Camera rig rotation (PerAct)

Post-hoc rotation in `peract_to_zarr.py` (rotating RGB/PCD after loading `.dat`)
looks bad: interpolation, cropped corners, possible coordinate mismatches. For
clean rotated views, rotate the cameras at **collection** time.

The scene supports rotating the whole camera rig around the workspace Z axis at
scene init via `camera.set_pose(...)`:

- `ObservationConfig`: set `camera_rig_rotation_deg=10`.
- Dataset generator: pass `--camera_rig_rotation_deg=10`.

```bash
python RLBench/tools/dataset_generator.py \
  --save_path /path/to/peract_raw_rot10 \
  --tasks close_jar --episodes_per_task 5 \
  --camera_rig_rotation_deg 10
```

Then build the zarr from that path **without** any `--rotate_*` flags — the
images already come from rotated cameras.

`sbatch_experiments/peract_zarr_rot10_transform.sbatch` builds the rot10 zarr
this way. It expects the Peract_packaged layout (`train/`, `val/`,
`task+var/` with `.dat` files) at `PERACT_ROOT_ROT10` (default
`peract_raw_rot10/Peract_packaged/`); convert or symlink if your generator
output differs.
