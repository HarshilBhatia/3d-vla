# Commands

There are no tests or linter configured.

## Training

Training uses `torchrun` for multi-GPU DDP:

```bash
# Multi-GPU (canonical)
torchrun --nproc_per_node $ngpus --master_port $MASTER_PORT \
    main.py data=full experiment=camtoken_deltaM_full \
    run_log_dir=my_run checkpoint=train_logs/exp/my_run/last.pth

# Single-GPU / quick runs
python main.py data=single     # 1 task, 30k steps
python main.py data=two        # 2 tasks
python main.py data=full       # 13 PerAct2 tasks, 350k steps

# Hydra overrides — key=value syntax, no -- prefix
python main.py data=full batch_size=32 lr=0.0002 run_log_dir=test_run
```

SLURM scripts are in `scripts/train/`. Template: `train_generic.slurm` (4 GPUs, `shubhamlong` partition, grogu cluster).

## Online Evaluation (CoppeliaSim)

Requires virtual display and CoppeliaSim installed:

```bash
xvfb-run -a bash scripts/eval/online_eval.sh \
    --checkpoint train_logs/Orbital/my_run/last.pth \
    --run-log-dir my_run \
    --extra "miscalibration_noise_level=small seed=1"
```

Results written to `eval_logs/<run_log_dir>/results_<task>.json`.

## Data Conversion

```bash
# PerAct2 PKL → zarr
python data/processing/convert_to_zarr/peract2_to_zarr.py

# Orbital rollouts → zarr
python data/processing/convert_to_zarr/orbital_to_zarr.py \
    --root /path/to/rollouts/ --out /path/to/out.zarr --tasks open_drawer --overwrite
```

## Environment Setup

```bash
export PYTHONPATH=/root/3d_flowmatch_actor:$PYTHONPATH
export COPPELIASIM_ROOT=/root/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04
export LD_LIBRARY_PATH=$COPPELIASIM_ROOT:$LD_LIBRARY_PATH
unset QT_QPA_PLATFORM   # must be unset for headless
```

Set `USER_NAME=HB` to use Harshil's data paths from `paths.py` (default is `LUQMAN`). `paths.py` also adds `./RLBench` and `./PyRep` to `sys.path` — these are local checkouts at the repo root.
