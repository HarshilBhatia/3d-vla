# Hydra config

All entry points (main.py, online_evaluation_rlbench/evaluate_policy.py, analyse_qk.py) use the same config file and override from the CLI with **key=value** (no `--`).

**Config groups:**
- **dataset** – dataset *type* (which class): `peract2` | `peract` | `peract2_singlecam` | `peract_twocam` → see `config/dataset/`.
- **data** – which *tasks/split*: `single` | `two` | `full` → see `config/data/`. Paths are relative to project root. (Defaults use `@_global_` so these keys are merged at root.)
- **rope_mode** – RoPE variant: `none` | `standard` → see `config/rope_mode/`. Sets `traj_scene_rope`, `sa_blocks_use_rope`.
- **experiment** – run-specific overrides: `default` | `one_task` | `full` | `drope` → see `config/experiment/`.

- **run_mode:** Derived from `eval_only`: `eval_only=true` → `run_mode=eval_offline`; else `run_mode=train`.
- **use_front_camera_frame:** Canonical name (not `front_camera_frame`).

Examples:

```bash
# Train with different data configs (single task, 2 tasks, or full)
python main.py data=single
python main.py data=two
python main.py data=full

# Override paths if needed
python main.py data=two train_data_dir=/other/path/to/train.zarr

# Offline eval
python main.py checkpoint=path/to/best.pth eval_only=true

# Online eval
python online_evaluation_rlbench/evaluate_policy.py checkpoint=path/to/best.pth task=close_jar
```

Defaults live in `config/config.yaml`. Overrides are passed as `key=value`; lists and nested config are supported by Hydra when needed.

**Scripts updated to Hydra:** `main.py`, `online_evaluation_rlbench/evaluate_policy.py`, `analyse_qk.py`; `run_multi_task_zarr.sh`, `jobs/train_generic.slurm`, `jobs/train_1task.slurm`, `jobs/train_multitask.slurm`, `jobs/train_full.slurm`, `jobs/train_drope.slurm`, `online_evaluation_rlbench/eval_peract2.sh`, `scripts/rlbench/train_peract2.sh`. Any other script that calls `main.py` or `evaluate_policy.py` with `--key value` should be updated to `key=value`.
