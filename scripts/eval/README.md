# Evaluation entry points

## Current canonical entry point

- `eval_peract2_clean3d_miscal_camera_subset.slurm` — clean 3D checkpoint
  evaluation with optional camera-subset miscalibration. It supports the
  current external/wrist experiments and a true clean mode. It runs one array
  element per task and defaults to 10 total rollouts per task.

## Historical experiment families

- `online_eval_*` — earlier online RLBench campaigns.
- `eval_*` — older offline/online evaluation wrappers and camera-group runs.
- `sweep_miscal_loss*`, `plot_*`, `analyze_*`, `offline_*`, `collect_*`,
  `reconcile_*`, and `s3_transfer.py` — analysis, aggregation, or transfer
  utilities for prior results.

These files remain in place because historical reports reference their paths.
The active camera sweep should use only the canonical script above; archive
moves should happen after active arrays finish and after references are updated.

## Naming new evaluation cells

Pass `eval_protocol`, `eval_condition_id`, and the `eval_*miscal*` keys to the
evaluator. It writes a per-task manifest alongside the result JSON. See
[`docs/online_eval_naming.md`](../../docs/online_eval_naming.md) for canonical
protocols, condition IDs, layouts, and method labels.
## External-only checkpoint ladder

`checkpoint_ladder.py` evaluates every available `interm_step_<N>.pth` from
step 70k onward. The v01 config uses the six representative seen camera-group
tasks selected for SBRS, both the fixed trained external-only base (`e0`) and
the train-time maximum residual (`e3deg-t1cm`), and 20 rollouts per task:
240 rollouts per checkpoint/arm.

Generate plans for all historical base checkpoints without submitting:

```bash
python scripts/eval/checkpoint_ladder.py \
  --checkpoint-dir train_logs/PerAct2/peract2_orbital_new_external_only_miscal_base \
  --method-id base
```

Add `--submit` to submit one unthrottled 12-cell Slurm array per checkpoint.
The generated plan and submission ledger live alongside the checkpoints, so it
is safe to invoke again.

`watch_checkpoint_ladder.py` polls continuation run directories and submits a
new 12-cell array as each atomic 10k checkpoint appears. Launch its supplied
Slurm wrapper alongside the paired 100k→200k continuations:

```bash
sbatch scripts/eval/watch_checkpoint_ladder.slurm
```

No launcher in this protocol imposes an array concurrency throttle.
