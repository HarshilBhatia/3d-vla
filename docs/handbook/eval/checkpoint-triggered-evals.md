# Checkpoint-triggered evaluation recipes

Training can submit online evaluations directly after each intermediate
checkpoint. This replaces a polling watcher for newly launched runs.

Set `checkpoint_evals` to a list. Each entry selects a recipe and supplies the
stable method label used in its output path:

```yaml
checkpoint_evals:
  - recipe: instructions/eval_plans/robustness_ladder_ema_v01.json
    method_id: my_arm_ema
    min_step: 100000
  - recipe: instructions/eval_plans/sbrs_sanity_ladder_ema_v01.json
    method_id: my_arm_ema
```

This pair is the standard for new video experiments (see
[protocols.md](protocols.md#checkpoint-ladders-and-checkpoint-selection));
use `robustness_ladder_raw_v01.json` for a run without EMA. The trainer passes
its exact step as both bounds, so a recipe's own `min_step` only applies to a
manual `checkpoint_ladder.py` call: set `min_step` in the entry. The recipe's
`step_interval` always applies.

## Recipe schemas

Both use `schema_version: 1`, `protocol_id`, `description`, `output_root`,
`min_step`, `step_interval` and `runtime` (`num_demos_total` is the rollout
budget; `eval_weights` and other evaluator args go in `runtime.overrides`,
never at the top level of `runtime`, where the evaluator ignores them).

- **SBRS-shaped** (`tasks` with `spawn_camera_group`, `calibration_registry`,
  `base_level`, `realization_version`, `residual_conditions`): one plan per
  checkpoint, ledger key `<method>:s<step>`.
- **Source-plan** (`source_plans`): each entry is
  `{"name", "plan", "calibrations"}` — an existing campaign plan and the
  calibration levels (its `calibrations` labels) to keep. Tasks, viewpoints and
  registry come from that plan; methods are replaced by the one checkpoint.
  The recipe's `tasks` must equal every source plan's `tasks` (a guard against
  a source plan changing under the ladder).
  One plan and one array per source per checkpoint, ledger key
  `<method>:s<step>:<name>`, plan file
  `checkpoint_ladder_plans/<protocol>_<method>_s<step>_<name>.json`.

The recipe—not the trainer—defines tasks, camera groups, calibration/noise
conditions, rollout budget, and output location. Each entry is independently
considered for every `interm_step_<N>.pth`; optional bounds select its step
range. The trainer submits only its exact newly saved checkpoint, so historical
checkpoints are not scanned.

The equivalent override (prefer the experiment config, which is the record of the run):

```bash
sbatch --export=ALL,ARM=deltam_external \
  scripts/train/train_peract2_orbital_fixedbias_randall_4gpu.slurm \
  'checkpoint_evals=[{recipe:instructions/eval_plans/robustness_ladder_ema_v01.json,method_id:my_arm_ema,min_step:100000}]'
```

The rank-zero trainer invokes the ledger-backed checkpoint launcher. It returns
as soon as Slurm accepts the eval array; it never waits for rollouts. Arrays are
submitted without an array concurrency throttle. Submission failures are logged
but do not interrupt training; the checkpoint can be submitted manually with
`scripts/eval/checkpoint_ladder.py`.

Retroactive ladder on a finished run (dry run without `--submit`):

```bash
PYTHONPATH=$PWD python scripts/eval/checkpoint_ladder.py \
  --checkpoint-dir train_logs/PerAct2/<run> --method-id <arm>_raw \
  --config instructions/eval_plans/robustness_ladder_raw_v01.json --max-step 200000 --submit
```
