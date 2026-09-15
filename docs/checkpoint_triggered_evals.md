# Checkpoint-triggered evaluation recipes

Training can submit online evaluations directly after each intermediate
checkpoint. This replaces a polling watcher for newly launched runs.

Set `checkpoint_evals` to a list. Each entry selects a recipe and supplies the
stable method label used in its output path:

```yaml
checkpoint_evals:
  - recipe: instructions/eval_plans/sbrs_checkpoint_ladder_v01.json
    method_id: view_align
    min_step: 70000
    max_step: 200000
  - recipe: instructions/eval_plans/another_checkpoint_recipe.json
    method_id: view_align
```

The recipe—not the trainer—defines tasks, camera groups, calibration/noise
conditions, rollout budget, and output location. Each entry is independently
considered for every `interm_step_<N>.pth`; optional bounds select its step
range. The trainer submits only its exact newly saved checkpoint, so historical
checkpoints are not scanned.

For the current SBRS setup, pass the override when submitting training:

```bash
sbatch --export=ALL,ARM=deltam_external \
  scripts/train/train_peract2_orbital_fixedbias_randall_4gpu.slurm \
  'checkpoint_evals=[{recipe:instructions/eval_plans/sbrs_checkpoint_ladder_v01.json,method_id:view_align,min_step:70000,max_step:200000}]'
```

The rank-zero trainer invokes the ledger-backed checkpoint launcher. It returns
as soon as Slurm accepts the eval array; it never waits for rollouts. Arrays are
submitted without an array concurrency throttle. Submission failures are logged
but do not interrupt training; the checkpoint can be submitted manually with
`scripts/eval/checkpoint_ladder.py`.
