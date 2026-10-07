# Online-evaluation nomenclature and framework

An online-evaluation cell is exactly one method, viewpoint, **calibration
realization**, task, seed, and rollout budget. A calibration realization is the
complete per-camera transform table \(\delta\) used by that cell. It is never a
runtime recipe, a distribution, or a magnitude alone.

## Public names

Use `calibration realization` in prose and `eval_calibration_id` in configs.
The ID says what was evaluated; its registry supplies the exact matrices.

```text
calibrated
ext-r2deg-t2cm-v01
ext-r5deg-t5cm-v01
ext-r10deg-t10cm-v01
```

`ext` means only the external orbital cameras are non-identity. `r5deg-t5cm`
is a nominal magnitude label; `v01` distinguishes the fixed sampled direction.
It must not be dropped: another direction at the same magnitude is a different
realization. The bimanual registry currently keeps both wrist cameras at
identity in every `ext-*` realization.

The stored table uses the repository convention:

```text
T_observed = delta @ T_true
```

No public result label uses `base`, `probe`, `sampled`, `group`, or a
composition chain. If a research procedure builds a delta from components, it
must materialize and version the final table before evaluation.

## Registry and campaign plan

`instructions/eval_calibrations.json` is the versioned registry. It includes
the camera order and every 4×4 matrix. The evaluator validates SO(3), the
homogeneous row, camera order, and realization ID before starting RLBench.

An evaluation campaign is a JSON plan under `instructions/eval_plans/`. The
plan declares methods/checkpoints, viewpoints, calibration IDs, tasks, and
runtime budget. `scripts/eval/run_eval_plan.py` takes only a plan and cell
index. Its stable Cartesian-product order is:

```text
method × viewpoint × calibration realization × task
```

For the published PerAct2 rig, use `standard_front_wrist` as the viewpoint ID:
the native cameras are `front`, `wrist_left`, and `wrist_right`. Model family
is a method attribute, never a viewpoint regime: for example,
`denoise2d_s<step>` is a 2D method evaluated at
`standard_front_wrist | calibrated`. The canonical clean published-data
campaign is `instructions/eval_plans/peract2_original_clean_v01.json`; its
camera-specific identity table is
`instructions/eval_calibrations_peract2_standard.json`. Unlike orbital plans,
this native RLBench rig has no `cameras_file` or spawn-camera group.

For the G1--G6 task-heldout protocol, use `task_viewpoints` instead of a
shared `viewpoints` list. It maps each task to its one held-out camera group,
so the cells are `method × calibration realization × task` rather than the
incorrect all-tasks × all-groups cross product. Its manifest regime is
`task_heldout_group`; reserve `fully_ood` for G7.

For a protocol whose calibration realization must match each task's viewpoint,
add `task_calibrations`. It maps every task to its exact calibration IDs, so
cells are `method × task × task_calibration` rather than a cross product over
all groups. This prevents a valid G1 table from silently being evaluated at G2.

## Seen-Base Residual Sweep (SBRS)

`seen_base_residual_sweep_v01` is the in-distribution robustness protocol for
checkpoints trained with a persistent per-group table \(\delta_{i,j}\) and a
random top-up \(\epsilon\). Its materialized condition is
\(T_{observed} = (\epsilon @ \delta_{i,j}) @ T_{true}\), using the same camera
group seen by that task during training. `e0` means \(\epsilon=I\): it is the
trained fixed-base condition, **not** globally clean/calibrated extrinsics.

The v01 residual ladder is `e0`, `e2deg-t2cm`, `e5deg-t5cm`, and
`e3deg-t1cm` (the maximum augmentation magnitude used in training). Exact
tables are in `instructions/eval_calibrations_seen_base_residual_v01.json` and
are generated reproducibly by:

```bash
python scripts/eval/materialize_seen_base_residual_registry.py \
  --residual e2deg-t2cm:2deg:2cm \
  --residual e5deg-t5cm:5deg:5cm \
  --residual e3deg-t1cm:3deg:1cm
```

New output layout is self-describing:

```text
<output-root>/<campaign-id>/<method-id>/<viewpoint-id>/<calibration-id>/results_<task>.json
```

For example, G7 has viewpoint ID `g7_fully_ood`; a figure caption should say
“G7 fully OOD | ext-r5deg-t5cm-v01 calibration.” Existing historical folders
are archival artifacts: do not rename, move, or overwrite them. Plot scripts
may map canonical labels to those legacy paths, but captions and generated
tables use the canonical vocabulary.

## Checkpoint ladders and checkpoint selection

**Checkpoints are not selected by SBRS.** The SBRS ladder (6 tasks × 20,
seen group + medium bias + 3deg/1cm) is in-distribution, saturates by about
100k and moves ±4–5 points between neighbouring checkpoints, while
miscalibration robustness keeps climbing with training (exp 3, week of 6 Oct:
IDLT 60/60 48 → 72 from 50k to 200k, SBRS flat). Report a model at its
**final checkpoint**, or at the best checkpoint of the **robustness ladder**.

Standard ladders for every new video experiment (`checkpoint_evals`, see
[checkpoint-triggered-evals.md](checkpoint-triggered-evals.md)):

| Recipe | What | Cadence | Cells / checkpoint |
|---|---|---|---|
| `robustness_ladder_ema_v01.json` | IDLT task-heldout `calibrated` + `unknown-e60deg-t60cm`, 13 tasks × 20 rollouts, K=5 history, EMA | every 20k from 100k | 26 (two arrays of 13) |
| `sbrs_sanity_ladder_ema_v01.json` | SBRS 3deg/1cm, 6 tasks × 20, EMA; a sanity check only | every 50k | 6 |

`robustness_ladder_raw_v01.json` is the raw-weight twin for runs trained
without EMA. Both robustness recipes share protocol id `robustness_ladder_v01`
and write to `/grogu/datasets/hbhatia/3dfa_online_eval_20rollouts/robustness_ladder_v01_<method>_s<step>/`,
so 20-rollout ladder points never mix with the 100-rollout IDLT campaign
folders. Put the weights in the `method_id` (e.g. `exp3_ema`). A 20-rollout
point is a ladder reading; numbers that are cited come from the full
100-rollout IDLT campaigns.

## Launching

Print a resolved cell without running it:

```bash
python scripts/eval/run_eval_plan.py \
  --plan instructions/eval_plans/g7_external_calibration.json \
  --cell-index 0 --print-command
```

Run it on Slurm (the supplied G7 plan has 104 cells):

```bash
sbatch --array=0-103 scripts/eval/online_eval_plan.slurm \
  instructions/eval_plans/g7_external_calibration.json
```

SBRS v01 has 48 cells (2 methods × 6 tasks × 4 residual conditions):

```bash
sbatch --job-name=SBRS_v01 --array=0-47 \
  --output=/home/harshilb/3dfa_unified/logs/eval/SBRS_v01_%A_%a.out \
  --error=/home/harshilb/3dfa_unified/logs/eval/SBRS_v01_%A_%a.err \
  scripts/eval/online_eval_plan.slurm \
  instructions/eval_plans/seen_base_residual_sweep_v01.json
```

The legacy shell launchers remain for reproducibility only. New experiments
should add a registry entry and a campaign plan, not another condition-specific
shell script.
