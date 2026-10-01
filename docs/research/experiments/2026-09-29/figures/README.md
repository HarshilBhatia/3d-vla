# Figures: week of 22–29 Sept 2026

Every figure from this week's history ablations and EMA fine-tunes, regenerated from the raw eval JSONs.
Run the scripts from the repo root; the `*_ema_*` and per-model scripts import `scripts/plot_ablation_per_task.py`, so prefix them with `PYTHONPATH=scripts`.

They write to the **current week's** figures directory by default (`utils.plotting.week_figures_dir()`, weeks ending Tuesday). Pass `--output-dir` to send them elsewhere, e.g. when regenerating an earlier week's panels.

## Original vs EMA (the EMA fine-tunes, [../ema-finetunes.md](../ema-finetunes.md))

| Figure | Shows | Script |
|---|---|---|
| `ema_idlt_mean.{png,csv}` | IDLT task-mean vs miscalibration, best EMA checkpoints of VideoPooled, random history, no delta_M, A and Base | `scripts/plot_ema_idlt_mean.py` |
| `ema_vs_original.png` | Task-mean success per model, original (solid) vs best-EMA checkpoint (dashed): IDLT held-out, G7 translation, G7 rotation | `scripts/plot_ema_vs_original.py` |
| `ema_per_model_<model>.png` | One figure per model (`vidpooled`, `randhist`, `noviewalign`, `regtest_a`, `regtest_c`, `base`): every IDLT and G7 task, original vs EMA | `scripts/plot_ema_per_model.py` |
| `ema_vs_raw_over_steps.png` | SBRS task-mean vs fine-tune step, raw (dashed grey) vs EMA (solid blue) overlaid, one panel per run | `scripts/plot_ema_vs_raw_steps.py` |
| `ema_vs_raw_over_steps_per_task.png` | The same overlay per task: rows = runs, columns = task mean + 6 SBRS tasks | `scripts/plot_ema_vs_raw_steps.py` |
| `ema20k_ladders_per_task.{png,csv}` | SBRS per task over the 20k fine-tunes, raw vs EMA weights; step 0 = starting checkpoint | `scripts/plot_ema_ladders.py` |

## Robustness per task, all six models

| Figure | Shows | Script |
|---|---|---|
| `ablation_idlt_per_task.png` | IDLT held-out, calibrated → 60/60, original checkpoints | `scripts/plot_ablation_per_task.py` |
| `ablation_g7_translation_per_task.png`, `ablation_g7_rotation_per_task.png` | G7 extremes, original checkpoints | same |
| `ablation_*_per_task_ema.png` | The same three, best-EMA checkpoints. The five non-random-history models have only calibrated and 10/10 at low noise | same |
| `ablation_per_task.csv` | Rows behind all six | same |

## Training-time SBRS ladders

| Figure | Shows | Script |
|---|---|---|
| `ladder_curves_sbrs.{png,csv}` | Each task vs total training steps, all runs | `scripts/plot_ladder_curves.py` |
| `ladder_per_run_sbrs.png` | One panel per run: 6 tasks + mean per checkpoint | same |

## `viewpoint-sweep/` — the memorisation test

Supports [../g7-viewpoint-sweep.md](../g7-viewpoint-sweep.md). 78 MB of renders,
**not committed** (gitignored).

| Path | Shows | Script |
|---|---|---|
| `captures/<task>/<group>.{npz,png}` | Rendered views at each of the 11 sweep poses | `scripts/capture_g7_sweep_views.py` |
| `g7_viewpoint_sweep.rrd` | All sweep poses with renders | `scripts/viz_g7_sweep_rerun.py` |
| `g7_vs_train_groups.rrd` | G7 against G1–G6 only | `scripts/viz_g7_sweep_rerun.py --no-sweep` |
