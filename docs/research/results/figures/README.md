# Results figures

Cross-model, paper-facing figures. These span campaigns and checkpoints, so
they belong to the results, not to any single experiment. Week-scoped figures
live with their week under
[`../../experiments/`](../../experiments/).

Conventions for writing new figure scripts:
[`../../../handbook/codebase/figure-conventions.md`](../../../handbook/codebase/figure-conventions.md).
Each `.md` next to a plot is its caption and the exact table behind it.

## Cross-model comparisons

| Figure | Shows | Script |
|---|---|---|
| `g7_extreme_task_graphs.png` | Six-panel per-task view of all three G7 extreme campaigns. Embedded in [../README.md](../README.md) | `scripts/plot_g7_extreme_task_graphs.py` |
| `g7_clean_model_comparison.*` | OOD G7, clean extrinsics: clean-3D, R1a/b/c, 2D baseline | `scripts/plot_g7_clean_comparison.py` |
| `g7_eval_miscal_model_comparison.*` | The same arms across external miscalibration levels | `scripts/plot_g7_eval_miscal_comparison.py` |
| `g7_r1_eval_miscal_comparison.*` | R1a/R1b/R1c only, same conditions | `scripts/plot_g7_eval_miscal_comparison.py` |
| `g7_new_external_only_miscal.*` | External-only miscalibration plus the RoPE phase-bound trend | `scripts/plot_g7_new_external_only_miscal.py` |
| `new_external_only_miscal_online_eval.*` | The online-eval view of the same campaign | `scripts/plot_new_external_only_miscal.py` |
| `heldout_miscal_curves.png` | Held-out-group miscalibration curves | `scripts/plot_heldout_miscal_curves.py` |
| `peract2_external_wrist_miscal_50rollouts.png` | External-only vs wrist-only sweep on the clean-trained baseline | `scripts/plot_camera_sweep.py` |

## `task-gallery/` — what each task looks like

248 MB of renders, **not committed** (gitignored). Regenerate with
`scripts/eval/capture_g7_sweep_and_tasks.slurm`.

| Path | Shows |
|---|---|
| `task_sr.md` | Per-task success table across G7 and IDLT held-out |
| `<task>/{G1,G7}.png` | Still of every task from the training and OOD camera groups |
| `all_tasks_G1.png` | Contact sheet of all 13 tasks |
| `rollouts/<task>/<model>_<outcome>_<variation>_demo<k>.gif` | Four-camera rollout GIFs, 5 per task, Base 160k and VideoPooled 84k. See `rollouts/README.md` |
