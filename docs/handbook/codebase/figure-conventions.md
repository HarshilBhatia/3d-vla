# Figure library

Use `utils.plotting` for all experiment figures. It provides:

- a shared visual theme and semantic arm colors;
- strict loading of final per-task result JSONs;
- task-set validation before aggregation;
- task-weighted mean-success aggregation;
- protocol-note and axis-style helpers; and
- standard high-resolution PNG plus editable SVG export.

Figure scripts should only declare experiment-specific arms, result paths,
conditions, and provenance. They should not reimplement JSON validation or the
global theme.

## Conventions

- `Clean`: no test miscalibration.
- `Test-N-Ext`: newly sampled external-camera test miscalibration at level N,
  fixed for that evaluation condition.
- `Train-Base-Ext+Jitter-All`: persistent external base miscalibration plus
  per-sample all-camera training jitter.
- A non-matched baseline must be visually and textually labeled as such; do not
  place it on a matched curve without a clear caveat.

## Template

```python
from utils.plotting import (
    COLORS, add_protocol_note, assert_shared_tasks, configure_theme,
    load_task_means, mean_success, save_figure, style_axis,
)

configure_theme()
task_results = {name: load_task_means(path) for name, path in paths.items()}
assert_shared_tasks(task_results)
means = {name: mean_success(values) for name, values in task_results.items()}
```

Run a figure script using the 3DFA environment, for example:

```bash
/home/harshilb/miniconda3/envs/3dfa/bin/python scripts/plot_g7_eval_miscal_comparison.py
```

## Where figures are written

Figures live with whatever owns them: `docs/handbook/eval/figures/` for
model-independent geometry, `docs/research/results/figures/` for cross-model
plots, and `docs/research/experiments/<week>/figures/` for a week's own panels.
Each of those has a README naming the script behind every file.

Week-scoped scripts default to the current week via
`utils.plotting.week_figures_dir()` and take `--output-dir` to override, so no
script carries a hardcoded date:

```bash
PYTHONPATH=scripts python scripts/plot_ladder_curves.py
PYTHONPATH=scripts python scripts/plot_ladder_curves.py --output-dir docs/research/experiments/2026-10-06/figures
```
