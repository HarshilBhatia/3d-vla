# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Repo Does

3D Flow Matching Actor (3DFA) is a robotic manipulation policy that predicts end-effector trajectories using rectified flow denoising in 3D space. Given RGB-D images from multiple cameras, language instructions, and proprioception, the model outputs keypose trajectories for RLBench / CoppeliaSim tasks (bimanual PerAct2 benchmark). Supports an "orbital camera" variant for camera-position generalization.

## Docs

`docs/` is the canonical state of the codebase, split by how fast it changes.
[docs/README.md](docs/README.md) indexes the whole tree; start there.

- [`docs/handbook/`](docs/handbook/) — how the system works: `model/`, `data/`,
  `eval/`, `codebase/`, plus `setup.md` and `commands.md`. The four you will
  need most: [commands](docs/handbook/commands.md),
  [architecture](docs/handbook/model/architecture.md),
  [config](docs/handbook/model/config.md),
  [terminology](docs/handbook/model/terminology.md).
- [`docs/research/`](docs/research/) — what we are finding out:
  `experiments/<week>/` (one page per experiment),
  `weekly_logs/YYYY-MM-DD.md` (per-week tracker, named by the week's end date),
  `results/` (settled numbers), `archive/` (**not** canonical).

Figures live with whatever owns them — `handbook/eval/figures/` for geometry
illustrations, `research/results/figures/` for cross-model plots,
`research/experiments/<week>/figures/` for that week's. Each `figures/` has a
README naming the script that makes every file.

A new page goes where its change rate says: a fact about the system →
`handbook/`; something you are testing → `research/experiments/<week>/`; a
number you would cite → `research/results/`; what happened this week →
`research/weekly_logs/`. A figure goes in the `figures/` folder of whichever of
those owns it.

## Experiments

**Every new experiment gets its own `config/experiment/*.yaml`.** Never launch a
run by piling CLI overrides onto an existing experiment config — the config file
is the record of what was run. Add a matching `ARM` to the launcher script.

## Comments

**One line maximum.** Never write multi-line comments or docstrings. Rationale,
history and caveats go in the commit message or `docs/`, never in the source.

## Writing Eval Scripts

Use existing scripts in `scripts/eval/` as templates — don't invent values.

**Rules:**
1. **Only override what differs from `config/config.yaml` defaults.** Read the defaults first; if a value matches, omit it. Exception: always include `bimanual` explicitly — it changes the model architecture and must be unambiguous.
2. **Don't hardcode model architecture args** (e.g. `fps_subsampling_factor`, `num_vis_instr_attn_layers`, `head_positional_encoding`). These must match the checkpoint — let the caller pass them via `"$@"` if needed.
3. **Pass `"$@"` at the end** so the caller can override task, checkpoint, output_file, and any arch args.
4. **Ask before assuming** checkpoint path, data dir, and task list — don't infer from other scripts.
