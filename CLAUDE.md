# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Repo Does

3D Flow Matching Actor (3DFA) is a robotic manipulation policy that predicts end-effector trajectories using rectified flow denoising in 3D space. Given RGB-D images from multiple cameras, language instructions, and proprioception, the model outputs keypose trajectories for RLBench / CoppeliaSim tasks (bimanual PerAct2 benchmark). Supports an "orbital camera" variant for camera-position generalization.

## Docs

- [Commands](docs/commands.md) — training, evaluation, data conversion, environment setup
- [Architecture](docs/architecture.md) — model pipeline, key modules, non-obvious design details
- [Config](docs/config.md) — Hydra config system and important flags
- [Data](docs/data.md) — zarr format, dataset classes, preprocessing, data generation

## Writing Eval Scripts

Use existing scripts in `scripts/eval/` as templates — don't invent values.

**Rules:**
1. **Only override what differs from `config/config.yaml` defaults.** Read the defaults first; if a value matches, omit it. Exception: always include `bimanual` explicitly — it changes the model architecture and must be unambiguous.
2. **Don't hardcode model architecture args** (e.g. `fps_subsampling_factor`, `num_vis_instr_attn_layers`, `sa_blocks_use_rope`). These must match the checkpoint — let the caller pass them via `"$@"` if needed.
3. **Pass `"$@"` at the end** so the caller can override task, checkpoint, output_file, and any arch args.
4. **Ask before assuming** checkpoint path, data dir, and task list — don't infer from other scripts.
