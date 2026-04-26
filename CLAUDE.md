# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Repo Does

3D Flow Matching Actor (3DFA) is a robotic manipulation policy that predicts end-effector trajectories using rectified flow denoising in 3D space. Given RGB-D images from multiple cameras, language instructions, and proprioception, the model outputs keypose trajectories for RLBench / CoppeliaSim tasks (bimanual PerAct2 benchmark). Supports an "orbital camera" variant for camera-position generalization.

## Docs

- [Commands](docs/commands.md) — training, evaluation, data conversion, environment setup
- [Architecture](docs/architecture.md) — model pipeline, key modules, non-obvious design details
- [Config](docs/config.md) — Hydra config system and important flags
- [Data](docs/data.md) — zarr format, dataset classes, preprocessing, data generation
