# Handover sanity: dense keypose path vs fixed-time chunk (exp 8a, 8b)

**Status:** data building and training queued 12 Oct
**Week introduced:** [7–13 Oct](../../weekly_logs/2026-10-13.md)

## Why

The dense 4-task data still anchors every target at the next keypose: each row predicts 8 points resampled between now and the next keypose, so a row just before a keypose predicts nearly the same pose 8 times, and the time spacing of the target changes from row to row. Before scaling up, compare that against a keypose-free target on one task with the simplest model. Exp 7a–7f were cancelled for this.

## Arms

Base (no history, K = 1, no δM), handover_item only, absolute actions, from scratch, EMA, global batch 128 on 2 GPUs, 50k steps, `keypose_only: false` (all 8 action steps are learned).

| Exp | Target (8 steps) | Proprio history | Zarr | Config | ARM |
|---|---|---|---|---|---|
| **8a** | 8 points resampled from t to the next keypose (the 8th is the keypose) | the last two keyframes, then t (eval queries once per keypose) | `3dfa_dense/handover_keypose_path` | `peract2_orbital_handover_dense_base_keypose_path` | `exp8a_handover_keypose_path` |
| **8b** | the next 8 recorded sim steps, t+1 … t+8, crossing keyposes; held at the last frame past the episode end | t−16, t−8, t (eval re-queries every 8 steps) | `3dfa_dense/handover_chunk8` | `peract2_orbital_handover_dense_base_chunk8` | `exp8b_handover_chunk8` |

Both zarrs are rebuilt from the raw episodes of the [dense 4-task collection](dense-4task-dataset.md) with new converter options (`--target keypose_path|chunk`, `--chunk-stride`, `--proprio keypose|timestep`, `--proprio-stride`), so neither run needs the `train_tasks` filter.

## Jobs

| | Conversion | Training (after conversion) |
|---|---|---|
| 8a | 4027416 | 4027418 |
| 8b | 4027417 | 4027419 |

## Evaluation

Not set up yet. 8a can run with the planner loop, which sends each of the 8 poses to the motion planner in turn. 8b needs a low-level controller that tracks the 8 poses step by step and re-queries every 8 steps; the planner is the wrong tool for 1-step moves. Until then compare the offline validation metrics (position and rotation error on the val split).
