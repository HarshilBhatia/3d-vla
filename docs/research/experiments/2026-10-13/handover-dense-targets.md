# Handover sanity: dense keypose path vs fixed-time chunk (exp 8a, 8b)

**Status:** the v01 rollouts below are INVALID (eval sampled 1-step trajectories, `prediction_len` defaulted to 1); 8c/8d re-evaluated with `prediction_len: 8` (v02, running 13 Oct). The step-embedding bug is real and fixed.
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

## Results (50k, EMA, handover, 100 rollouts; videos 5)

| | Held-out G4 calibrated | Held-out G4 60/60 | Seen G1 calibrated | Val pos error (do-nothing baseline) |
|---|---|---|---|---|
| 8a keypose path, planner | 10% | 1% | 13% | 4.8 cm (8.9) |
| 8b chunk, IK, execute 8 | 0% | 0% | 0% | 2.0 cm (2.3) |
| 8b chunk, IK, execute 4 | ~0% | 0% | 0% | — |
| **8c** = 8a + step-embedding fix, planner | 0% | 0% | 0% | 2.2 cm |
| **8d** = 8b + fix, IK, execute 8 | 0% | 0% | 0% | 1.7 cm |
| **8d**, IK, execute 4 | 0% | 0% | 0% | — |
| *Pipeline check: exp 1b 200k (keypose, 13 tasks), same eval code* | | | *20/20* | |

## What went wrong (v01; points 2 and 3 retracted)

1. **Step-embedding bug** (fixed in `f371522`). The 1D sinusoidal position embedding of the trajectory tokens was tiled arm-major while the tokens are step-major, so token *i* got step *i mod T*: steps t and t+4 shared an embedding and the right arm's steps were mislabelled. Harmless for 1-step (keypose) targets, where the layouts coincide; fatal for 8-step bimanual targets. 8a could not fit its own training data (3.9 cm train error); 8c fits to 0.7 cm.
2. *Retracted (based on invalid 1-step rollouts):* **Keypose-path targets on dense rows stall at keyposes (8c).** About 20 rows before each keypose say "barely move"; the one row at the keypose says "go to the next one". Their inputs are nearly identical (keypose-cadence proprio has no velocity), and eval queries exactly at keyposes, so the policy hovers after the first move.
3. *Retracted (based on invalid 1-step rollouts):* **Fixed-time chunks learn the copycat shortcut (8d).** Proprio at t−16, t−8, t carries velocity, and "continue at the current velocity" predicts the next 8 steps well (val 1.7 cm vs 2.3 cm for holding still). From rest the policy barely moves: in the videos the arms leave the start pose only after about 320 steps, longer than a demo.

Also built along the way: an IK chunk controller with planner fallback (`eval_controller: ik`), `eval_execute_steps`, `eval_proprio_stride`, and a ground-truth replay check (`scripts/eval/replay_chunks_ik.py`; 2/3 replays succeed, 0.1–0.2 cm tracking).

## Next (proposed)

- 8d with only the current proprio state (no history), to remove the velocity shortcut.
- Offline: 8d's validation error with the scene blanked, to see whether the policy uses vision at all.

## Eval correction (13 Oct)

The eval Actioner builds the trajectory mask from `prediction_len`, which defaulted to 1, so every v01 rollout of an 8-step model sampled and executed a 1-step trajectory: 8b/8d moved one sim step per query, 8a/8c moved 1/8 of the way to the keypose per query. The stall (8c) and copycat (8d) explanations were drawn from those rollouts and are withdrawn. v02 plans (`exp8_stepfix_*_v02`, jobs 4028431–34) re-evaluate 8c/8d with `prediction_len: 8`.

## Exp 8e: the reference keypose-horizon setup

The upstream 3DFA repo trains PerAct2 keypose-only (`keypose_only=true`, 1-step actions); its keypose-horizon trajectory (paper) is only used for single-arm HiveFormer: keyframe rows only, the path from this keypose to the next as a 50-point cubic spline over xyz + Euler with nearest-neighbour gripper, last-3-keyframe proprio, `lv2_batch_size=4`, lr 1e-4, 100k, and every point sent to the motion planner at eval. The upstream step-embedding code has the same latent bimanual bug.

8e replicates that on handover keyframe rows (`3dfa_dense/handover_keypose_segment50`, 879 train / 96 val, built by `data/processing/convert_to_zarr/keypose_segment_from_dense.py`) with one deviation: Euler angles are unwrapped before the spline. Raw Euler splines sweep the long way round when an angle crosses ±π, which corrupted the middle of the path in 36% of rows (left arm, gripper pointing down); `--exact` rebuilds without the unwrap. Config `peract2_orbital_handover_keypose_segment50_base`, ARM `exp8e_handover_keypose_segment50`, training 4028444; 100k evals 4028445–48 with `prediction_len: 50`, every point (reference) and the last point only (`eval_execute_steps: -1`). Multi-step chunks now skip a pose the planner cannot reach instead of abandoning the chunk, as the reference does.

## Valid results: 8c/8d with `prediction_len: 8` (v02, 100 rollouts)

| | Seen G1 calibrated | Held-out G4 calibrated | Held-out G4 60/60 |
|---|---|---|---|
| 8c keypose path, planner on all 8 points | 80% | 77% | 1% |
| **8d chunk, IK, execute 8** | **96%** | **91%** | 1% |
| 8d chunk, IK, execute 4 | 87% | 87% | 0% |

Dense single-task Base policies work once the 8 steps are actually sampled: the fixed-time chunk executed in full is best, on par with the 13-task keypose models on this task (87–99%). Every variant collapses at 60/60, where keypose control A keeps 66%. A held-out G4 sweep from 10/10 to 50/50 (`exp8_sweep_*`, jobs 4028530–32) will show where the drop happens.
