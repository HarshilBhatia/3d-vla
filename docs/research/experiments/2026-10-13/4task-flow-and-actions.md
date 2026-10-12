# 4-task subset: x0 vs velocity, relative vs absolute actions (exp 7a–7d)

**Status:** relaunched 12 Oct on the dense 4-task data with 8-step targets (7a–7f); the first keypose-data launch was cancelled
**Week introduced:** [7–13 Oct](../../weekly_logs/2026-10-13.md)

## Why

Exp 6 needs about 48 h at 200k on 4 GPUs. A 4-task subset on 2 GPUs for 100k gives a faster read on exp 6's architecture and lets us test two training choices at the same time: what the flow head predicts, and which frame the action lives in. Exp 6 itself was cancelled after 6 minutes in favour of these runs.

## Subset

`train_tasks: [handover_item, take_tray_out_of_oven, dual_push_buttons, pick_laptop]`, all 4 cameras. `pick_laptop` replaced `sweep_to_dustpan` (92–98 everywhere): it is hard calibrated (30–52) and collapses under miscalibration (A 52 → 7 at 60/60). `dual_push_buttons` is flat at about 77 for every model, including A.

IDLT task-heldout at 200k for this week's 13-task models (calibrated / 60-60):

| Task | exp 3 | 1b | exp 2 | 4b | 5a | A |
|---|---|---|---|---|---|---|
| handover_item | 87 / 88 | 91 / 91 | 98 / 96 | 99 / 93 | 95 / 97 | 67 / 66 |
| take_tray_out_of_oven | 85 / 79 | 91 / 94 | 87 / 75 | 86 / 12 | 94 / 81 | 27 / 36 |
| dual_push_buttons | 77 / 78 | 79 / 79 | 78 / 74 | 76 / 76 | 81 / 67 | 77 / 78 |
| pick_laptop | 30 / 10 | 48 / 40 | 42 / 13 | 37 / 34 | 44 / 32 | 52 / 7 |

## Arms

From scratch, EMA, 100k, global batch 128 on 2 GPUs, medium-group training miscalibration on cameras 0/1.

| Exp | Architecture | Action | Flow target | Config | ARM |
|---|---|---|---|---|---|
| **7a** | exp 6 (full-patch 3D-RoPE random history, pre-LN + input skip, Direct δM) | absolute | velocity | `peract2_orbital_4task_fullpatch_direct_vel` | `exp7a_4task_vel` |
| **7b** | exp 6 | absolute | **x0** | `peract2_orbital_4task_fullpatch_direct_x0` | `exp7b_4task_x0` |
| **7c** | exp 6 | **relative** | velocity | `peract2_orbital_4task_fullpatch_direct_rel` | `exp7c_4task_rel` |
| **7d** | **Base** (no history, no δM, K = 1) | **relative** | velocity | `peract2_orbital_4task_base_rel` | `exp7d_4task_base_rel` |

Comparisons: 7a vs 7b (flow parameterisation), 7a vs 7c (action frame), 7c vs 7d (history + δM, both relative). There is no absolute-action Base on the subset, so 7d alone cannot say whether relative actions help Base.

## Code

- **x0 prediction** (`flow_prediction: x0`, default `velocity`; `base_denoise_actor.py`): the head outputs the clean pose x̂0, converted to v̂ = (z_t − x̂0) / max(t, 0.05). The L1 loss is still on v̂ against ε − x0, and Euler sampling uses v̂, so the last step lands exactly on x̂0. The gripper logit is unchanged.
- **Relative actions** (`relative_action: true`): the target is each arm's next keypose as a delta from that arm's current state, Δxyz and Δq = q_next · q_now⁻¹ (`to_relative_keypose`, `datasets/utils.py`). The workspace normaliser is fitted to the deltas (up to 0.63 m on these tasks). The model converts back to absolute poses at inference (`from_relative_keypose`), so online eval and the motion planner see absolute keyposes, and 3D RoPE still places the trajectory token at current xyz + Δ.
- Fixed three bugs in the old relative path, which was written for single-arm Euler PerAct: the anchor was the whole proprio history (indexed the chunk axis), the bimanual arm axis was treated as time, and validation added Euler angles to quaternions. The Euler path is unchanged for old models.
- Real-data check on the 4 tasks: relative actions converted back match the absolute ones to 1e-8 m.
- Tests: `tests/test_flow_x0_and_relative.py`.

## Evaluation

- Ladder every 20k from 40k, EMA, 100 rollouts per task: IDLT task-heldout calibrated and 60/60 on the 4 tasks. Recipe `robustness_ladder_ema_4task_v01.json`, or `_k1_v01` for 7d (K = 1).
- 100k suite (IDLT all levels + G7, 56 cells per run) from `scripts/eval/create_exp7_robustness_plans.py`, chained on training.

## Jobs

| | Training chain (2 GPUs) | 100k suite (on training dependency) |
|---|---|---|
| 7a | 4027118 → 19 (A6000) | 4027120–26 |
| 7b | 4027127 → 28 (A6000) | 4027129–35 |
| 7c | 4027136 → 37 (A6000) | 4027138–44 |
| 7d | 4027145 → 46 | 4027147–53 |

Smokes: 7b, 7c (batch 32) and 7d (batch 64) passed; 7a–7c at 64 per GPU ran out of memory on 24 GB cards, so they run on 48 GB A6000s.

## What we're checking

- Does x0 prediction change accuracy or robustness at 5 sampling steps?
- Do relative actions make the policy less sensitive to miscalibrated geometry (the action no longer has to be placed in world coordinates read from the miscalibrated cloud)?
- Is exp 6's architecture worth it on this subset: 7c vs 7d.

## Relaunch on dense data (12 Oct)

The keypose data has 7.2 keyframes per demo, so history was mostly padding. All runs now train on the [dense 4-task data](dense-4task-dataset.md) (67,782 rows) and learn the **8-step** path to the next keypose (`keypose_only: false`).

| | Model | Visual history | Action | Flow target | GPUs | Training chain | 100k suite |
|---|---|---|---|---|---|---|---|
| **7a** | exp 6 (full-patch 3D RoPE, pre-LN + skip, Direct δM) | K = 5, random earlier timesteps | absolute | velocity | 2× 24 GB+, batch 64, 120k | 4027372 → 73 | 4027374–80 |
| **7b** | exp 6 | K = 5 | absolute | x0 | 2× 24 GB+, batch 64, 120k | 4027381 → 82 | 4027383–89 |
| **7c** | exp 6 | K = 5 | relative | velocity | 2× 24 GB+, batch 64, 120k | 4027390 → 91 | 4027392–98 |
| **7d** | Base | K = 1 | relative | velocity | 2 GPUs | 4027236 → 37 | 4027238–44 |
| **7e** | Base | K = 1 | absolute | velocity | 2 GPUs | 4027245 → 46 | 4027247–53 |
| **7f** | exp 2 (pooled Direct δM) | K = 5, random earlier timesteps | absolute | velocity | 2× 24 GB+, batch 64, 120k | 4027399 → 4027400 | 4027401–07 |

Changes for the dense data:

- `visual_history_dense: true` (saved in the checkpoint): online eval keeps every motion-planner frame between keyposes in the random-history pool, through RLBench's per-step hook, so eval samples from the same kind of pool as training.
- Proprio history on dense rows comes from the earlier keyframe rows plus the current state, matching eval's once-per-keypose proprio (equal to the zarr's `proprioception_keypose` on 300/300 rows).
- Relative actions anchor every step at the current state (not chained), so any slice of the 8 steps stays valid.
- The diverse-chunk cache sampler is off (`CACHE_BATCHES=0`); its 8-record window would be 8 near-identical frames.
- The history runs (7a–7c, 7f) use global batch 64 (32 per GPU) for 120k so they fit any 24 GB pair; no A6000 pair was free and the only A6000Ada node is drained. 7d/7e were already running at batch 128 for 100k, so history vs Base also differs in batch.
- No `train_tasks`: the dense zarr holds only these 4 tasks, and the filter copies every kept row into RAM (52 GB), which hung the first smokes.
- Eval is not adapted to 8-step outputs yet: the rollout loop sends each of the 8 poses to the motion planner in turn. Fix before reading the ladders.
