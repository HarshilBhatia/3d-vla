# PerAct2 Extrinsic-Miscalibration Sweep

Checkpoint: `/grogu/datasets/hbhatia/3dfa_egress_extracted/3dfa_egress_20260820/checkpoints/peract2_orbital_nhist3_b200_best.pth`  
Model: clean-trained 3D PerAct2 baseline; no deltaM/extrinsics-prediction head.  
Evaluation: 13 bimanual tasks, 10 rollouts/task. Miscalibration changed only the extrinsics used for depth-to-point-cloud reconstruction; RGB/depth pixels were unchanged.

| Condition | All cameras clean | External/orbital-only miscal | Wrist-only miscal |
|---|---:|---:|---:|
| No noise | 0.669 | — | — |
| 2° + 2 cm | — | 0.708 | 0.362 |
| 5° + 5 cm | — | 0.692 | 0.162 |
| 10° + 10 cm | — | 0.615 | 0.115 |

The no-noise baseline used the standard per-task held-out camera group with all cameras clean. The subset sweeps used the same camera groups and checkpoint, but miscalibrated only the indicated cameras. The external-only values therefore should not be interpreted as noise improving performance over 0.669; they are different perturbation conditions with rollout variance.

## Per-task success rates

| Task | Clean, all cameras | External 2°+2 cm | Wrist 2°+2 cm | External 5°+5 cm | Wrist 5°+5 cm | External 10°+10 cm | Wrist 10°+10 cm |
|---|---:|---:|---:|---:|---:|---:|---:|
| push_box | 0.9 | 0.9 | 0.9 | 1.0 | 0.6 | 0.8 | 1.0 |
| lift_ball | 1.0 | 1.0 | 1.0 | 1.0 | 0.1 | 1.0 | 0.5 |
| dual_push_buttons | 0.8 | 0.9 | 0.1 | 0.9 | 0.0 | 0.9 | 0.0 |
| pick_plate | 0.4 | 0.6 | 0.0 | 0.5 | 0.0 | 0.6 | 0.0 |
| put_item_in_drawer | 0.2 | 0.6 | 0.0 | 0.8 | 0.0 | 0.4 | 0.0 |
| put_bottle_in_fridge | 1.0 | 0.8 | 0.4 | 0.6 | 0.3 | 0.9 | 0.0 |
| handover_item | 0.8 | 1.0 | 0.3 | 1.0 | 0.2 | 0.9 | 0.0 |
| pick_laptop | 0.4 | 0.4 | 0.0 | 0.4 | 0.0 | 0.0 | 0.0 |
| straighten_rope | 0.2 | 0.0 | 0.3 | 0.1 | 0.0 | 0.0 | 0.0 |
| sweep_to_dustpan | 0.2 | 0.2 | 0.0 | 0.1 | 0.0 | 0.1 | 0.0 |
| lift_tray | 0.9 | 1.0 | 1.0 | 0.8 | 0.6 | 1.0 | 0.0 |
| handover_item_easy | 1.0 | 1.0 | 0.1 | 1.0 | 0.3 | 1.0 | 0.0 |
| take_tray_out_of_oven | 0.9 | 0.8 | 0.6 | 0.8 | 0.0 | 0.4 | 0.0 |

Camera indices: external/orbital = `[0,1]`; wrist = `[2,3]`. The clean baseline has no selected subset because no perturbation was applied.

## Provenance

| Condition | Slurm jobs |
|---|---|
| Clean, all cameras | `3935835` |
| External 2°+2 cm | `3935802` |
| Wrist 2°+2 cm | `3935803` |
| External 5°+5 cm | `3935658` |
| Wrist 5°+5 cm | `3935659`, wrist retry `3935703` |
| External 10°+10 cm | `3935833` |
| Wrist 10°+10 cm | `3935834` |

Every listed job completed all 13 task evaluations. Each cell is based on 10 rollout attempts, so the per-task resolution is 0.1 and the aggregate is the unweighted mean of the 13 task success rates.

## G7 clean-camera evaluation

G7 is a fully unseen bimanual camera group with two external cameras (`L_el47_0`, `R_el47_0`). It was not used in training or the normal G1–G6 task-group rotation. No miscalibration or test-time noise was applied.

| Model | Mean success rate |
|---|---:|
| R1a: miscal-trained baseline | 0.646 |
| R1b: miscal-trained + deltaM | 0.554 |
| R1c: miscal-trained + deltaM + EE-aux | 0.531 |

| Task | R1a | R1b | R1c |
|---|---:|---:|---:|
| push_box | 0.6 | 0.9 | 1.0 |
| lift_ball | 1.0 | 1.0 | 1.0 |
| dual_push_buttons | 0.9 | 0.2 | 0.7 |
| pick_plate | 0.3 | 0.2 | 0.0 |
| put_item_in_drawer | 0.1 | 0.5 | 0.1 |
| put_bottle_in_fridge | 0.4 | 0.4 | 0.2 |
| handover_item | 0.9 | 0.8 | 0.4 |
| pick_laptop | 0.3 | 0.5 | 0.6 |
| straighten_rope | 0.5 | 0.2 | 0.2 |
| sweep_to_dustpan | 1.0 | 0.8 | 1.0 |
| lift_tray | 1.0 | 0.8 | 0.9 |
| handover_item_easy | 1.0 | 0.7 | 0.6 |
| take_tray_out_of_oven | 0.4 | 0.2 | 0.2 |

G7 jobs: R1a `3935888` + retries `3935928/3935956`; R1b `3935889`; R1c `3935890` + retries `3935929/3935957`.

### G7 random-noise sweep

Random test-time extrinsic noise only; no fixed miscalibration base. Noise was applied to the G7 external cameras `[0,1]`.

| Noise | R1a | R1b | R1c |
|---|---:|---:|---:|
| 0°+0 cm | 0.646 | 0.554 | 0.531 |
| 2°+2 cm | 0.631 | 0.515 | 0.554 |
| 5°+5 cm | 0.700 | 0.554 | 0.638 |
| 10°+10 cm | 0.700 | 0.546 | 0.592 |
| 15°+15 cm | 0.400 | 0.146 | 0.254 |

All 156 sweep tasks completed successfully: 3 checkpoints × 4 noise levels × 13 tasks, with 10 rollouts/task. Jobs: R1a `3935993–3935996`, R1b `3935997–3936000`, R1c `3936001–3936004`.

Source jobs: clean `3935835`; 2°+2 cm `3935802/3935803`; 5°+5 cm `3935658/3935659` plus wrist retry `3935703`; 10°+10 cm `3935833/3935834`. Results were reconstructed from Slurm logs because the original output paths were reused across noise levels. The eval script now includes the condition in each output path.

## Newer retrained miscalibration-only Base

The clean-trained checkpoint above is historical. The newer retrained Base
checkpoint is:

```text
train_logs/PerAct2/peract2_orbital_miscal_only_base/best.pth
```

Training used a persistent per-group medium calibration bias but no sampled
perturbation noise (`perturbation_noise_rot_deg=null`,
`perturbation_noise_trans_m=null`). It was evaluated with fixed external-camera
residuals at 5/5, 10/10, 15/15, and 20/20 degrees/cm. Each condition is 13
tasks × 100 rollouts/task.

| Residual noise | Task-seen group | Task-heldout group |
|---|---:|---:|
| Clean residual | 73.0% | 72.2% |
| 5° / 5 cm | 73.1% | 71.7% |
| 10° / 10 cm | 72.1% | 69.7% |
| 15° / 15 cm | 67.7% | 66.9% |
| 20° / 20 cm | 66.4% | 66.5% |

Raw result roots:

```text
/grogu/datasets/hbhatia/3dfa_online_eval_100rollouts/
  miscal_only_base_task_seen_group_v01/
  miscal_only_base_task_heldout_group_v01/
```

The associated plans are
`instructions/eval_plans/miscal_only_base_task_seen_group_v01.json` and
`instructions/eval_plans/miscal_only_base_task_heldout_group_v01.json`.
