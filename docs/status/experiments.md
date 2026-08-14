# Experiment Log

| Job ID   | Cluster | Job Name                 | Experiment Config                      | Notes                                              | Miscal                                      | Status             |
|----------|---------|--------------------------|----------------------------------------|----------------------------------------------------|---------------------------------------------|--------------------|
| 18343783 | delta   | multicam_default         | —                                      | Baseline 3DFA, no extrinsics prediction            | fixed medium per-group + randnoise (≤3°, ≤1cm) | DONE (ckpt @ 192k) |
| 18343790 | delta   | multicam_deltaM_med      | orb_deltaM_full_fixed_medium_randnoise | DeltaM full (DxD), fixed medium miscal + randnoise | fixed medium per-group + randnoise (≤3°, ≤1cm) | DONE (ckpt @ 160k) |
| 18346455 | delta   | multicam_deltaM_6x6_med  | orb_deltaM_fixed_medium_randnoise      | DeltaM 6x6, fixed medium miscal + randnoise        | fixed medium per-group + randnoise (≤3°, ≤1cm) | DONE (ckpt @ 160k) |

## Online Eval — open_drawer, orbital_miscal_noise_level=medium (20 May 2026)

Slurm job 3904050 (+ resubmit 3904056 for default/G5). Script: `scripts/eval/online_eval_delta_miscal_G1G5.slurm`.  
All 3 checkpoints at `train_logs/delta/<run>/interm_step_160000.pth`. camera_groups=GT, spawn_camera_group=G1/G5.

| Run        | extrinsics_prediction_mode | predict_extrinsics | G1   | G5   |
|------------|----------------------------|--------------------|------|------|
| default    | —                          | False              | 0.71 | 0.52 |
| deltaM     | delta_m_full (D×D)         | True               | 0.71 | 0.60 |
| deltaM_6x6 | delta_m (6×6)              | True               | 0.84 | 0.70 |


## Online Eval — Full PerAct Dataset (16 tasks), G7 cameras, GT demos (23 May 2026)

Jobs 3905363–3905534. Script: `scripts/eval/online_eval_full_dataset.slurm`.  
Checkpoints: `train_logs/full_dataset/<run>/last.pth`. dataset=OrbitalWrist, data_dir=`/grogu/user/harshilb/low_dim_demos` (100 GT demos/task), `spawn_camera_group=G7`, `camera_groups=GT`. `*` = inferred from partial progress file.

| Task | default_3dfa | fixmed_rn | deltaM_EEF |
|------|:---:|:---:|:---:|
| insert_onto_square_peg | 0.020 | 0.000 | 0.000 |
| light_bulb_in | 0.000 | 0.010 | 0.000* |
| meat_off_grill | 0.750 | 0.070 | 0.000 |
| open_drawer | 0.000 | 0.390 | 0.600 |
| place_cups | 0.025* | 0.000 | 0.000* |
| place_shape_in_shape_sorter | — | — | — |
| place_wine_at_rack_location | 0.092* | 0.050 | 0.000 |
| put_groceries_in_cupboard | 0.060 | 0.000* | 0.007* |
| put_item_in_drawer | 0.000 | 0.000 | 0.000* |
| put_money_in_safe | 0.520 | 0.570 | 0.740 |
| reach_and_drag | 0.180 | 0.000 | 0.000 |
| slide_block_to_color_target | 0.510 | 0.480 | 0.500 |
| stack_blocks | 0.030 | 0.010 | 0.000 |
| stack_cups | 0.000 | 0.000 | 0.000 |
| sweep_to_dustpan_of_size | 0.400 | 0.670 | 0.710 |
| turn_tap | 0.660 | 0.730 | 0.860 |
| **MEAN** | **0.216** | **0.199** | **0.228** |

**Conclusion:** All three models are trained on the full PerAct single-arm dataset and evaluated on a novel G7 camera (unseen during training). `deltaM_EEF` edges out the others (0.228 mean), driven by strong performance on `turn_tap` (0.86), `open_drawer` (0.60), `put_money_in_safe` (0.74), and `sweep_to_dustpan` (0.71) — suggesting the delta-M extrinsics prediction helps with camera generalization. `fixmed_randnoise` (trained with miscalibration noise) is weakest overall (0.199), possibly because the noise schedule hurts performance at G7 which is a clean but very different viewpoint. Most manipulation-heavy tasks (`stack_cups`, `stack_blocks`, `light_bulb_in`, `put_item_in_drawer`) score near 0 across all models.

## Online Eval — turn_tap Miscalibration Noise Sweep (24 May 2026)

Jobs 3905563 / 3905588. Script: `scripts/eval/online_eval_full_dataset_miscal_sweep.slurm`.  
Same setup as full eval above (G7, GT demos) but with paired rot+trans noise applied at eval time (`miscal_rot_level` + `miscal_trans_level`). 0deg row is from the full eval above. `*` = partial result.  
**Status:** `fixmed_rn` and `deltaM_EEF` fully complete. `default_3dfa` noise 2/5/10/15deg paused mid-run (progress saved); noise 20deg complete. Resume with `sbatch --array=0-3 scripts/eval/online_eval_full_dataset_miscal_sweep.slurm`.

rephrase as calibrated / no-miscal 3DFA*. 


| Noise | default_3dfa | fixmed_rn | deltaM_EEF |
|------:|:---:|:---:|:---:|
| 0deg | 0.660 | 0.730 | **0.860** |
| 2deg + 2cm| 0.907* | 0.790 | **0.860** |
| 5deg + 5cm| 0.429* | 0.770 | **0.860** |
| 10deg + 10 cm| 0.132* | 0.770 | 0.793* |
| 15deg + 15 cm| 0.000* | 0.710 | **0.770** |

**Conclusion:** `default_3dfa` collapses sharply with noise (0.91→0.00 by 15°), while `fixmed_rn` and `deltaM_EEF` are far more robust, both staying above 0.6 at 20°. `deltaM_EEF` is consistently the strongest, flat from 0–5° before degrading gracefully. Training with miscalibration noise (`fixmed_rn`) gives the flattest curve overall. Notable: `default_3dfa` ticks up at 2° vs 0° — likely variance, not a real effect.

## Online Eval — default_3dfa, Task-Specific Camera Groups, No Noise (24 May 2026)

Val set: `data_dir=/grogu/user/harshilb/low_dim_demos`, `camera_groups=GT`, `spawn_camera_group` per task.  
Train set: `data_dir=/grogu/datasets/hbhatia/full_rollouts`, `camera_groups` = task group (demos stored per group, no GT).  
Checkpoint: `train_logs/full_dataset/default_3dfa/last.pth`. 20 demos each. Scripts: `online_eval_default_taskcam_nonoise.slurm`, `online_eval_default_taskcam_nonoise_trainset.slurm`.

| Task (group) | Val SR | Train SR |
|---|:---:|:---:|
| light_bulb_in (G4) | 0.000 | 0.000 |
| place_cups (G1) | 0.000 | 0.100 |
| stack_cups (G4) | 0.000 | 0.000 |

**Conclusion:** Near-zero performance across all 3 tasks at task-specific camera groups, even without noise. Only glimmer: train place_cups at 10% SR. Suggests default_3dfa does not generalize to these off-G7 camera positions.

## Online Eval — default_3dfa, Full Dataset, Train-Set Camera Groups, No Noise (24 May 2026)

All 16 tasks evaluated on training camera groups (no noise, no GT extrinsics swap).  
Data: `/grogu/datasets/hbhatia/full_rollouts` + `/grogu/user/harshilb/full_rollouts_merged`. 20 demos each.  
Checkpoint: `train_logs/full_dataset/default_3dfa/last.pth`. Scripts: `online_eval_default_trainset_remaining.slurm`, `online_eval_default_trainset_missed.slurm`.  
`—` = did not run (hangs); `*` = pending at time of writing.

| Task (group) | Train SR |
|---|:---:|
| turn_tap (G1) | **1.000** |
| meat_off_grill (G1) | **0.750** |
| slide_block_to_color_target (G2) | **0.600** |
| put_money_in_safe (G1) | **0.550** |
| sweep_to_dustpan_of_size (G1) | **0.350** |
| reach_and_drag (G1) | **0.300** |
| stack_blocks (G3) | **0.200** |
| put_groceries_in_cupboard (G4) | **0.150** |
| place_cups (G1) | **0.100** |
| place_wine_at_rack_location (G2) | **0.050** |
| insert_onto_square_peg (G3) | 0.000 |
| light_bulb_in (G4) | 0.000 |
| stack_cups (G4) | 0.000 |
| open_drawer (G1) | 0.000 |
| put_item_in_drawer (G1) | 0.000 |
| place_shape_in_shape_sorter | — |
| **Mean (15 tasks)** | **0.267** |

**Conclusion:** Even on training camera groups with no noise, default_3dfa only achieves 0.27 mean SR. Strong on "simple" tasks (turn_tap 1.0, meat_off_grill 0.75, slide_block 0.6) but fails on manipulation-heavy ones (light_bulb_in, stack_cups, insert_onto_square_peg, open_drawer, put_item_in_drawer all 0). Comparison to G7 eval (0.216 mean) shows modest improvement on train cameras — the model has some camera-position sensitivity but the bigger bottleneck is task difficulty.

babel - 3dfa (base ) (no miscal)


Current status
1. running high miscal experiments (hoping 3DFA fails? lol.)
2. running pi_0.5 on mini data.
TODO here: setup online eval for pi_0.5 lol. Ugh this will be so much effort....

Main dataset
training
1. default_3dfa.
2. 3dfa + noise schedule.
3. ours + noise_schedule.
