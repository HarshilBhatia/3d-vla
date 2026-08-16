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

---

# August 2026 Campaign — FAR infra (B200 training, L40S eval)

Repo moved from grogu to FAR infra 11 Aug 2026. Training: sky managed jobs on ll-sea k8s (B200), staging on `/k8s-nfs/harsvbha/3dfa/`. Online eval: L40S:1 per task on AWS k8s (sky-us-east-1/-2), docker image `rfm-h-eval-job:hb-3dfa-peract2-20260811` (ECR), test seeds + results on `s3://far-research-internal/harsvbha/3dfa/eval/`. All wandb: `far-wandb/3dfa`.

## Training runs

| Job ID | Date | Run name | Config | Hardware | Status |
|---|---|---|---|---|---|
| 112602 | 11 Aug | peract2_base_b200 | fork, PerAct2 (HF zarr), siglip2, nhist=1, bs256, lr3e-4, 350k iters | B200:4 | DONE ~8h |
| 117608 | 12 Aug | peract2_base_nhist3_b200 | = 112602 + num_history=3 | B200:1 | DONE ~13h |
| 117772 | 12 Aug | peract2_orbital_b200 | orbital PerAct2 (13 tasks x 3 cam groups x 27 eps), nhist=1, 100k iters | B200:4 | CANCELLED @ ~13k (user; pending nhist decision) |
| 117879 | 12 Aug | upstream_peract2_repro | upstream code @ab70932, their recipe verbatim (CLIP RN50 frozen, bs64, lr1e-4, nhist=3), 350k | B200:1 | DONE ~18.5h (348k last ckpt) |
| 118960 | 12 Aug | peract2_base_nhist3_clip_b200 | = 117608 + backbone=clip (RN50+FPN) | B200:1 | DONE ~21h |

Batch-size probe (11 Aug, 1x B200, real NFS data): pipeline is dataloader-bound, not GPU-bound — throughput saturates ~950-980 samples/s/GPU from bs128 up (never OOMed through bs1024 = 41GB/183GB). num_workers is the dominant lever (8→16 workers ~2x at bs128). Chosen: bs256 global, 16 workers, lr 3e-4 (capped 3x linear scaling — a deliberate recipe change vs paper's bs32).

## Online eval — PerAct2 13 tasks, NUM_DEMOS=25/variation, max_tries=1

**CRITICAL BUG + CORRECTION (13-14 Aug):** all pre-fix fork evals ran with a train/eval sampler mismatch — `image_space_sampling: true` in config was never forwarded by the trainer (models trained with density FPS) but WAS overlaid from checkpoint config at eval (rollouts ran uniform sampling). Offline val couldn't see it (uses the trainer's model). Cost: ~9 pts (siglip2) to ~31 pts (clip; 32x32 grid = bigger sampler distribution shift). Fixed: trainer forwards the flag (`eb06b4c`), eval takes explicit `ISS` env (`39efdae`), config default flipped to false = density FPS = what was actually trained (`b933802`), shared construction helper + step-0 guardrail so this class of bug crashes instead of silently degrading (`5777b27`..`b933802`). Audit also found `use_learned_abs_pe` dropped (benign — defaults matched) and offline eval scripts dropping 8-9 flags.

Success rates (%), corrected (`_issfix`) where applicable:

| task | base nhist1 | nhist3 (old→fix) | nhist3+clip (old→fix) | upstream repro | released ckpt |
|---|---|---|---|---|---|
| push_box | 100 | 92→96 | 88→92 | 96 | 88 |
| lift_ball | 92 | 100→100 | 96→100 | 100 | 100 |
| dual_push_buttons | 35 | 82→85 | 61→92 | 90 | 93 |
| pick_plate | 20 | 84→80 | 68→36 | 76 | 68 |
| put_item_in_drawer | 75 | 69→89 | 35→91 | 89 | 93 |
| put_bottle_in_fridge | 4 | 64→76 | 52→84 | 76 | 88 |
| handover_item | 11 | 90→96 | 26→85 | 83 | 87 |
| pick_laptop | 0 | 24→60 | 48→76 | 76 | 48 |
| straighten_rope | 4 | 8→52 | 24→52 | 24 | 16 |
| sweep_to_dustpan | 80 | 88→100 | 20→100 | 24 | 100 |
| lift_tray | 96 | 92→92 | 96→100 | 76 | 92 |
| handover_item_easy | 68 | 96→100 | 20→76 | 100 | 84 |
| take_tray_out_of_oven | 0 | 100→80 | 40→92 | 88 | 88 |
| **MEAN** | **45.0** | **76.1→85.1** | **51.8→82.8** | **76.8** | **80.4** |

Eval campaigns: base 11 Aug (jobs 116805-116826), released ckpt 11-12 Aug (117323-117442, upstream code path + patches), nhist3 12 Aug (118915-118931), clip + repro 13 Aug (120514-120549), issfix falsification + re-eval 13-14 Aug (120580-120619).

**Conclusions**
1. `num_history=3` is the single biggest factor: 45.0 → 76.1 (pre-fix numbers, same recipe otherwise). Tasks the nhist=1 model scored ≤11 on (take_tray 0→100, handover_item 11→90, put_bottle 4→64) recovered dramatically — history disambiguates multi-phase/occluded states.
2. With the sampler bug fixed, **the fork beats the released checkpoint**: siglip2+nhist3 85.1, clip+nhist3 82.8, vs released 76.8-80.4. siglip2 ≥ clip on our fork; backbone choice is second-order.
3. Upstream recipe reproduces on our infra (76.8 vs released 80.4, within noise; per-task profiles match). sweep_to_dustpan repro=24 vs released=100 remains an unexplained single-task outlier (fails largely the same seeds as pre-fix fork+clip).
4. clip pick_plate 68→36 post-fix is the one unexplained regression (possibly variance; re-run with more demos if it matters).

## Orbital PerAct2 dataset (collected 11-12 Aug, local docker swarm)

13 bimanual tasks x 6 camera groups (G1-G6) x 30 eps = 2,340 episodes, 0 failures (2 pick_plate shards clobbered by a pre-flock race, recollected). 4 cams/episode (orbital pair + 2 over-shoulder). Train mapping: task i → groups [i, i+1, i+2] mod 6 (`instructions/peract2_orbital_task_group_mapping.json`), eval group = i+3, 10 rollouts/task protocol. G7 never collected (fully OOD option). Final zarr: 1053 train / 117 val eps, staged `/k8s-nfs/harsvbha/3dfa/data/orbital_peract2/`. Unused shards on local disk (`/local/home/harsvbha/3dfa_data/orbital_peract2/`). Orbital training relaunch pending — recipe decision: siglip2 + nhist=3.

## Orbital PerAct2 — training + camera-generalization eval (14-15 Aug 2026)

Job 120769 `peract2_orbital_nhist3_b200` (B200:4, ~5h20m, 2 recoveries, full 100k iters verified in ckpt). siglip2, nhist=3, bs256, lr3e-4, dataset=OrbitalPeract2. Note: the orbital zarr has `demo_id`, so nhist=3 enables the full visual-history path (3 stacked rgb/depth frames), unlike standard PerAct2 where it was proprio-only. First run under the workdir-upload pattern (provenance assertion in job log).

Eval (jobs hb-3dfa-orb-eval-*, 15 Aug): 10 rollouts/task/condition, in-domain cam = first train group, OOD cam = held-out group per `peract2_orbital_task_group_mapping.json`. Needed a new harness — no code path did bimanual + orbital spawning (`e042de3`, utils_with_orbital_bimanual_rlbench.py; also `num_demos_total` whole-task budget — per-variation num_demos would have run 460 eps on dual_push_buttons).

| task | in-domain | OOD | delta |
|---|---|---|---|
| push_box | 0.90 | 1.00 | +0.10 |
| lift_ball | 1.00 | 0.90 | -0.10 |
| dual_push_buttons | 1.00 | 0.90 | -0.10 |
| pick_plate | 0.20 | 0.20 | 0.00 |
| put_item_in_drawer | 0.80 | 0.50 | -0.30 |
| put_bottle_in_fridge | 0.90 | 0.90 | 0.00 |
| handover_item | 1.00 | 1.00 | 0.00 |
| pick_laptop | 0.60 | 0.30 | -0.30 |
| straighten_rope | 0.20 | 0.20 | 0.00 |
| sweep_to_dustpan | 0.70 | 0.60 | -0.10 |
| lift_tray | 1.00 | 1.00 | 0.00 |
| handover_item_easy | 1.00 | 1.00 | 0.00 |
| take_tray_out_of_oven | 1.00 | 0.90 | -0.10 |
| **MEAN** | **0.792** | **0.723** | **-0.069** |

**Conclusion:** 3-cams-per-task training generalizes well to an unseen camera group: -6.9 pts, with 8/13 tasks unchanged. The drop concentrates in put_item_in_drawer and pick_laptop (-0.30 each). pick_plate and straighten_rope are weak in BOTH conditions (0.20) — task/policy limitation, not viewpoint. G7 (never-collected, fully-OOD pose) remains an untested harder condition. Results/videos: s3://far-research-internal/harsvbha/3dfa/eval/results/peract2_orbital_nhist3_b200/{indomain,ood}/.

## Orbital PerAct2 — camera-miscalibration noise sweep, OOD camera (16 Aug 2026)

Jobs `hb-3dfa-orbnoise-<level>-<task>` (52 jobs, L40S:1, sky-us-east-1/-2). Checkpoint `peract2_orbital_nhist3_b200` @ iter 100000, `predict_extrinsics=false`. Same condition as the OOD column above — per-task `eval_group`, 10 rollouts/task, `image_space_sampling=false` — plus paired rot+trans miscalibration at eval time (`miscal_rot_level` + `miscal_trans_level`). The 0 column IS the OOD column above (not rerun: with no levels set the harness leaves extrinsics untouched, and the 5deg+5cm smoke confirmed the noise path only engages when a level is named).

The grogu-era sweep machinery was single-arm only; `d586f71` extends it to the 4-camera bimanual orbital harness. Noise perturbs only the extrinsics fed to depth→PCD (RGB and depth untouched), so the model sees a corrupted 3D scene. Directions are pre-sampled per camera in `instructions/random_miscal_noise_bimanual.json`.

| Task | 0 | 2deg+2cm | 5deg+5cm | 10deg+10cm | 15deg+15cm |
|---|:---:|:---:|:---:|:---:|:---:|
| push_box | 1.0 | 0.8 | 0.5 | 1.0 | 0.0 |
| lift_ball | 0.9 | 1.0 | 0.0 | 0.0 | 0.2 |
| dual_push_buttons | 0.9 | 0.0 | 0.0 | 0.0 | 0.0 |
| pick_plate | 0.2 | 0.0 | 0.0 | 0.0 | 0.0 |
| put_item_in_drawer | 0.5 | 0.0 | 0.0 | 0.0 | 0.0 |
| put_bottle_in_fridge | 0.9 | 0.5 | 0.2 | 0.0 | 0.0 |
| handover_item | 1.0 | 0.5 | 0.3 | 0.0 | 0.0 |
| pick_laptop | 0.3 | 0.0 | 0.0 | 0.0 | 0.0 |
| straighten_rope | 0.2 | 0.2 | 0.0 | 0.0 | 0.0 |
| sweep_to_dustpan | 0.6 | 0.0 | 0.0 | 0.0 | 0.0 |
| lift_tray | 1.0 | 0.9 | 0.3 | 0.0 | 0.0 |
| handover_item_easy | 1.0 | 0.1 | 0.2 | 0.0 | 0.0 |
| take_tray_out_of_oven | 0.9 | 0.7 | 0.0 | 0.0 | 0.0 |
| **MEAN** | **0.723** | **0.362** | **0.115** | **0.077** | **0.015** |
| retained vs 0 | 100% | 50% | 16% | 11% | 2% |
| tasks at 0.0 | 0/13 | 6/13 | 8/13 | 12/13 | 12/13 |

**Conclusion:** The curve collapses far earlier than the grogu-era single-arm default 3DFA did. There, `default_3dfa` on turn_tap held 0.91 at 2deg and 0.43 at 5deg before reaching 0.00 at 15deg; here half the performance is gone by **2deg+2cm** and 84% by 5deg+5cm, with 6/13 tasks already at zero at 2deg. Two plausible contributors: bimanual tasks need both arms' 3D estimates to be right simultaneously, so per-arm failures compound, and the OOD camera group already spends the model's viewpoint slack before any noise is added (the single-arm sweep ran at a trained G7 pose). Task strength at 0 noise does not predict robustness — handover_item_easy and dual_push_buttons are both 1.0/0.9 clean yet drop to 0.1/0.0 at 2deg, while push_box holds 0.5 at 5deg.

Two non-monotonic cells: push_box 1.0 at 10deg (vs 0.5 at 5deg) and lift_ball 0.2 at 15deg (vs 0.0 at 5/10deg). Both were verified as genuine runs (correct level in the log, 10/10 episodes) — the noise direction is a single fixed sample per level, so a level can happen to perturb along an axis this task tolerates. At 10 rollouts/task the per-cell noise floor is ±0.15, so read the level means, not individual cells.

This is the **no-deltaM baseline** for the upcoming extrinsics-prediction experiments. The bar the deltaM variants must clear: grogu-era `deltaM_EEF` stayed at 0.77 and `fixmed_rn` at 0.71 at 15deg+15cm on turn_tap, against this checkpoint's 0.015 mean. The steepness here means even a modest deltaM gain will be legible — but note grogu's robust variants were also *trained* with miscal noise, which this baseline was not, so a fair comparison needs a noise-trained bimanual control too.

Results: `s3://far-research-internal/harsvbha/3dfa/eval/results/peract2_orbital_nhist3_b200/noise_{2deg2cm,5deg5cm,10deg10cm,15deg15cm}/`. Re-run one level with: `sky jobs launch -y -d -n hb-3dfa-orbnoise-5deg5cm-<task> --infra k8s/sky-us-east-1 --env PREEMPTIBLE=1 --env TASK=<task> --env SPAWN_GROUP=<eval_group> --env MISCAL_ROT=5deg --env MISCAL_TRANS=5cm --env OUT_S3=.../noise_5deg5cm scripts/sky/peract2_orbital_online_eval.yaml`. sky-us-east-2 had no L40S capacity for the duration — three jobs sat STARTING for 30+ min and were relaunched into east-1; prefer east-1 for L40S eval waves.
