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

## R2 — deltaM under miscalibration: the matched miscal-trained pair (16 Aug 2026)

The decisive rung of `docs/status/deltaM_plan.md`. Two checkpoints trained as a matched pair on the orbital PerAct2 zarr with **persistently miscalibrated extrinsics** (`miscal=orbital_fixed_medium_randnoise`: fixed per-group medium base from `instructions/orbital_miscalibration_noise.json`, plus a <=3deg/<=1cm random top-up resampled per batch), differing in **exactly one flag**:

| arm | job | ckpt | wandb | the one delta |
|---|---|---|---|---|
| R1a baseline | 126269 | `orbital_miscal_base.pth` | [aq54hwdi](https://far.wandb.io/far-wandb/3dfa/runs/aq54hwdi) | `predict_extrinsics=false` |
| R1b deltaM | 126270 | `orbital_miscal_deltam.pth` | [2ks5zjmt](https://far.wandb.io/far-wandb/3dfa/runs/2ks5zjmt) | `predict_extrinsics=true`, `extrinsics_prediction_mode=delta_m`, `dynamic_rope_from_camtoken=true` |

Both verified at `iter=100000` before staging, with identical `orbital_miscal_noise_level=medium`, `miscal_max_angle_deg=3.0`, `miscal_max_translation_m=0.01`, `num_history=3`, `bimanual=true`, `image_space_sampling=false`, `backbone=siglip2`. The `predict_extrinsics` / `delta_m` flags were confirmed to arrive in each rollout process by reading the `loaded_from_ckpt` dump, not assumed.

**156 eval jobs** (`hb-3dfa-r2-{base,dm}-<level>-<task>`, L40S:1, all sky-us-east-1): 2 ckpts x 13 tasks x (5 noise levels + 1 clean0), 10 rollouts each, per-task `eval_group` (OOD camera) from `instructions/peract2_orbital_task_group_mapping.json`. Zero failures; 10 jobs the submission loop silently dropped were detected by diffing S3 against the task list and relaunched.

### The world stays miscalibrated — eval-side code change (`b9e05b0`)

The bimanual orbital harness supported only the *random* half of the miscal recipe, so these checkpoints' training condition could not be reproduced at eval. `orbital_miscal_noise_level` is now threaded through the bimanual branch of `evaluate_policy.py` and the harness, composed under the random levels in training's order:

    T_applied = T_random @ T_base[spawn_camera_group]

Reproduced exactly, including training's quirk: `orbital_miscalibration_noise.json` lists three cameras, so at `ncam=4` the fourth (`wrist_right`) is identity-padded. Verified numerically — cams 0-2 get 5.7-9.7 deg / 2-6 cm, cam 3 gets identity. Level **0 means fixed base only**, not clean; the `clean0` column below is the no-base condition. `orbital_miscal_noise_level` is in `_EVAL_RUNTIME_KEYS`, so the checkpoint's saved `medium` is never silently inherited — `ORBITAL_MISCAL_LEVEL` must name it.

Smoke gate before fan-out: base ckpt, `push_box`, level 0 with base → **0.70**, and the log confirmed `level='medium', group=G4`. A near-zero here would have meant the base was applied wrong; it was not.

### Three-curve comparison — mean SR over 13 tasks, OOD camera

| curve | 0 | 2deg+2cm | 5deg+5cm | 10deg+10cm | 15deg+15cm |
|---|:---:|:---:|:---:|:---:|:---:|
| clean-trained baseline (no deltaM, no train miscal) | 0.723 | 0.362 | 0.115 | 0.077 | 0.015 |
| **R1a miscal-trained, no deltaM** | **0.623** | **0.654** | **0.485** | **0.254** | **0.077** |
| **R1b miscal-trained + deltaM** | **0.508** | **0.500** | **0.469** | **0.262** | **0.138** |

| retained vs own level 0 | 0 | 2deg+2cm | 5deg+5cm | 10deg+10cm | 15deg+15cm |
|---|:---:|:---:|:---:|:---:|:---:|
| clean-trained | 100% | 50% | 16% | 11% | 2% |
| R1a | 100% | 105% | 78% | 41% | 12% |
| R1b | 100% | 98% | 92% | 52% | 27% |

Note the first column is **not** a common condition: the clean-trained 0.723 is clean extrinsics, whereas R1a/R1b's level 0 already carries the fixed medium base. The comparable cell for the clean-trained model under that base does not exist (it was never evaluated with one), so read columns 2-5, where all three curves share identical extrinsics.

**Verdict against the pre-registered signatures: neither. Train-time miscal is what buys robustness; deltaM adds a little at the extreme and costs accuracy everywhere else.**

1. **Training with miscal noise is the whole effect.** At 5deg+5cm the clean-trained model is at 0.115 while both miscal-trained arms are at ~0.47-0.49 — a **4x** gain. At 10deg the gap is 0.077 vs ~0.26. This reproduces grogu's `fixmed_rn` finding at bimanual scale and is by far the largest effect in the table.
2. **deltaM does not deliver "flat and high".** R1b is *below* R1a at three of five levels (0, 2deg, 5deg) and its level-0 mean is 11.5 pts lower. The pre-registered R1b signature required staying high; it did not.
3. **deltaM is flatter, from a lower start.** R1b retains 27% at 15deg vs R1a's 12%, and its absolute 15deg mean (0.138) is nearly double R1a's (0.077) — but 0.138 vs 0.077 on 13 tasks x 10 rollouts is ~1 extra success per 13 tasks, inside the ±0.15/cell noise floor. The plan's R2 decision rule (deltaM above baseline by >=10 pts at >=10deg) is met at 15deg by +6.1 pts and at 10deg by +0.8 pts — i.e. **not met**.
4. **This is the "tolerance, not correction" signature — and it belongs to R1a, not R1b.** The plan predicted the *baseline* would show tolerance and deltaM correction. What happened is that both learned tolerance from the training noise, and deltaM's extra degrees of freedom cost clean-condition accuracy without recovering a correction. Read straight: **on bimanual orbital, deltaM adds essentially nothing beyond the training augmentation it rides along with**, and it is net-negative at the noise levels that matter for real deployment (0-5 deg).

The fixed per-group base was persistent and in principle learnable from camera features — the sharpest possible test of the correction hypothesis, per the plan's own reasoning — and deltaM still did not beat a model that merely averaged over it. That closes the mechanism question the ladder was built to answer.

### Per-task: R1a (miscal-trained, no deltaM)

| Task | 0 | 2deg+2cm | 5deg+5cm | 10deg+10cm | 15deg+15cm |
|---|:---:|:---:|:---:|:---:|:---:|
| push_box | 0.7 | 0.6 | 0.4 | 0.7 | 0.3 |
| lift_ball | 1.0 | 1.0 | 0.9 | 0.7 | 0.2 |
| dual_push_buttons | 0.5 | 0.6 | 0.1 | 0.0 | 0.0 |
| pick_plate | 0.2 | 0.3 | 0.3 | 0.0 | 0.0 |
| put_item_in_drawer | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| put_bottle_in_fridge | 0.5 | 0.7 | 0.5 | 0.3 | 0.0 |
| handover_item | 0.6 | 0.7 | 0.4 | 0.0 | 0.0 |
| pick_laptop | 1.0 | 0.9 | 0.3 | 0.1 | 0.1 |
| straighten_rope | 0.1 | 0.1 | 0.0 | 0.0 | 0.0 |
| sweep_to_dustpan | 1.0 | 1.0 | 1.0 | 0.7 | 0.0 |
| lift_tray | 0.8 | 0.9 | 0.9 | 0.5 | 0.2 |
| handover_item_easy | 1.0 | 0.9 | 0.9 | 0.2 | 0.2 |
| take_tray_out_of_oven | 0.7 | 0.8 | 0.6 | 0.1 | 0.0 |
| **MEAN** | **0.623** | **0.654** | **0.485** | **0.254** | **0.077** |

### Per-task: R1b (miscal-trained + deltaM)

| Task | 0 | 2deg+2cm | 5deg+5cm | 10deg+10cm | 15deg+15cm |
|---|:---:|:---:|:---:|:---:|:---:|
| push_box | 0.9 | 0.5 | 0.7 | 0.9 | 0.5 |
| lift_ball | 1.0 | 1.0 | 1.0 | 0.7 | 0.5 |
| dual_push_buttons | 0.3 | 0.5 | 0.1 | 0.0 | 0.0 |
| pick_plate | 0.3 | 0.0 | 0.5 | 0.0 | 0.0 |
| put_item_in_drawer | 0.1 | 0.2 | 0.0 | 0.0 | 0.0 |
| put_bottle_in_fridge | 0.4 | 0.3 | 0.5 | 0.0 | 0.0 |
| handover_item | 0.1 | 0.6 | 0.5 | 0.0 | 0.0 |
| pick_laptop | 0.5 | 0.5 | 0.2 | 0.0 | 0.0 |
| straighten_rope | 0.0 | 0.0 | 0.0 | 0.0 | 0.1 |
| sweep_to_dustpan | 1.0 | 0.8 | 0.9 | 0.9 | 0.0 |
| lift_tray | 0.8 | 0.9 | 0.9 | 0.9 | 0.0 |
| handover_item_easy | 0.8 | 0.7 | 0.6 | 0.0 | 0.7 |
| take_tray_out_of_oven | 0.4 | 0.5 | 0.2 | 0.0 | 0.0 |
| **MEAN** | **0.508** | **0.500** | **0.469** | **0.262** | **0.138** |

`put_item_in_drawer` and `straighten_rope` are at ~0 for both arms at every level — they were already the weakest tasks on clean extrinsics (0.5 / 0.2 in the clean-trained OOD column), so miscal training did not rescue them and deltaM had nothing to correct. The deltaM gain at 15deg is carried almost entirely by three cells (`handover_item_easy` 0.7, `push_box` 0.5, `lift_ball` 0.5), the same easy-task concentration that made grogu's effect sizes fragile (`docs/status/stuls.md`). Several non-monotonic cells appear in both arms (R1b `push_box` 0.9 at 10deg vs 0.7 at 5deg; `handover_item_easy` 0.7 at 15deg vs 0.0 at 10deg) — a level is one fixed noise direction, so a task can tolerate one axis and not another. Read level means, not cells.

### clean0 secondary — did noise training cost clean-extrinsics performance?

Same checkpoints, no fixed base and no random noise, OOD camera:

| Task | R1a clean0 | R1b clean0 | R1a level 0 (with base) | R1b level 0 (with base) |
|---|:---:|:---:|:---:|:---:|
| push_box | 0.6 | 1.0 | 0.7 | 0.9 |
| lift_ball | 1.0 | 1.0 | 1.0 | 1.0 |
| dual_push_buttons | 0.8 | 0.8 | 0.5 | 0.3 |
| pick_plate | 0.7 | 0.1 | 0.2 | 0.3 |
| put_item_in_drawer | 0.2 | 0.2 | 0.0 | 0.1 |
| put_bottle_in_fridge | 0.8 | 0.6 | 0.5 | 0.4 |
| handover_item | 0.6 | 0.3 | 0.6 | 0.1 |
| pick_laptop | 0.4 | 0.7 | 1.0 | 0.5 |
| straighten_rope | 0.4 | 0.1 | 0.1 | 0.0 |
| sweep_to_dustpan | 1.0 | 0.9 | 1.0 | 1.0 |
| lift_tray | 1.0 | 0.9 | 0.8 | 0.8 |
| handover_item_easy | 1.0 | 0.7 | 1.0 | 0.8 |
| take_tray_out_of_oven | 0.5 | 0.1 | 0.7 | 0.4 |
| **MEAN** | **0.692** | **0.569** | **0.623** | **0.508** |

**Miscal training costs ~3 pts of clean OOD performance, not the ~10 grogu implied.** R1a reaches 0.692 on clean extrinsics against the clean-trained model's 0.723 — within the noise floor. Grogu's `fixmed_rn` was the *weakest* of its three variants on clean novel views; on bimanual the penalty is much smaller, so **train-time miscal is close to free**: 4x robustness at 5deg for ~3 pts clean. That is the actionable result of this round.

R1b again pays for deltaM: 0.569 clean vs R1a's 0.692, a **-12.3 pt** gap that matches its -11.5 at level 0. deltaM's cost is not condition-specific — it is a flat accuracy tax the mechanism levies whether or not there is anything to correct.

### Program status

**Stop the deltaM line.** The plan gated R3 (deltaM + train miscal) on R2 showing deltaM degrade by >15 pts at 15deg — but R3's config *is* R1b, already run and answered. `delta_m_full` and `rt` were already cut. What is left worth doing:

* **R4 (G7 held-out) on R1a, not R1b.** R1a is the winner of this pair on every level except 15deg, and it is the cheaper model. 13 jobs.
* The `RTExtrinsicsPredictor` signature bug (`base_denoise_actor.py:903-905` passes `fps_*` kwargs it cannot accept) is still live and still a 1-line fix. Worth fixing so the path does not rot, not worth a run.
* Train-time miscal noise should become the **default** for orbital training given the ~free 4x robustness.

Results: `s3://far-research-internal/harsvbha/3dfa/eval/results/{orbital_miscal_base,orbital_miscal_deltam}/{clean0,noise_0,noise_2deg2cm,noise_5deg5cm,noise_10deg10cm,noise_15deg15cm}/`. Checkpoints: `s3://far-research-internal/harsvbha/3dfa/eval/ckpt/orbital_miscal_{base,deltam}.pth`. Re-run one cell with `--env ORBITAL_MISCAL_LEVEL=medium --env MISCAL_ROT=5deg --env MISCAL_TRANS=5cm` on `scripts/sky/peract2_orbital_online_eval.yaml`.

## R2b — OOD fixed miscalibration: does miscal training generalize or memorize? (16 Aug 2026)

R2 established that train-time miscal buys 4x robustness at 5deg. But every R2 cell
applied the *same* fixed per-group base the checkpoints trained on
(`instructions/orbital_miscalibration_noise.json`), so "robustness" was untested
against the alternative reading: the models simply absorbed one specific calibration
error into their weights. This round applies a **never-seen fixed miscalibration of
the same magnitude** and no random noise on top, isolating the fixed-base
generalization question.

### The held-out miscalibration (`17a440e`)

`instructions/orbital_miscalibration_noise_ood.json` — same generator
(`scripts/generate_orbital_miscal_noise.py`), same magnitude configs, seed **3187**
instead of 42. The seed was chosen by searching 4000 candidates for the closest
match to the training file's aggregate magnitude, so an SR difference cannot be
dismissed as an easier or harder condition:

| medium level | mean angle | mean \|t\| | angle range | \|t\| range | mean axis separation vs training |
|---|:---:|:---:|:---:|:---:|:---:|
| training (seed 42) | 5.95 deg | 5.26 cm | 1.06-9.72 deg | 2.1-7.1 cm | — |
| **OOD (seed 3187)** | **5.90 deg** | **5.26 cm** | **1.42-9.64 deg** | **3.0-6.7 cm** | **82 deg** |

Magnitudes match to within 1%; the per-camera rotation axes point in genuinely
different directions (82 deg apart on average). Same three-camera structure, so
`wrist_right` stays identity-padded at ncam=4 exactly as in training and R2. The
training file is untouched and still pinned.

`orbital_miscal_noise_file` (env `MISCAL_FILE`) selects the file; the default is
unchanged, and the harness logs which file it loaded. Added to `_EVAL_RUNTIME_KEYS`
so it is never inherited from a checkpoint.

**26 jobs** (`hb-3dfa-oodmiscal-{base,dm}-<task>`, L40S:1, all sky-us-east-1): 2 ckpts
x 13 tasks x OOD base at level medium x 10 rollouts, per-task `eval_group`. Zero
failures. All 26 logs were verified to carry
`file=instructions/orbital_miscalibration_noise_ood.json`, the correct group, **no**
`random miscal:` line, and the right `predict_extrinsics` value read from the
checkpoint. Smoke gate: base ckpt, `push_box` -> **0.70**, matching its trained-miscal
cell exactly.

### Results — mean SR over 13 tasks, OOD camera, no random noise

| condition | R1a (no deltaM) | R1b (deltaM) |
|---|:---:|:---:|
| clean extrinsics (clean0) | 0.692 | 0.569 |
| trained fixed miscal (R2 level 0) | 0.623 | 0.508 |
| **held-out fixed miscal (this round)** | **0.415** | **0.446** |
| drop vs trained miscal | **-20.8 pts** | **-6.2 pts** |
| retained vs trained miscal | 67% | 88% |

| Task | R1a ood | R1b ood | R1a trained | R1b trained | R1a clean0 | R1b clean0 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| push_box | 0.7 | 0.8 | 0.7 | 0.9 | 0.6 | 1.0 |
| lift_ball | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| dual_push_buttons | 0.4 | 0.4 | 0.5 | 0.3 | 0.8 | 0.8 |
| pick_plate | 0.2 | 0.2 | 0.2 | 0.3 | 0.7 | 0.1 |
| put_item_in_drawer | 0.1 | 0.1 | 0.0 | 0.1 | 0.2 | 0.2 |
| put_bottle_in_fridge | 0.2 | 0.4 | 0.5 | 0.4 | 0.8 | 0.6 |
| handover_item | 0.1 | 0.2 | 0.6 | 0.1 | 0.6 | 0.3 |
| pick_laptop | 0.0 | 0.0 | 1.0 | 0.5 | 0.4 | 0.7 |
| straighten_rope | 0.1 | 0.0 | 0.1 | 0.0 | 0.4 | 0.1 |
| sweep_to_dustpan | 1.0 | 0.9 | 1.0 | 1.0 | 1.0 | 0.9 |
| lift_tray | 0.8 | 1.0 | 0.8 | 0.8 | 1.0 | 0.9 |
| handover_item_easy | 0.0 | 0.2 | 1.0 | 0.8 | 1.0 | 0.7 |
| take_tray_out_of_oven | 0.8 | 0.6 | 0.7 | 0.4 | 0.5 | 0.1 |
| **MEAN** | **0.415** | **0.446** | **0.623** | **0.508** | **0.692** | **0.569** |

### Verdict: partial memorization, and R1b overtakes R1a for the first time

Against the pre-registered signatures, this is closest to **"both drop, but not to
zero"** — with the twist that the drops are very unequal.

1. **R1a's robustness was substantially in-distribution.** Losing 20.8 pts (0.623 ->
   0.415, 67% retained) when only the *direction* of a same-magnitude perturbation
   changes means a real part of what looked like calibration robustness in R2 was
   the model having absorbed one specific miscalibration. R2's headline 4x-at-5deg
   should be discounted accordingly: some of it is tolerance, some is fit to a fixed
   base.
2. **But it is not pure memorization either.** 0.415 on a never-seen fixed base is
   still far above the clean-trained baseline's 0.115 at 5deg+5cm — the closest
   available comparison at similar perturbation magnitude. Train-time miscal
   transfers *something*; it just transfers less than R2 implied.
3. **R1b beats R1a here — the first condition where deltaM leads on absolute SR
   below 15deg.** 0.446 vs 0.415 (+3.1 pts), and far more telling, R1b drops only
   6.2 pts against R1a's 20.8. deltaM is markedly less dependent on the specific
   miscalibration it trained under, which is exactly what a correction mechanism
   should look like.
4. **This is not enough to revive the deltaM line.** The pre-registered revival
   condition was "R1a drops hard but R1b holds". R1a dropped hard; R1b did not
   *hold* — it declined too, and its 0.446 remains below R1a's own trained-miscal
   0.623 and its clean-extrinsics 0.569. The +3.1 pt lead is ~4 successes across 130
   rollouts, inside the +/-0.15/cell noise floor, and deltaM still pays the ~12 pt
   clean-condition tax documented in R2. The honest summary is that **deltaM buys
   invariance to *which* calibration error, not accuracy under calibration error**,
   and it pays for that invariance at a rate that leaves it behind on every
   in-distribution condition.
5. **The drop is concentrated, as usual.** Seven of 13 tasks are within +/-0.1 of
   their trained-miscal value for R1a; the -20.8 pt mean is carried almost entirely
   by three collapses — `pick_laptop` 1.0 -> 0.0, `handover_item_easy` 1.0 -> 0.0,
   `handover_item` 0.6 -> 0.1. R1b's advantage is that it does not collapse on
   `handover_item_easy` (0.8 -> 0.2) and was already low on `pick_laptop`. Both
   `pick_laptop` cells are 0.0, so that task is genuinely broken by the new base for
   both arms, not a deltaM win. Read the means, not the cells.

**Program impact:** R2's "stop the deltaM line" stands, but its rationale narrows.
deltaM is dead as an *accuracy* mechanism; it is not dead as a *calibration-invariance*
mechanism, and that distinction matters if a future setting has calibration error
that genuinely varies at test time (multi-robot fleets, re-mounted cameras). The
actionable change to R2's recommendation: train-time miscal should still be the
orbital default, but with **resampled** fixed bases rather than one pinned base —
this round shows a single fixed base is partly memorized. That is a cheap
augmentation change worth more than any deltaM run.

Results: `s3://far-research-internal/harsvbha/3dfa/eval/results/{orbital_miscal_base,orbital_miscal_deltam}/ood_miscal/`. Re-run one cell with `--env ORBITAL_MISCAL_LEVEL=medium --env MISCAL_FILE=instructions/orbital_miscalibration_noise_ood.json` on `scripts/sky/peract2_orbital_online_eval.yaml`.

## R2c — deltaM + EE-aux head: the third arm (17 Aug 2026)

R2 found deltaM (R1b) *below* the plain miscal-trained baseline (R1a) on every
in-distribution condition, and R2b found it flat-but-not-better under a held-out
base. Both readings left one loose end: grogu's winning `deltaM_EEF` variant also
trained an **end-effector auxiliary head** off the camera trunk, and R1b did not.
The aux head is the plausible missing piece — it supervises the same per-camera
features that produce deltaM with a task the features can only solve if they encode
metric camera geometry, so deltaM's 6x6 correction has a reason to be geometrically
meaningful rather than a free reparameterization. R1c tests that.

| arm | job | ckpt | wandb | the delta vs R1b |
|---|---|---|---|---|
| R1c deltaM + EE-aux | 127191 | `orbital_miscal_deltam_eeaux.pth` | [9w9w8xwy](https://far.wandb.io/far-wandb/3dfa/runs/9w9w8xwy) | `predict_ee_aux=true`, `lambda_aux=1.0`, `ee_aux_cam_ids=[0,1]` |

Verified at `iter=100000` before staging, with `predict_ee_aux=true`,
`predict_extrinsics=true`, `extrinsics_prediction_mode=delta_m`,
`dynamic_rope_from_camtoken=true` and otherwise identical to R1b
(`orbital_miscal_noise_level=medium`, `miscal_max_angle_deg=3.0`,
`miscal_max_translation_m=0.01`, `num_history=3`, `bimanual=true`,
`image_space_sampling=false`, `backbone=siglip2`). As in R2, the flags were
confirmed to arrive in the rollout process by reading the overlay dump, not assumed.

**91 eval jobs** (`hb-3dfa-r1c-<cond>-<task>`, L40S:1, all sky-us-east-1): 13 tasks
x (level 0 + 4 noise levels + held-out base + clean0), 10 rollouts each, per-task
`eval_group` (OOD camera). Smoke gate: `push_box` at level 0 -> **1.00**, log
confirming `level='medium', group=G4` and the training miscal file. All 91 cells
landed.

### The aux loss kept descending — it did not saturate

The pre-registered diagnostic was whether `train/ee_aux_loss` kept falling past the
early plateau at 0.028 (step 900). It did, by more than an order of magnitude:

| step | 49 | 900 | 5k | 20k | 50k | 75k | 100k |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `train/ee_aux_loss` | 0.0582 | 0.0284 | 0.0141 | 0.0059 | 0.0034 | 0.0027 | **0.0027** |

Minimum 0.00235 at step 96549; final summary value 0.00267. The plateau at 0.028 was
an early transient, not a ceiling — the curve fell ~10x after it and only flattened
in the last ~25k steps. **So the aux head genuinely learned to regress EE position
from camera-trunk features.** The mechanism was trained as intended, which means the
success-rate result below is a verdict on the mechanism, not on a failed
optimization.

### condition | R1a | R1b | R1c — mean SR over 13 tasks, OOD camera

| condition | R1a (no deltaM) | R1b (deltaM) | R1c (deltaM + EE-aux) | R1c-R1b | R1c-R1a |
|---|:---:|:---:|:---:|:---:|:---:|
| clean extrinsics (clean0) | 0.692 | 0.569 | **0.623** | +0.054 | -0.069 |
| trained fixed miscal (level 0) | 0.623 | 0.508 | **0.462** | -0.046 | -0.161 |
| + random 2deg+2cm | 0.654 | 0.500 | **0.431** | -0.069 | -0.223 |
| + random 5deg+5cm | 0.485 | 0.469 | **0.523** | +0.054 | +0.038 |
| + random 10deg+10cm | 0.254 | 0.262 | **0.308** | +0.046 | +0.054 |
| + random 15deg+15cm | 0.077 | 0.138 | **0.262** | +0.124 | +0.185 |
| held-out fixed miscal | 0.415 | 0.446 | **0.408** | -0.038 | -0.007 |

| aggregate over the 4 shared noise levels | R1a | R1b | R1c |
|---|:---:|:---:|:---:|
| mean of 2/5/10/15 deg | 0.367 | 0.342 | **0.381** |
| mean of the two high levels (10+15) | 0.166 | 0.200 | **0.285** |
| retained at 15deg vs own level 0 | 12% | 27% | **57%** |

### Per-task: R1c (miscal-trained + deltaM + EE-aux)

| Task | 0 | 2deg+2cm | 5deg+5cm | 10deg+10cm | 15deg+15cm | ood | clean0 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| push_box | 1.0 | 1.0 | 1.0 | 1.0 | 0.7 | 1.0 | 1.0 |
| lift_ball | 0.9 | 0.9 | 0.9 | 0.9 | 0.7 | 0.9 | 0.9 |
| dual_push_buttons | 0.4 | 0.7 | 0.1 | 0.0 | 0.0 | 0.5 | 0.6 |
| pick_plate | 0.4 | 0.2 | 0.5 | 0.0 | 0.1 | 0.2 | 0.5 |
| put_item_in_drawer | 0.0 | 0.0 | 0.0 | 0.1 | 0.0 | 0.1 | 0.2 |
| put_bottle_in_fridge | 0.2 | 0.2 | 0.2 | 0.0 | 0.0 | 0.2 | 0.4 |
| handover_item | 0.6 | 0.3 | 0.1 | 0.1 | 0.2 | 0.2 | 0.4 |
| pick_laptop | 0.9 | 0.8 | 0.8 | 0.1 | 0.2 | 0.3 | 0.5 |
| straighten_rope | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.3 |
| sweep_to_dustpan | 0.0 | 0.0 | 1.0 | 0.5 | 1.0 | 0.6 | 1.0 |
| lift_tray | 0.6 | 0.5 | 1.0 | 1.0 | 0.3 | 0.8 | 1.0 |
| handover_item_easy | 0.6 | 0.4 | 0.9 | 0.1 | 0.1 | 0.1 | 0.6 |
| take_tray_out_of_oven | 0.4 | 0.6 | 0.3 | 0.2 | 0.1 | 0.4 | 0.7 |
| **MEAN** | **0.462** | **0.431** | **0.523** | **0.308** | **0.262** | **0.408** | **0.623** |

`sweep_to_dustpan` is the loudest non-monotonic cell in the whole program: 0.0 at
level 0 and 2deg but 1.0 at 5deg and 15deg, and 1.0 clean. The level-0 log was
checked directly — 10 valid demos, correct group G1, correct miscal file, no
traceback, a genuine 0/10. A level is one fixed noise *direction*, so this is the
same axis-sensitivity seen in both R2 arms, in its most extreme form. It is a
reminder that individual cells at n=10 are close to uninformative here; only the
13-task level means should be read.

### Verdict: the aux head recovers deltaM's tax and extends its high-noise edge, but does not beat the baseline where it counts

Answering the three pre-registered questions directly, judging on level means over
13 tasks against the ±0.15/cell noise floor:

1. **Does EE-aux widen the deltaM gap (R1c > R1b)? Partly — at high noise, yes;
   in-distribution, no.** R1c beats R1b at 5deg (+5.4), 10deg (+4.6), 15deg (+12.4)
   and clean0 (+5.4), but *loses* at level 0 (-4.6), 2deg (-6.9) and the held-out
   base (-3.8). Only the 15deg gap clears the noise floor. The honest summary is
   that the aux head **rotates deltaM's tradeoff further toward high-noise
   robustness** rather than lifting it uniformly.
2. **Does it recover the clean tax? Roughly half of it.** R1b paid -12.3 pts on
   clean extrinsics vs R1a (0.569 vs 0.692); R1c is at 0.623, recovering +5.4 of
   those 12.3 and leaving -6.9 vs R1a — now inside the noise floor. So the aux head
   does make deltaM cheaper on clean data, consistent with it regularizing the
   camera trunk toward real geometry. It does not make it free.
3. **Does it beat R1a anywhere meaningful? Only at 15deg, and R1a's collapse there
   makes that cheap.** R1c's +18.5 at 15deg (0.262 vs 0.077) and +5.4 at 10deg are
   the only positive deltas; the first clears the floor, the second does not.
   Against that, R1c is **-16.1 at level 0 and -22.3 at 2deg** — two clear
   floor-clearing *losses* on the conditions closest to a real deployment. Averaged
   over the four shared noise levels the three arms are 0.367 / 0.342 / 0.381, i.e.
   indistinguishable.

**This closes the deltaM ladder with the same answer R2 gave, better supported.**
The aux head was the mechanism's best remaining shot, it trained correctly (the loss
fell 10x, so this is not a null from a broken run), and it still does not produce a
model that is better than plain train-time miscal augmentation at 0-5 deg. What it
produces is the flattest curve in the program — 57% retained at 15deg vs R1a's 12% —
purchased with 16-22 pts at low noise. **deltaM+aux is a robustness/accuracy trade,
not a Pareto improvement**, and the trade only pays if the deployment genuinely sees
>10deg calibration error, which is far outside any plausible real miscalibration.

**Program impact:** R2's "stop the deltaM line" now stands on all three arms, and
R2b's narrowed rationale is confirmed and sharpened: deltaM (with or without the aux
head) buys *tolerance to large and varying* calibration error at a fixed accuracy
cost. R2b's recommendation is unchanged and remains the cheap win — make train-time
miscal the orbital default with **resampled** fixed bases. R4 (G7 held-out) should
still run on R1a, which is the best arm at every condition below 10deg.

Results: `s3://far-research-internal/harsvbha/3dfa/eval/results/orbital_miscal_deltam_eeaux/{level0,noise_2deg2cm,noise_5deg5cm,noise_10deg10cm,noise_15deg15cm,ood_miscal,clean0}/`. Checkpoint: `s3://far-research-internal/harsvbha/3dfa/eval/ckpt/orbital_miscal_deltam_eeaux.pth`. Fan-out and collection: `scripts/eval/reconcile_r1c.sh` (idempotent — diffs the grid against the sky queue and S3, resubmits only missing cells) and `scripts/eval/collect_r1c.py`.

**Infra note for future waves:** submitting these loops in parallel does *not* work.
Three concurrent `sky jobs launch` loops overwhelmed the `RestfulAdminPolicy`
sidecar (`admin-policy:80` connection refused / read timeout) and silently dropped
66 of the first 91 submissions. Submit serially with retry, then reconcile against
S3. `reconcile_r1c.sh` does both.
