# Figures: week of 30 Sept – 6 Oct 2026

Run the scripts from the repo root with `/home/harshilb/miniconda3/envs/3dfa/bin/python`.

## delta_M per layer ([TODOs.md](../TODOs.md), TODO 1)

The data comes from `scripts/analysis/delta_m_per_layer.py` (GPU, via `scripts/analysis/delta_m_per_layer.slurm`), which writes `results/analysis/delta_m_per_layer/<model>.pt`. The figures and `summary.json` come from `scripts/analysis/plot_delta_m_per_layer.py` (CPU). Models: exp 3 at 200k and 110k (EMA), VideoPooled 84k and warm random history 50k. Data: all 853 val samples, cameras 0 and 1.

| Figure | Shows | Script |
|---|---|---|
| `dm_layer_dist_from_identity.{png,svg}` | ‖ΔM − I‖_F per layer (CA, SA0–SA3), mean ± std over samples | `scripts/analysis/plot_delta_m_per_layer.py` |
| `dm_layer_depth_change.{png,svg}` | ‖log(ΔM_l ΔM_{l−1}ᵀ)‖_F between consecutive predictions | same |
| `dm_layer_task_share.{png,svg}` | Share of across-sample variance explained by task (η²); no point where delta_M has no spread | same |
| `dm_layer_clean_vs_heavy.{png,svg}` | ‖ΔM_clean − ΔM_60/60‖_F on the same sample with the same flow noise | same |
| `dm_layer_across_steps.{png,svg}` | RMS variation of delta_M across the 5 flow steps | same |
| `dm_layer_cam0_vs_cam1.{png,svg}` | ‖log(ΔM_cam0 ΔM_cam1ᵀ)‖_F per layer | same |

## Camera attention ([TODOs.md](../TODOs.md), TODO 3)

The data comes from `scripts/analysis/camera_attention.py` (GPU, via `scripts/analysis/camera_attention.slurm`), which writes `results/analysis/camera_attention/<model>.pt`. The figures, `summary.json` and `summary_table.txt` come from `scripts/analysis/plot_camera_attention.py` (CPU). Models: exp 3 200k (EMA), 1b 200k (EMA), control A 200k (raw) and ema20k_randhist 15k (EMA). Data: all 853 val samples, clean vs heavy60 on cameras 0 and 1.

| Figure | Shows | Script |
|---|---|---|
| `cam_attn_external_share.{png,svg}` | Trajectory-query attention on cam 0+1 scene tokens as a share of all scene-token attention, per layer, clean vs heavy60 (dotted = token share); Δ heavy60 − clean ± SE; Δ attention entropy | `scripts/analysis/plot_camera_attention.py` |
| `cam_attn_sa_mass_breakdown.{png,svg}` | Trajectory-query attention mass by key group (scene cams 0+1 / 2+3, C-tokens, registers, camera token, trajectory) per SA and output-head layer, clean vs heavy60 | same |

## 1b camera-adaLN signal ([TODOs.md](../TODOs.md), TODO 2)

The data comes from `scripts/analysis/adaln_signal.py` (GPU), which writes `results/analysis/adaln_signal/exp1b_200k.pt`. The figures and `exp1b_200k.json` come from `scripts/analysis/plot_adaln_signal.py` (CPU). Data: all 853 val samples, clean / train / heavy60 on cameras 0 and 1.

| Figure | Shows | Script |
|---|---|---|
| `adaln_1b_modulation_strength.{png,svg}` | Per AdaLN module: ‖scale_full − scale_time‖/‖scale_time‖ and the same for shift | `scripts/analysis/plot_adaln_signal.py` |
| `adaln_1b_camera_term_layers.{png,svg}` | Camera term per signal: across-sample spread / norm (time_embs as reference), task η², clean-vs-miscal change / across-sample spread | same |
| `adaln_1b_camera_term_pca.{png,svg}` | PCA of the per-sample camera term at SA1 and at the output heads, coloured by task | same |

## delta_M and the RoPE kernel ([TODOs.md](../TODOs.md), TODO 4)

`scripts/analysis/rope_kernel.py` (CPU) reads TODO 1's K=5 tensors and writes `results/analysis/rope_kernel/summary.json` plus these figures.

| Figure | Shows | Script |
|---|---|---|
| `rope_kernel_displacement.{png,svg}` | Even RoPE logit factor vs key displacement along x/y/z, plain basis vs exp 3's per-layer delta_M; query at the origin (top) and averaged over the workspace (bottom) | `scripts/analysis/rope_kernel.py` |
| `rope_delta_m_heatmaps.{png,svg}` | Sample-mean 6×6 delta_M (camera 0) for exp 3 CA/SA1/SA2/SA3 and VideoPooled CA/SA1 | same |

## Non-history models: delta_M and camera attention ([TODOs.md](../TODOs.md), TODO 8)

The data comes from `scripts/analysis/todo8.slurm` (GPU). `PART=dm` runs `delta_m_per_layer.py` and `delta_m_trunk_activity.py`; `PART=attn` runs `camera_attention.py`. All use `--faithful` (`scripts/analysis/eval_faithful_load.py`) and write to `results/analysis/todo8/`. Models: DeltaM 140k (K=1, delta_M on all 4 cameras by the legacy rule) and Base 160k (K=1), next to the history models at their saved K=5. Data: all 853 val samples. The TODO 3 figures `cam_attn_*` were regenerated from the same K=5 run.

| Figure | Shows | Script |
|---|---|---|
| `todo8_dm_layer_{dist_from_identity,depth_change,task_share,clean_vs_heavy,across_steps,cam0_vs_cam1}.{png,svg}` | The six TODO 1 delta_M metrics with DeltaM 140k added | `scripts/analysis/plot_delta_m_per_layer.py --in-dir results/analysis/todo8/delta_m_per_layer --fig-prefix todo8_dm_layer` |
| `todo8_cam_attn_external_share.{png,svg}` | External attention share per layer, clean vs heavy60, Δ ± SE and Δ entropy; DeltaM 140k and Base 160k overlaid on the four history models | `scripts/analysis/plot_camera_attention.py --in-dir results/analysis/todo8/camera_attention --fig-prefix todo8_cam_attn` |
| `todo8_cam_attn_sa_mass_breakdown.{png,svg}` | Attention mass by key group per SA/output-head layer, six models | same |

## Linear probes: camera pose and injected miscalibration ([TODOs.md](../TODOs.md), TODO 5)

The data comes from `scripts/analysis/probe_features.py` (GPU, via `scripts/analysis/probe_features.slurm`, one model per job), which writes `results/analysis/probes/<model>.pt`. Probes are fit by `scripts/analysis/fit_probes.py` (CPU, via `scripts/analysis/fit_probes.slurm <model>`) into `results/analysis/probes/fits/`. The figures and `summary_table.md` come from `scripts/analysis/plot_probes.py` (CPU). Models: exp 3 200k (EMA), 1b 200k (EMA), control A 200k (raw), ema20k_randhist 15k (EMA) and random-init exp 3, all at K=5. Data: 3000 training-split rows to fit, all 853 val rows to test. Cameras 0+1 are averaged. Solid lines are C-tokens at the last flow step; dotted lines are the first step; hollow markers are delta_M / the adaLN term at their input depth.

| Figure | Shows | Script |
|---|---|---|
| `probe_pose.{png,svg}` | (A) True pose vs depth (enc, hist, SA1–SA4): G1–G6 accuracy, iid position error, leave-one-group-out position and viewing-direction error with mean and nearest-other-pose baselines | `scripts/analysis/plot_probes.py` |
| `probe_miscal.{png,svg}` | (B) Probe-only sweep (0–60°, 0–60 cm) vs depth: R² for rotation magnitude, rotation vector and translation vector; clean vs miscalibrated accuracy | same |

## Following the miscalibration ([TODOs.md](../TODOs.md), TODO 6)

The data comes from `scripts/analysis/follow_miscal.py` (GPU, via `scripts/analysis/follow_miscal.slurm`), which writes `results/analysis/follow_miscal/<model>.pt`. The figures, `summary.json` and `summary_table.txt` come from `scripts/analysis/plot_follow_miscal.py` (CPU). Models: exp 3 200k and 110k (EMA), 1b 200k (EMA), control A 200k (raw). Data: all 853 val samples, K = 5.

| Figure | Shows | Script |
|---|---|---|
| `follow_alpha_pos.{png,svg}` | Position following α, from (p̂ − p) ≈ α (N_j p − p), against miscalibration magnitude. Panels: both external cameras (joint fit), camera 0 only, camera 1 only (shared zoomed axis), and all 4 cameras with the same N (0–1 axis). 95% bootstrap CI | `scripts/analysis/plot_follow_miscal.py` |
| `follow_beta_rot.{png,svg}` | Orientation following β, from log(R̂ Rᵀ) ≈ β log(R_Nj), same panels | same |
| `follow_closest_share.{png,svg}` | Share of (sample, hand) pairs whose 60/60 prediction is closest to the true p, N₀p or N₁p | same |

## Activation patching ([TODOs.md](../TODOs.md), TODO 7)

The data comes from `scripts/analysis/activation_patching.py` (GPU, via `scripts/analysis/activation_patching.slurm`), which writes `results/analysis/activation_patching/<model>.pt`. The figure, `summary.json` and `summary_table.txt` come from `scripts/analysis/plot_activation_patching.py` (CPU). Models: exp 3 200k (EMA), 1b 200k (EMA), control A 200k (raw). Data: all 853 val samples, K = 5, clean vs heavy60 on cameras 0+1, with identical flow noise.

| Figure | Shows | Script |
|---|---|---|
| `patch_fraction_recovered.{png,svg}` | Per patched component: the fraction of the clean → heavy60 degradation recovered (clean into heavy) or caused (heavy into clean), for position L2 and rotation L1, models side by side, 95% paired bootstrap CI | `scripts/analysis/plot_activation_patching.py` |

## Activation patching on non-history models ([TODOs.md](../TODOs.md), TODO 9)

The data comes from `scripts/analysis/activation_patching.py --faithful` (GPU, via `scripts/analysis/activation_patching_todo9.slurm`), which writes `results/analysis/todo9/activation_patching/<model>.pt`. The figure, `summary.json` and `summary_table.txt` come from `scripts/analysis/plot_activation_patching.py --in-dir results/analysis/todo9/activation_patching --fig-stem todo9_patch_fraction_recovered`. Models: DeltaM 140k and Base 160k (raw, K=1, legacy all-camera delta_M), plus exp 3, 1b and A at their saved K=5.

| Figure | Shows | Script |
|---|---|---|
| `todo9_patch_fraction_recovered.{png,svg}` | Per patched component (C-token, FPS and dense positions, proprio, delta_M/adaLN replay, and combinations): the fraction of the clean → heavy60 degradation recovered (clean into heavy) and caused (heavy into clean), for position L2 and rotation L1, five models, 95% paired bootstrap CI | `scripts/analysis/plot_activation_patching.py` |

## C-token behaviour ([TODOs.md](../TODOs.md), TODO 10)

The data comes from `scripts/analysis/ctoken_behaviour.py` (GPU, via `scripts/analysis/ctoken_behaviour.slurm`), which writes `results/analysis/ctoken_behaviour/<model>.pt`. The figure, `summary.json` and `summary_table.txt` come from `scripts/analysis/plot_ctoken_behaviour.py` (CPU). Models: Base 160k and DeltaM 140k (raw, K=1), 1b 200k and exp 3 200k (EMA, K=5). Data: all 853 val samples, clean vs heavy60.

| Figure | Shows | Script |
|---|---|---|
| `ctoken_rel_change.{png,svg}` | ‖h_heavy60 − h_clean‖/‖h_clean‖ per layer for each C-token (cams 0–3), external and wrist FPS tokens, and trajectory tokens; one panel per model | `scripts/analysis/plot_ctoken_behaviour.py` |

## Full-patch models: delta_M, attention and C-tokens ([TODOs.md](../TODOs.md), TODO 12)

The data comes from `scripts/analysis/fullpatch_suite.slurm` (GPU; `PART=attn|dm|ctok`), which writes `results/analysis/todo12/`. Models: FP, FP-d8, FP-R3D-refine and FP-R3D-direct (raw, K=5, loaded as online eval does). Data: all 853 val samples, clean vs heavy60.

| Figure | Shows | Script |
|---|---|---|
| `todo12_dm_layer_{dist_from_identity,depth_change,task_share,clean_vs_heavy,across_steps,cam0_vs_cam1}.{png,svg}` | The six TODO 1 delta_M metrics for FP, FP-d8 and FP-R3D-refine (the Direct delta_M is one constant matrix; see `summarise_todo12.py`) | `scripts/analysis/plot_delta_m_per_layer.py --in-dir results/analysis/todo12/delta_m_per_layer --fig-prefix todo12_dm_layer` |
| `todo12_cam_attn_external_share.{png,svg}`, `todo12_cam_attn_sa_mass_breakdown.{png,svg}` | External attention share per layer, clean vs heavy60, Δ and entropy; mass by key group | `scripts/analysis/plot_camera_attention.py --in-dir results/analysis/todo12/camera_attention --fig-prefix todo12_cam_attn` |
| `todo12_ctoken_rel_change.{png,svg}` | Per-layer relative change under heavy60 of C-tokens, external/wrist FPS tokens and trajectory tokens | `scripts/analysis/plot_ctoken_behaviour.py --in-dir results/analysis/todo12/ctoken_behaviour --fig docs/research/experiments/2026-10-06/figures/todo12_ctoken_rel_change` |

## Linear probes through the full-patch extractor ([TODOs.md](../TODOs.md), TODO 11)

The data comes from `scripts/analysis/probe_features.py --fp32 --out_dir results/analysis/probes_fullpatch` (GPU, via `scripts/analysis/probe_features.slurm`), which writes `results/analysis/probes_fullpatch/<model>.pt`. Probes come from `scripts/analysis/fit_probes.slurm <model> results/analysis/probes_fullpatch`. The figure and `summary_table.md` come from `scripts/analysis/plot_probes_fullpatch.py` (CPU). Models: Refine FullPatch rope3d, Direct FullPatch rope3d, FullPatch a6000_b64, FullPatch d8 (all `best.pth`, raw) and random-init Refine rope3d, at K=5. Data and protocol are TODO 5's.

| Figure | Shows | Script |
|---|---|---|
| `probe_fullpatch_extractor.{png,svg}` | Group accuracy, LOGO position error, rotation/translation error-vector R² and clean-vs-sweep accuracy along enc → extractor blocks (ST/TP; solid = patch mean, dashed = per-image token; grey band) → extractor output → head SA1–SA4; Direct's delta_M as a square | `scripts/analysis/plot_probes_fullpatch.py` |

## Full-patch models: following and activation patching ([TODOs.md](../TODOs.md), TODO 13)

The data comes from `scripts/analysis/follow_miscal.py --eval-loader` and `scripts/analysis/activation_patching.py --eval-loader` (GPU, via `scripts/analysis/todo13.slurm` with `PART=follow|patch`), which write `results/analysis/todo13/{follow_miscal,activation_patching}/<model>.pt`. The figures come from `scripts/analysis/plot_follow_miscal.py --in-dir results/analysis/todo13/follow_miscal --fig-prefix todo13_` and `scripts/analysis/plot_activation_patching.py --in-dir results/analysis/todo13/activation_patching --fig-stem todo13_patch_fraction_recovered`. Models: FP, FP-d8, FP-R3D-refine, FP-R3D-direct (raw weights, K=5, loaded through `load_model_for_evaluation`).

| Figure | Shows | Script |
|---|---|---|
| `todo13_follow_alpha_pos.{png,svg}` | Position following α against miscalibration magnitude. Panels: both external cameras, camera 0 only, camera 1 only, and all four cameras with the same N | `scripts/analysis/plot_follow_miscal.py` |
| `todo13_follow_beta_rot.{png,svg}` | Orientation following β, same panels | same |
| `todo13_follow_closest_share.{png,svg}` | Share of 60/60 predictions nearest the true p, N₀p or N₁p | same |
| `todo13_patch_fraction_recovered.{png,svg}` | Activation patching: share of the clean → heavy60 degradation recovered or caused per component, including the extractor's patch xyz and the Direct delta_M. FP's bars are ill-posed because it has no offline gap | `scripts/analysis/plot_activation_patching.py` |

## exp 4a attention and C-tokens ([TODOs.md](../TODOs.md), TODO 14a)

The data comes from `scripts/analysis/todo14a.slurm` (GPU; `PART=attn|ctok`, `STEP=120000|200000`), which writes `results/analysis/todo14a/`. Models: exp 4a (pre-LN extractor) and exp 3 at the same step, EMA, K=5. Data: all 853 val samples, clean vs heavy60.

| Figure | Shows | Script |
|---|---|---|
| `todo14a_ctoken_rel_change_120k.{png,svg}` | Per-layer relative change under heavy60 of C-tokens, external/wrist FPS tokens and trajectory tokens, exp 4a vs exp 3 at 120k | `scripts/analysis/plot_ctoken_behaviour.py --in-dir results/analysis/todo14a/ctoken_behaviour --fig docs/research/experiments/2026-10-06/figures/todo14a_ctoken_rel_change_120k` |

## exp 4a vs exp 3: following and activation patching ([TODOs.md](../TODOs.md), TODO 14c)

The data comes from `scripts/analysis/follow_miscal.py --eval-loader` and `scripts/analysis/activation_patching.py --eval-loader` (GPU, via `scripts/analysis/todo14c.slurm` with `PART=follow|patch` and `STEP=120000|200000`), which write `results/analysis/todo14c/{follow_miscal,activation_patching}/<model>_<step>.pt`. The figures come from `scripts/analysis/plot_follow_miscal.py --in-dir results/analysis/todo14c/follow_miscal --fig-prefix todo14c_` and `scripts/analysis/plot_activation_patching.py --in-dir results/analysis/todo14c/activation_patching --fig-stem todo14c_patch_fraction_recovered`. Models: exp 4a and exp 3, EMA, K=5, at every checkpoint present.

| Figure | Shows | Script |
|---|---|---|
| `todo14c_follow_alpha_pos.{png,svg}` | Position following α against magnitude: both external cameras, camera 0 only, camera 1 only, and the rigid all-camera control | `scripts/analysis/plot_follow_miscal.py` |
| `todo14c_follow_beta_rot.{png,svg}` | Orientation following β, same panels | same |
| `todo14c_follow_closest_share.{png,svg}` | Share of 60/60 predictions nearest the true p, N₀p or N₁p | same |
| `todo14c_patch_fraction_recovered.{png,svg}` | Activation patching: share of the clean → heavy60 degradation recovered or caused per component, including the per-call delta_M replay | `scripts/analysis/plot_activation_patching.py` |
