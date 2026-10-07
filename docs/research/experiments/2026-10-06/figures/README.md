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
