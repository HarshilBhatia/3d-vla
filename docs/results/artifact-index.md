# Evaluation results index

This is the central index for online-evaluation artifacts. The raw rollout
results live on `/grogu` and are not copied into git. Each campaign below is a
directory containing task-level result files, logs/manifests, and (for selected
runs) rollout videos. The paths are absolute on the cluster.

## Existing detailed summaries

- [Video-DeltaM results](video_deltam_results.md): checkpoint ladders, ULRS G7,
  task-level CSV/JSON exports, and historical diagnostics.
- [Extrinsic-miscalibration sweep](results/peract2_extrinsic_miscal_sweep.md):
  clean/base external-only and wrist-only results with task tables.
- [ULRS G7 complete results](../eval/interp/ulrs_g7_100rollouts_v01_results.md):
  Base and DeltaM, 13 tasks × 100 rollouts, clean/2°+2 cm/5°+5 cm/10°+10 cm.
- Machine-readable Video-DeltaM index:
  `docs/results/video_deltam_results_index.json` and
  `docs/results/video_deltam_task_metrics.{json,csv}`.
- ULRS machine-readable index:
  `eval/interp/ulrs_g7_100rollouts_v01_data_index.json`.

## Raw result roots

### 100-rollout campaigns

Root:
`/grogu/datasets/hbhatia/3dfa_online_eval_100rollouts/`

- `ULRS_G7_100rollouts_v01`
- `ULRS_G7_video_deltam_100rollouts_v01`
- `idlt_extreme_task_heldout_group_v01`
- `idlt_extreme_task_seen_group_v01`
- `idlt_scaled_extreme_task_heldout_group_v01`
- `idlt_scaled_extreme_task_heldout_group_v02_5060`
- `idlt_scaled_extreme_task_seen_group_v01`
- `idlt_scaled_extreme_task_seen_group_v02_5060`
- `idlt_task_heldout_group_fixed_v02`
- `idlt_task_heldout_group_fixed_v03_video_k5`
- `idlt_task_heldout_group_v01`
- `idlt_task_seen_group_fixed_v02`
- `idlt_task_seen_group_fixed_v03_video_k5`
- `idlt_task_seen_group_v01`
- `miscal_only_base_task_heldout_group_v01`
- `miscal_only_base_task_seen_group_v01`
- `seen_base_residual_sweep_v01`
- `vdm_rope3d_sweep_task_heldout_group_v01`

### 20-rollout campaigns

Root:
`/grogu/datasets/hbhatia/3dfa_online_eval_20rollouts/`

- G7 stress tests: `g7_extreme_opposite_rope3d_v01`,
  `g7_extreme_opposite_v01`, `g7_extreme_opposite_videopooled_direct_v01`,
  `g7_opposite_camera_sbrs_v01`, `g7_rotation_only_rope3d_v01`,
  `g7_rotation_only_v01`, `g7_rotation_only_videopooled_direct_v01`.
- G7 200 cm runs: `g7_sbrs_200cm_10rollouts_v01`,
  `g7_sbrs_200cm_10rollouts_videos_v01`.
- SBRS checkpoint ladder: `sbrs_checkpoint_ladder_v01_base_s110000` through
  `sbrs_checkpoint_ladder_v01_base_s170000`, and
  `sbrs_checkpoint_ladder_v01_view_align_s110000` through
  `sbrs_checkpoint_ladder_v01_view_align_s200000`.
- Direct/full-patch checkpoint ladders:
  `vid_deltam_direct_checkpoint_ladder_v01_vid-deltaM-direct_s200000` and
  `vid_deltam_fullpatch_checkpoint_ladder_v01_vid-deltaM-fullpatch_s050000`
  through `_s270000`.
- Original Video-DeltaM checkpoint ladder:
  `video_deltam_checkpoint_ladder_v01_video_deltam_s050000`, `_s060000`,
  `_s070000`, `_s080000`, `_s090000`, `_s100000`, `_s120000`, `_s130000`.
- Selected Video-DeltaM runs: `video_deltam_sbrs_best_v01`,
  `video_deltam_sbrs_smoke_80k_v01`, `vdm_rope3d_sweep_task_heldout_group_v01`.

### Incomplete 20-rollout campaigns

Root:
`/grogu/datasets/hbhatia/3dfa_online_eval_20rollouts_incomplete/`

- `vid_deltam_fullpatch_checkpoint_ladder_v01_vid-deltaM-fullpatch_s130000_k1debug`
- `video_deltam_checkpoint_ladder_v01_video_deltam_s110000`
- `video_deltam_checkpoint_ladder_v01_video_deltam_s140000`

These must not be compared as complete checkpoint means.

### Clean/original and 50-rollout campaigns

Roots:

- `/grogu/datasets/hbhatia/3dfa_online_eval_25rollouts/`
  - `peract2_original_clean_v01`
- `/grogu/datasets/hbhatia/3dfa_online_eval_50rollouts/`
  - Baseline/miscalibration: `clean`, `level0`, `external_2deg_2cm`,
    `external_5deg_5cm`, `external_10deg_10cm`, `wrist_2deg_2cm`,
    `wrist_5deg_5cm`, `wrist_10deg_10cm`.
  - G7/R1 runs: `R1a_G7_clean`, `R1b_G7_clean`, `R1c_G7_clean`,
    `R1a_G7_random_2deg_2cm`, `R1a_G7_random_5deg_5cm`,
    `R1a_G7_random_10deg_10cm`, `R1a_G7_random_15deg_15cm`, and corresponding
    `R1b_*`/`R1c_*` directories; also `G7_random_2deg_2cm`,
    `G7_random_5deg_5cm`, `G7_random_10deg_10cm`, `G7_random_15deg_15cm`.
  - Historical R1 fixed-medium/heldout runs: `R1a_clean0`, `R1b_clean0`,
    `R1c_clean0`, `R1a_level0`, `R1b_level0`, `R1c_level0`,
    `R1a_heldout_fixed`, `R1b_heldout_fixed`, `R1c_heldout_fixed`, and the
    corresponding `R1a_fixedmedium_random_{2,5,10,15}deg_{2,5,10,15}cm`,
    `R1b_fixedmedium_random_{2,5,10,15}deg_{2,5,10,15}cm`, and
    `R1c_fixedmedium_random_{2,5,10,15}deg_{2,5,10,15}cm` runs.
  - Other campaigns: `2d_siglip2_best72k`, `2d_siglip2_debug`,
    `new_external_only_G7`, `video_deltam_history_smoke_v01`,
    `video_deltam_task_heldout_g1_g6_v01`,
    `video_deltam_task_heldout_g1_g6_v02_visual_history`,
    `video_deltam_task_heldout_g1_g6_v03_history_parity`.

### Audit and feasibility runs

- `/grogu/datasets/hbhatia/3dfa_online_eval_audit/`
  - Pick-laptop iteration checks: `video_deltam_pick_laptop_iter_v01` through
    `_v04_confirm`.
  - Proprio/history checks: `video_deltam_proprio_ab_causal_v01`,
    `video_deltam_proprio_ab_trainfaithful_v01`,
    `video_deltam_proprio_parity_smoke_v01`.
  - SBRS replay checks: `video_deltam_sbrs_replay_v02` through `_v07_latest_schema_fix`.
- `/grogu/datasets/hbhatia/3dfa_online_eval_probe/vdm_rope3d_2080ti_feasibility_v01`

## Interpretation rules

- Use the detailed markdown/JSON tables above for reported numbers; this page
  is the artifact map, not a replacement for task-level data.
- Do not pool campaigns with different checkpoint, task set, camera group,
  calibration, rollout count, or evaluator revision.
- Treat directories containing `legacy`, `debug`, `smoke`, `incomplete`, or
  `audit` as diagnostics unless the linked detailed report explicitly promotes
  them to a benchmark result.
- To inspect a campaign directly:

  ```bash
  find /grogu/datasets/hbhatia/3dfa_online_eval_20rollouts/<campaign> \
    -maxdepth 2 -type f | sort
  ```
