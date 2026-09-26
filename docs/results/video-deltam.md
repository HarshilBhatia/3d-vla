# Video-DeltaM results

This is the index of all Video-DeltaM training and online-evaluation evidence
currently present in the workspace.  It deliberately separates decision-grade
results from smoke tests, historical pre-parity results, and incomplete runs.
Regenerate the JSON exports with:

```bash
python scripts/eval/export_video_deltam_results.py
```

Machine-readable companions:

- `docs/results/video_deltam_results_index.json`: campaign metadata, raw roots,
  aggregate summaries, and checkpoint/plan provenance.
- `docs/results/video_deltam_task_metrics.json`: one row per raw task result,
  including campaign status/comparability and the absolute result and manifest
  paths.  This contains **all** evidence below, including buggy historical
  evaluations and incomplete cells; filter on `result_status` and
  `comparability` before making comparisons.
- `docs/results/video_deltam_task_metrics.csv`: the same 327 task-level rows
  in a flat form for spreadsheets, pandas, or plotting tools.

## Per-task result catalogue

The complete per-task values are intentionally kept in the JSON/CSV rather
than duplicating 327 rows in this narrative document.  Every row has
`campaign_id`, `campaign_status`, `result_status`, `comparability`, `task`, `mean_success`,
`method`, `viewpoint`, `calibration`, `result_path`, and `manifest_path`.

| Campaign | Per-task rows | Interpretation |
|---|---:|---|
| `ulrs_g7_100rollouts` | 156 | Decision-grade: 13 tasks x 4 calibration levels x 3 checkpoints. |
| `seen_base_ladder` plus 84k / 80k campaigns | 64 | Seen-group checkpoint selection; 110k and 140k are incomplete. |
| `task_heldout_v02` | 38 | Incomplete intermediate visual-history revision. |
| `task_heldout_v03` | 14 | Incomplete historical parity attempt. |
| history / proprio diagnostics | 4 | Smoke or single-task diagnostics; not benchmark results. |

For a fast visualisation entrypoint, load
`docs/results/video_deltam_task_metrics.csv`, then use `campaign_id` and
`calibration` as facets and `mean_success` as the metric.  Join to
`video_deltam_results_index.json` on `campaign_id` only if you need campaign
plans, raw-root metadata, or aggregate summaries.

## Model lineage

| Label | Checkpoint / run | Status | Notes |
|---|---|---|---|
| Original Video-DeltaM | `train_logs/PerAct2/peract2_orbital_video_deltam_external_20260910/last.pth` | completed 100k | Historical K5-visual model; used by the early smoke, proprio A/B, and heldout revisions. |
| K5v/K3p warm-start run | `train_logs/PerAct2/peract2_orbital_video_deltam_external_warmstart_k5v_k3p_a5000_resume/` | completed 100k | W&B `t8ro0crq`; `best.pth` is step 84k and `interm_step_100000.pth` is the 100k snapshot. |
| K5v/K3p continuation | `train_logs/PerAct2/peract2_orbital_video_deltam_external_last100k_resume50k/` | completed 150k | W&B `7866jrs4`; 110k--140k intermediates exist. |

`best.pth` is selected by heldout trajectory position accuracy at 1 cm
(`val traj_pos_acc_001`, stored negated as `best_loss`); it is not chosen by
online success rate.

## Evaluation conventions

- **Training-style seen group**: six representative tasks, 20 rollouts/task,
  a camera group seen for that task during training, external-only medium fixed
  group bias plus one materialized 3 degree / 1 cm residual; wrists clean.
- **Task-heldout G1--G6**: each task runs on its assigned held-out group.
  Historical revisions are retained below but are not comparable to the final
  K5v/K3p protocol unless explicitly labelled otherwise.
- **ULRS G7**: all 13 tasks, 100 rollouts/task, fully unknown G7 viewpoint,
  with calibrated / 2deg+2cm / 5deg+5cm / 10deg+10cm fixed external-only
  calibration realizations.
- Final K5v/K3p evaluator input is K=5 visual frames (oldest to current) and
  K=3 proprio states (past to current).

## Decision-grade results

### Training-style seen-group checkpoint ladder

All cells are six tasks x 20 rollouts.  110k and 140k are incomplete and must
not be compared as full means.

| Checkpoint | Mean success | Completion | Raw campaign root |
|---|---:|---:|---|
| 50k | 75.0% | 6/6 | `/grogu/datasets/hbhatia/3dfa_online_eval_20rollouts/video_deltam_checkpoint_ladder_v01_video_deltam_s050000` |
| 60k | 80.0% | 6/6 | `/grogu/datasets/hbhatia/3dfa_online_eval_20rollouts/video_deltam_checkpoint_ladder_v01_video_deltam_s060000` |
| 70k | 72.5% | 6/6 | `/grogu/datasets/hbhatia/3dfa_online_eval_20rollouts/video_deltam_checkpoint_ladder_v01_video_deltam_s070000` |
| 80k | 72.5% | 6/6 | `/grogu/datasets/hbhatia/3dfa_online_eval_20rollouts/video_deltam_checkpoint_ladder_v01_video_deltam_s080000` |
| best 84k | **83.3%** | 6/6 | `/grogu/datasets/hbhatia/3dfa_online_eval_20rollouts/video_deltam_sbrs_best_v01` |
| 90k | 69.2% | 6/6 | `/grogu/datasets/hbhatia/3dfa_online_eval_20rollouts/video_deltam_checkpoint_ladder_v01_video_deltam_s090000` |
| 100k | 79.2% | 6/6 | `/grogu/datasets/hbhatia/3dfa_online_eval_20rollouts/video_deltam_checkpoint_ladder_v01_video_deltam_s100000` |
| 110k | 98.3% | 3/6 | incomplete; do not compare |
| 120k | 78.3% | 6/6 | `/grogu/datasets/hbhatia/3dfa_online_eval_20rollouts/video_deltam_checkpoint_ladder_v01_video_deltam_s120000` |
| 130k | 79.2% | 6/6 | `/grogu/datasets/hbhatia/3dfa_online_eval_20rollouts/video_deltam_checkpoint_ladder_v01_video_deltam_s130000` |
| 140k | 95.0% | 1/6 | incomplete; do not compare |

The selected 84k checkpoint is the strongest completed result on this
six-task training-style condition.

### ULRS G7: 13 tasks x 100 rollouts

All 156 cells completed.  This is the cleanest broad robustness result because
each table entry averages all 13 tasks and every task has 100 rollouts.

| Checkpoint | Calibrated | 2deg / 2cm | 5deg / 5cm | 10deg / 10cm |
|---|---:|---:|---:|---:|
| best 84k | **78.5%** | 78.3% | 78.0% | 78.1% |
| 100k | 77.9% | 78.0% | **78.8%** | 78.5% |
| 130k | 76.0% | 75.2% | 75.5% | 74.6% |

Raw root: `/grogu/datasets/hbhatia/3dfa_online_eval_100rollouts/ULRS_G7_video_deltam_100rollouts_v01`.
Plan: `instructions/eval_plans/ulrs_g7_video_deltam_100rollouts_v01.json`.

## Diagnostics and historical results

| Campaign | Result | Interpretation | Raw root |
|---|---:|---|---|
| Visual-history smoke | push-box 100% (1 rollout) | Confirms the K=5 visual-history path executes; not a performance estimate. | `/grogu/datasets/hbhatia/3dfa_online_eval_50rollouts/video_deltam_history_smoke_v01` |
| Proprio A/B causal | handover-item-easy 90% (10 rollouts) | Single-task temporal-order diagnostic only. | `/grogu/datasets/hbhatia/3dfa_online_eval_audit/video_deltam_proprio_ab_causal_v01` |
| Proprio A/B train-faithful | handover-item-easy 0% (10 rollouts) | Historical diagnostic; superseded by the final evaluator parity fix. | `/grogu/datasets/hbhatia/3dfa_online_eval_audit/video_deltam_proprio_ab_trainfaithful_v01` |
| Proprio parity smoke | push-box 100% (1 rollout) | Execution smoke only. | `/grogu/datasets/hbhatia/3dfa_online_eval_audit/video_deltam_proprio_parity_smoke_v01` |
| Heldout v02 | clean 40.7% (12/13) | Intermediate visual-history attempt; incomplete. | `/grogu/datasets/hbhatia/3dfa_online_eval_50rollouts/video_deltam_task_heldout_g1_g6_v02_visual_history` |
| Heldout v03 | clean 20.6% (12/13) | Historical parity attempt; incomplete, and non-clean cells are missing. | `/grogu/datasets/hbhatia/3dfa_online_eval_50rollouts/video_deltam_task_heldout_g1_g6_v03_history_parity` |

The JSON exports preserve every available task-level value and raw file path,
including the incomplete and historical campaigns above.  New plots should
filter on `comparability` and require the relevant expected task count before
computing a mean.
