# Evaluation results summary

This is the single decision-facing summary of the current online evaluations.
Percentages are mean task success rates. Unless noted otherwise, means are
unweighted across tasks and the result cells are from the raw JSON files under
`/grogu/datasets/hbhatia/3dfa_online_eval_*`.

Raw artifact map: [eval_results_index.md](eval_results_index.md). Detailed
task-level tables: [Video-DeltaM results](video_deltam_results.md),
[extrinsic-miscalibration sweep](results/peract2_extrinsic_miscal_sweep.md),
and [ULRS G7 results](../eval/interp/ulrs_g7_100rollouts_v01_results.md).

## 1. Protocols at a glance

| Suite | Tasks / rollout budget | Viewpoint regime | Purpose |
|---|---:|---|---|
| SBRS | 6 representative tasks, generally 20 rollouts/task | Seen G1--G6 groups | Fast checkpoint/model selection and stress tests |
| ULRS G7 | 13 tasks × 100 rollouts | Fully unseen G7 external-camera group | Decision-grade broad robustness |
| IDLT | 13 tasks × 100 rollouts | Task-seen or task-heldout G1--G6 | In-distribution transfer under calibration noise |
| Clean / external / wrist | 13 tasks × 10 rollouts (historical sweep) | Standard heldout groups | Base-model calibration ablations |
| Rope3D sweep | 13 tasks × 100 rollouts | Task-heldout G1--G6 | Direct vs Refine full-patch Rope3D |
| G7 extreme stress | 6 SBRS tasks, 20 rollouts/task | Unseen G7 | Opposing translations and large rotations |

## 2. SBRS checkpoint/model results

SBRS uses six representative tasks: push-box, lift-ball, pick-laptop,
pick-plate, straighten-rope, and handover-item-easy. The standard checkpoint
ladder uses the seen task camera group and the fixed medium calibration plus a
3°/1 cm residual.

### Base vs VideoPooled/Refine checkpoint ladder

| Checkpoint | Base | VideoPooled/Refine |
|---:|---:|---:|
| 50k | — | 75.0% |
| 60k | — | 80.0% |
| 70k | — | 72.5% |
| 80k | — | 72.5% |
| 84k best | — | **83.3%** |
| 90k | — | 69.2% |
| 100k | — | 79.2% |
| 120k | — | 78.3% |
| 130k | — | 79.2% |

The 84k VideoPooled/Refine result is the best completed run in this ladder.
The Base SBRS checkpoint ladder is recorded below because it uses a separate
checkpoint plan and output root.

### Base and DeltaM SBRS ladder

| Step | Base | DeltaM / view-align |
|---:|---:|---:|
| 110k | 61.7% | 65.4% |
| 120k | 63.7% | 74.6% |
| 130k | 60.4% | 67.5% |
| 140k | 66.7% | **77.9%** |
| 150k | 62.1% | 67.5% |
| 160k | **76.7%** | 69.6% |
| 170k | 66.7% | 63.3% |
| 180k | — | 68.8% |
| 190k | — | 74.2% |
| 200k | — | 70.4% |

### Full-patch Video-DeltaM and Direct VideoPooled

| Model/checkpoint | SBRS mean | Notes |
|---|---:|---|
| Full-patch, 50k | 33.3% | Early checkpoint |
| Full-patch, 100k | 32.5% | Early checkpoint |
| Full-patch, 150k | 33.3% | Early checkpoint |
| Full-patch, 160k | 73.3% | First strong full-patch checkpoint |
| Full-patch, 170k | **76.7%** | Best in sampled ladder through 270k |
| Full-patch, 200k | 58.3% | Later regression |
| Full-patch, 270k | 75.8% | Later recovery |
| Direct VideoPooled, 200k | 68.3% | Six-task direct checkpoint run |

The early low full-patch values are retained as historical checkpoint results;
they should not be mixed with the later fixed-evaluator results without their
checkpoint and code provenance.

## 3. ULRS G7: 13 tasks × 100 rollouts

G7 is a fully unseen external-camera group. Wrist cameras stay clean. These are
the broadest and most statistically stable results in the repository.

| Checkpoint | Clean | 2° / 2 cm | 5° / 5 cm | 10° / 10 cm |
|---|---:|---:|---:|---:|
| Base 140k | 68.1% | 70.0% | 69.2% | 69.6% |
| Base 160k | 72.4% | 71.5% | 71.3% | 71.9% |
| DeltaM 140k | **75.5%** | **75.4%** | **75.1%** | **74.7%** |
| DeltaM 160k | 65.9% | 66.3% | 65.8% | 66.1% |
| DeltaM 180k | 72.1% | 71.3% | 70.5% | 70.5% |

The complete per-task tables are in the linked ULRS report; the raw cell index
is `eval/interp/ulrs_g7_100rollouts_v01_data_index.json`.

### ULRS clean / no-perturbation task results

These are the exact Base clean G7 results that were easy to miss when looking
only at the extreme-perturbation campaigns. They use the `calibrated` cells in
`ULRS_G7_100rollouts_v01`, 13 tasks × 100 rollouts/task, with no external
camera perturbation.

| Task | Base 140k | Base 160k |
|---|---:|---:|
| dual push buttons | 76% | 82% |
| handover item | 65% | 74% |
| handover item easy | 82% | 91% |
| lift ball | 100% | 100% |
| lift tray | 91% | 93% |
| pick laptop | 38% | 35% |
| pick plate | 75% | 80% |
| push box | 92% | 88% |
| put bottle in fridge | 56% | 61% |
| put item in drawer | 65% | 71% |
| straighten rope | 3% | 24% |
| sweep to dustpan | 95% | 98% |
| take tray out of oven | 47% | 44% |
| **mean** | **68.1%** | **72.4%** |

The same no-perturbation task-level table for DeltaM 140k/160k/180k, plus all
2°/2 cm, 5°/5 cm, and 10°/10 cm levels, is preserved in the complete
[ULRS task-level report](../eval/interp/ulrs_g7_100rollouts_v01_results.md).

## 4. IDLT: in-distribution lab transfer

These runs use the 13-task suite and 100 rollouts/task. “Seen” means the target
task uses its training camera group; “heldout” uses a different G1--G6 group.

| Campaign / condition | Task-seen | Task-heldout |
|---|---:|---:|
| Original IDLT v01 (historical) | 57.1% | 56.3% |
| Fixed calibration v02 | 61.1% | 59.7% |
| Fixed calibration, Video K=5 v03 | **79.5%** | **78.4%** |
| Extreme 15/15 and 20/20 | 70.0% | 67.3% |
| Scaled extreme 30/30 and 40/40 | 62.2% | 60.0% |
| Scaled extreme 50/50 and 60/60 | 54.8% | 52.1% |

The v03 K=5 rows are the corrected history-order evaluator. The earlier v01/v02
rows are retained for regression diagnosis, not as the final model comparison.

## 5. Clean and camera-subset calibration ablations

These are the original clean/base sweep: 13 tasks × 10 rollouts/task. External
means perturb only orbital cameras; wrist means perturb only wrist cameras.

**Checkpoint:**
`/grogu/datasets/hbhatia/3dfa_egress_extracted/3dfa_egress_20260820/checkpoints/peract2_orbital_nhist3_b200_best.pth`
This checkpoint was trained clean, with no miscalibration/noise training and no
DeltaM head. The evaluation changed only the camera extrinsics used for
depth-to-point-cloud reconstruction. This is separate from the ULRS G7
checkpoint campaign below, whose checkpoints were trained under different
training regimes.

| Condition | Mean success |
|---|---:|
| Clean, all cameras | 66.9% |
| External 2° + 2 cm | 70.8% |
| External 5° + 5 cm | 69.2% |
| External 10° + 10 cm | 61.5% |
| Wrist 2° + 2 cm | 36.2% |
| Wrist 5° + 5 cm | 16.2% |
| Wrist 10° + 10 cm | 11.5% |

Raw campaign directories are `clean`, `external_2deg_2cm`,
`external_5deg_5cm`, `external_10deg_10cm`, `wrist_2deg_2cm`,
`wrist_5deg_5cm`, and `wrist_10deg_10cm` under
`/grogu/datasets/hbhatia/3dfa_online_eval_50rollouts/`. The clean run was job
`3935835`; the noisy runs were jobs `3935802`, `3935658`, `3935833` (external)
and `3935803`, `3935659`/`3935703`, `3935834` (wrist).

### Newer retrained Base: no random perturbation noise during training

The newer retrained checkpoint is:

```text
train_logs/PerAct2/peract2_orbital_miscal_only_base/best.pth
```

Its training config uses a persistent per-group medium calibration bias but
**no sampled perturbation noise** (`perturbation_noise_rot_deg=null`,
`perturbation_noise_trans_m=null`). It was then evaluated with residual
external-camera noise at 5/5, 10/10, 15/15, and 20/20. These are the campaigns
`miscal_only_base_task_seen_group_v01` and
`miscal_only_base_task_heldout_group_v01` (13 tasks × 100 rollouts/task).

| Residual noise | Task-seen group | Task-heldout group |
|---|---:|---:|
| Clean residual | 73.0% | 72.2% |
| 5° / 5 cm | 73.1% | 71.7% |
| 10° / 10 cm | 72.1% | 69.7% |
| 15° / 15 cm | 67.7% | 66.9% |
| 20° / 20 cm | 66.4% | 66.5% |

The newer original-clean campaign (`peract2_original_clean_v01`) is 67.7% over
13 tasks × 25 rollouts. It is a separate rollout budget and should not be
merged with the 10-rollout sweep.

## 6. G7 extreme stress tests

All rows below use six SBRS tasks and 20 rollouts/task on unseen G7.

| Evaluation | Model | Mean success |
|---|---|---:|
| Opposing translation, 100/200/500 cm | Base + DeltaM + VideoPooled-Refine + FullPatch | 58.8% |
| Opposing translation, 100/200/500 cm | Direct/Refine Rope3D | 61.9% |
| Opposing translation, 100/200/500 cm | Direct VideoPooled | 69.4% |
| Rotation-only, 20/90/180/270° | Base + DeltaM + VideoPooled-Refine + FullPatch | 64.7% |
| Rotation-only, 20/90/180/270° | Direct/Refine Rope3D | 63.5% |
| Rotation-only, 20/90/180/270° | Direct VideoPooled | 68.8% |
| Opposite-camera translation, no rotation | SBRS comparison | 73.0% |

The aggregate opposing-translation rows combine the tested distance levels;
the task/condition JSON files should be used when comparing individual 100,
200, and 500 cm cells.

### Per-model extreme translation results

| Model | 100 cm | 200 cm | 500 cm |
|---|---:|---:|---:|
| Base | 41.7% | 42.5% | 52.5% |
| DeltaM | 57.5% | 57.5% | 37.5% |
| VideoPooled-Refine | 71.7% | **78.3%** | 68.3% |
| FullPatch Video | 65.8% | 69.2% | 62.5% |
| Direct Rope3D | 65.8% | 72.5% | 60.0% |
| Refine Rope3D | 56.7% | 53.3% | 63.3% |
| Direct VideoPooled | **73.3%** | 70.0% | 65.0% |

### Per-model extreme rotation results

| Model | 20° | 90° | 180° | 270° |
|---|---:|---:|---:|---:|
| Base | 77.5% | 51.7% | 48.3% | 56.7% |
| DeltaM | 79.2% | 53.3% | 49.2% | 49.2% |
| VideoPooled-Refine | 77.5% | 77.5% | 72.5% | 79.2% |
| FullPatch Video | 68.3% | 65.8% | 67.5% | 62.5% |
| Direct Rope3D | 67.5% | 66.7% | 63.3% | 65.0% |
| Refine Rope3D | 61.7% | 60.0% | 60.0% | 64.2% |
| Direct VideoPooled | 75.8% | 66.7% | 63.3% | 69.2% |

### Opposite-camera translation (no rotation)

This separate G7 SBRS run uses 10/20/40/60 cm opposing translations.

| Model | Clean | 10 cm | 20 cm | 40 cm | 60 cm |
|---|---:|---:|---:|---:|---:|
| Base | 70.8% | 78.3% | 79.2% | 70.0% | 49.2% |
| DeltaM | 78.3% | **84.2%** | 79.2% | 74.2% | 68.3% |
| VideoPooled-Refine | 77.5% | 79.2% | 81.7% | **81.7%** | **80.0%** |
| FullPatch Video | 70.0% | 68.3% | 60.8% | 66.7% | 62.5% |

## 7. Rope3D task-heldout sweep

The campaign is `vdm_rope3d_sweep_task_heldout_group_v01`, with Direct and
Refine full-patch Rope3D checkpoints and clean through 60/60 calibration
conditions. The currently materialized raw result set is incomplete: only two
cells are present in the 100-rollout root, so its observed mean (95.0%) is not
a valid full-suite benchmark. Do not compare it against the complete ULRS or
IDLT tables until all planned cells exist.

## 8. Diagnostics and excluded results

- `*_smoke`, `*_debug`, and `*_audit` campaigns are execution/provenance checks,
  not benchmark estimates.
- `3dfa_online_eval_20rollouts_incomplete/` contains partial checkpoint cells;
  it is excluded from means.
- Legacy camera-readout/history campaigns are retained to explain regressions,
  but are not comparable to the corrected evaluator.
- The full raw inventory and exact paths are in
  [eval_results_index.md](eval_results_index.md).

## 9. Per-task G7 extreme graph

The six-panel task-level graph below combines the three G7 extreme campaigns
so that every method is shown in the same figure. Each panel is one SBRS task;
solid curves are opposing translation at 100/200/500 cm and dashed curves are
rotation-only at 20/90/180°. The 270° condition is intentionally omitted.

![G7 extreme task-level results](figures/g7_extreme_task_graphs.png)

The figure is generated reproducibly by
[`scripts/plot_g7_extreme_task_graphs.py`](../scripts/plot_g7_extreme_task_graphs.py)
from the raw JSON/manifest cells. Only the PNG is emitted; no SVG or PDF is
generated.
