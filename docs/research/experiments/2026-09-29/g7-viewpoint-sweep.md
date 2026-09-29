# G7 viewpoint sweep (memorisation test)

**Status:** done (eval complete, 22 Sept)
**Week introduced:** [22–29 Sept](../../weekly_logs/2026-09-29.md) (TODO 3)

## Question

Move the cameras physically and keep the extrinsics **correct**. That separates
viewpoint generalisation from calibration error: is there a cliff where the
model stops recognising the scene?

## Camera groups

All training cameras sit on a sphere of r ≈ 1.31 m around (0.25, 0, 0.95).

| Group | az / el / r | Direction to nearest training camera (L / R) |
|---|---|---|
| G7 (anchor) | 43 / 47.9 / 1.33 | 10.5° / 11.9° |
| el72 | 45 / 72 / 1.31 | 10.4° / 10.6° |
| el85 | 45 / 85 / 1.31 | 23.2° / 23.4° |
| az25 | 25 / 47.5 / 1.31 | 14.0° / 16.4° |
| az65 | 65 / 47.5 / 1.31 | 16.3° / 14.9° |
| az90 | 90 / 47.5 / 1.31 | 26.6° / 26.2° |
| r180 | 45 / 47.5 / 1.80 | 11.2° / 12.3° (farther) |
| slight_az56 | 56 / 47.5 / 1.31 | 12.7° / 12.5° |
| slight_el66_r145 | 45 / 66 / 1.45 | 4.7° / 4.9° (farther) |
| extreme_r350 | 45 / 35 / 3.50 | 4.1° / 4.2°, but 2.7× the distance |
| extreme_az150_r250 | 150 / 30 / 2.50 | 68.4° / 69.4° (far side) |

The distance column is direction only and ignores radius.

## Setup

| | |
|---|---|
| Cameras | `instructions/orbital_cameras_g7_sweep_v01.json`, from `scripts/eval/create_g7_viewpoint_sweep.py` |
| Eval plan | `instructions/eval_plans/g7_viewpoint_sweep_v01.json` |
| Arms | Base 160k, VideoPooled 84k, FullPatch best |
| Cells | 11 groups × 6 SBRS tasks × 20 rollouts, calibrated |
| Output | `/grogu/datasets/hbhatia/3dfa_online_eval_20rollouts/g7_viewpoint_sweep_v01/` |
| Array | 3966648 |
| Renders | `docs/research/experiments/2026-09-29/figures/viewpoint-sweep/captures/<task>/<group>.{npz,png}`, from `scripts/capture_g7_sweep_views.py` |
| Rerun | `docs/research/experiments/2026-09-29/figures/viewpoint-sweep/g7_viewpoint_sweep.rrd`, `docs/research/experiments/2026-09-29/figures/viewpoint-sweep/g7_vs_train_groups.rrd`, from `scripts/viz_g7_sweep_rerun.py` |

## Results

Mean %, 6 tasks × 20 rollouts. Each cell has roughly ±4 points of standard error.

| Group | Base 160k | VideoPooled 84k | FullPatch |
|---|---|---|---|
| G7 | 70.0 | 75.0 | 66.7 |
| el72 | 73.3 | 80.8 | 64.2 |
| el85 | 70.0 | 82.5 | 66.7 |
| az25 | 73.3 | 80.8 | 72.5 |
| az65 | 68.3 | 76.7 | 65.0 |
| az90 | **60.0** | 75.0 | 70.8 |
| r180 | 73.3 | 76.7 | 62.5 |
| slight_az56 | 73.3 | 80.8 | 64.2 |
| slight_el66_r145 | 65.8 | 80.0 | 64.2 |
| extreme_r350 | 65.8 | 74.2 | 70.8 |
| extreme_az150_r250 | **59.2** | 74.2 | 65.0 |

## Reading

- **No memorisation cliff.** VideoPooled stays at 74–83% everywhere, including
  3.5 m away and from the far side (az 150°).
- Base loses about 10 points only at az 90° and az 150°, almost all of it on
  `pick_laptop` (40% → 0%).
- FullPatch is flat at 62–73%.
- Likely reason: with calibrated extrinsics the world-frame point cloud barely
  changes when a camera moves — only which surfaces are visible changes — and
  the clean wrist cameras also see the scene. Viewpoint therefore mainly
  affects appearance features.

## Implications

- [no-wrist-4task.md](no-wrist-4task.md) is the real test of how much the
  wrists carry.
- [camframe-rope6d.md](camframe-rope6d.md) removes the world-frame invariance,
  so that is where the size of the training camera distribution should
  actually matter.
