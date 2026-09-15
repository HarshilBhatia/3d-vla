# ULRS G7 — Complete Results

**Unknown-Lab Robustness Sweep (ULRS), G7 external-camera group.**

## Protocol

- 13 bimanual tasks × 100 rollouts per task.
- External cameras only; wrist cameras remain clean.
- Conditions are fixed registry transforms: clean, 2°/2 cm, 5°/5 cm, and 10°/10 cm.
- Results are per-task mean success rate, reported as percentages. Aggregate rows are unweighted means across the 13 tasks.
- Checkpoints: Base 140k/160k and DeltaM 140k/160k/180k.
- Plan: `instructions/eval_plans/ulrs_g7_100rollouts_v01.json`.
- Raw results: `/grogu/datasets/hbhatia/3dfa_online_eval_100rollouts/ULRS_G7_100rollouts_v01/`.
- Machine-readable cell index: `eval/interp/ulrs_g7_100rollouts_v01_data_index.json`.

## Aggregate success (%)

| Checkpoint | Clean | 2° / 2 cm | 5° / 5 cm | 10° / 10 cm |
|---|---:|---:|---:|---:|
| Base — 140k | 68.1 | 70.0 | 69.2 | 69.6 |
| Base — 160k | 72.4 | 71.5 | 71.3 | 71.9 |
| DeltaM — 140k | 75.5 | 75.4 | 75.1 | 74.7 |
| DeltaM — 160k | 65.9 | 66.3 | 65.8 | 66.1 |
| DeltaM — 180k | 72.1 | 71.3 | 70.5 | 70.5 |

## Task-level success (%)

### Base — 140k

| Task | Clean | 2° / 2 cm | 5° / 5 cm | 10° / 10 cm |
|---|---:|---:|---:|---:|
| dual push buttons | 76.0 | 77.0 | 79.0 | 78.0 |
| handover item | 65.0 | 74.0 | 77.0 | 69.0 |
| handover item easy | 82.0 | 79.0 | 75.0 | 84.0 |
| lift ball | 100.0 | 100.0 | 99.0 | 100.0 |
| lift tray | 91.0 | 95.0 | 94.0 | 94.0 |
| pick laptop | 38.0 | 38.0 | 41.0 | 34.0 |
| pick plate | 75.0 | 73.0 | 76.0 | 68.0 |
| push box | 92.0 | 87.0 | 92.0 | 94.0 |
| put bottle in fridge | 56.0 | 64.0 | 51.0 | 64.0 |
| put item in drawer | 65.0 | 70.0 | 56.0 | 54.0 |
| straighten rope | 3.0 | 2.0 | 3.0 | 2.0 |
| sweep to dustpan | 95.0 | 92.0 | 94.0 | 95.0 |
| take tray out of oven | 47.0 | 59.0 | 63.0 | 69.0 |

### Base — 160k

| Task | Clean | 2° / 2 cm | 5° / 5 cm | 10° / 10 cm |
|---|---:|---:|---:|---:|
| dual push buttons | 82.0 | 78.0 | 82.0 | 82.0 |
| handover item | 74.0 | 74.0 | 78.0 | 73.0 |
| handover item easy | 91.0 | 85.0 | 84.0 | 84.0 |
| lift ball | 100.0 | 99.0 | 98.0 | 96.0 |
| lift tray | 93.0 | 94.0 | 92.0 | 95.0 |
| pick laptop | 35.0 | 32.0 | 42.0 | 23.0 |
| pick plate | 80.0 | 83.0 | 83.0 | 72.0 |
| push box | 88.0 | 89.0 | 85.0 | 90.0 |
| put bottle in fridge | 61.0 | 58.0 | 54.0 | 65.0 |
| put item in drawer | 71.0 | 68.0 | 64.0 | 70.0 |
| straighten rope | 24.0 | 26.0 | 22.0 | 25.0 |
| sweep to dustpan | 98.0 | 97.0 | 98.0 | 98.0 |
| take tray out of oven | 44.0 | 47.0 | 45.0 | 62.0 |

### DeltaM — 140k

| Task | Clean | 2° / 2 cm | 5° / 5 cm | 10° / 10 cm |
|---|---:|---:|---:|---:|
| dual push buttons | 77.0 | 77.0 | 77.0 | 81.0 |
| handover item | 85.0 | 82.0 | 78.0 | 79.0 |
| handover item easy | 82.0 | 85.0 | 86.0 | 80.0 |
| lift ball | 100.0 | 100.0 | 100.0 | 100.0 |
| lift tray | 95.0 | 96.0 | 97.0 | 92.0 |
| pick laptop | 61.0 | 62.0 | 57.0 | 45.0 |
| pick plate | 75.0 | 68.0 | 71.0 | 67.0 |
| push box | 93.0 | 93.0 | 95.0 | 97.0 |
| put bottle in fridge | 57.0 | 56.0 | 49.0 | 56.0 |
| put item in drawer | 52.0 | 51.0 | 60.0 | 54.0 |
| straighten rope | 48.0 | 49.0 | 45.0 | 51.0 |
| sweep to dustpan | 93.0 | 94.0 | 96.0 | 98.0 |
| take tray out of oven | 64.0 | 67.0 | 65.0 | 71.0 |

### DeltaM — 160k

| Task | Clean | 2° / 2 cm | 5° / 5 cm | 10° / 10 cm |
|---|---:|---:|---:|---:|
| dual push buttons | 69.0 | 69.0 | 71.0 | 71.0 |
| handover item | 71.0 | 68.0 | 61.0 | 63.0 |
| handover item easy | 92.0 | 95.0 | 92.0 | 94.0 |
| lift ball | 100.0 | 99.0 | 99.0 | 100.0 |
| lift tray | 94.0 | 93.0 | 91.0 | 99.0 |
| pick laptop | 43.0 | 42.0 | 47.0 | 37.0 |
| pick plate | 41.0 | 35.0 | 33.0 | 36.0 |
| push box | 94.0 | 97.0 | 93.0 | 98.0 |
| put bottle in fridge | 43.0 | 48.0 | 54.0 | 44.0 |
| put item in drawer | 74.0 | 75.0 | 77.0 | 62.0 |
| straighten rope | 15.0 | 17.0 | 18.0 | 22.0 |
| sweep to dustpan | 77.0 | 75.0 | 78.0 | 78.0 |
| take tray out of oven | 44.0 | 49.0 | 41.0 | 55.0 |

### DeltaM — 180k

| Task | Clean | 2° / 2 cm | 5° / 5 cm | 10° / 10 cm |
|---|---:|---:|---:|---:|
| dual push buttons | 81.0 | 77.0 | 78.0 | 74.0 |
| handover item | 64.0 | 57.0 | 60.0 | 61.0 |
| handover item easy | 87.0 | 87.0 | 89.0 | 86.0 |
| lift ball | 98.0 | 100.0 | 98.0 | 99.0 |
| lift tray | 94.0 | 91.0 | 94.0 | 93.0 |
| pick laptop | 29.0 | 35.0 | 35.0 | 25.0 |
| pick plate | 80.0 | 73.0 | 77.0 | 80.0 |
| push box | 97.0 | 95.0 | 97.0 | 98.0 |
| put bottle in fridge | 68.0 | 62.0 | 52.0 | 65.0 |
| put item in drawer | 62.0 | 69.0 | 62.0 | 62.0 |
| straighten rope | 1.0 | 4.0 | 3.0 | 4.0 |
| sweep to dustpan | 98.0 | 99.0 | 99.0 | 98.0 |
| take tray out of oven | 78.0 | 78.0 | 72.0 | 72.0 |
