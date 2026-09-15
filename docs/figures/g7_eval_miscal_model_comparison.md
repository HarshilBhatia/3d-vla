# OOD G7 external-camera miscalibration comparison

| Model | Training | Camera | Calibrated | External: 2° + 2 cm | External: 5° + 5 cm | External: 10° + 10 cm | External: 15° + 15 cm |
|---|---|---|---:|---:|---:|---:|---:|
| Calibrated-training 3D | Calibrated training; no view alignment | OOD G7 | 61.46% | 60.38% | 62.08% | 63.46% | 29.08% |
| R1a | Group miscalibration + sampled miscalibration; no view alignment | OOD G7 | 61.92% | 61.31% | 62.69% | 62.54% | 34.15% |
| R1b | View alignment + group/sampled miscalibration | OOD G7 | 53.08% | 44.92% | 53.38% | 52.54% | 15.85% |
| R1c | View alignment + EE auxiliary supervision + group/sampled miscalibration | OOD G7 | 56.38% | 54.23% | 60.23% | 56.69% | 24.46% |
| 2D baseline† | Calibrated training; no view alignment | Held-out G1–G6 | 2.31% | — | — | — | — |

All cells contain the mean over 13 bimanual tasks and 50 rollouts/task. The four 3D rows are OOD-G7; each external-camera condition is a newly sampled geometric miscalibration held fixed per evaluation condition, with wrist cameras calibrated. The available 2D result is not G7 and is included only as a clearly marked reference.
