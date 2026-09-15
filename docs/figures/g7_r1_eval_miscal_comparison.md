# OOD G7 external-camera miscalibration comparison

| Model | Training | Camera | Calibrated | External: 2° + 2 cm | External: 5° + 5 cm | External: 10° + 10 cm | External: 15° + 15 cm |
|---|---|---|---:|---:|---:|---:|---:|
| R1a | Group miscalibration + sampled miscalibration; no view alignment | OOD G7 | 61.92% | 61.31% | 62.69% | 62.54% | 34.15% |
| R1b | View alignment + group/sampled miscalibration | OOD G7 | 53.08% | 44.92% | 53.38% | 52.54% | 15.85% |
| R1c | View alignment + EE auxiliary supervision + group/sampled miscalibration | OOD G7 | 56.38% | 54.23% | 60.23% | 56.69% | 24.46% |

All cells contain the mean over 13 bimanual tasks and 50 rollouts/task. Each external-camera condition is a newly sampled geometric miscalibration held fixed per evaluation condition; wrist cameras remain calibrated.
