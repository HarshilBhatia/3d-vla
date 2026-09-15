# OOD camera, clean-extrinsics comparison

| Model | Training | Evaluation camera | Test miscal | Mean success |
|---|---|---|---|---:|
| Clean 3D baseline | Clean | OOD G7 | Clean | 61.46% |
| R1a | Base + jitter | OOD G7 | Clean | 61.92% |
| R1b | ΔM + base + jitter | OOD G7 | Clean | 53.08% |
| R1c | ΔM + EE aux + base + jitter | OOD G7 | Clean | 56.38% |
| 2D baseline† | Clean | Held-out G1–G6 group | Clean | 2.31% |

All rows contain 13 task means. The four 3D rows are the G7 clean evaluation; the 2D row is marked separately because its available result uses the held-out G1–G6 group mapping rather than G7.
