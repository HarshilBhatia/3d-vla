# OOD G7 external-camera-only miscalibration

| Model | Calibrated | External: 2° + 2 cm | External: 5° + 5 cm | External: 10° + 10 cm |
|---|---:|---:|---:|---:|
| Calibrated-training 3D | 61.46% | 60.38% | 62.08% | 63.46% |
| Miscalibration-trained control | 60.62% | 60.85% | 58.46% | 54.62% |
| View alignment (external) | 60.69% | 59.62% | 64.23% | 63.15% |
| RoPE phase bound $s$ (rad) | 0.000 | 0.088 | 0.220 | 0.439 |
| Scaled phase-bound trend (Base fit) | 60.62% | 59.52% | 57.89% | 55.17% |

All model values are the mean over 13 bimanual tasks and 50 rollouts/task on OOD G7. Only orbital_left and orbital_right receive test-time geometric miscalibration; wrist cameras remain calibrated. The dotted curve is not an independent prediction: it is the phase bound $s=\omega_{max}(2\sin(\alpha/2)\rho+\tau)$ mapped into success units by a non-negative least-squares slope fitted once to the control curve, anchored at the calibrated condition. Here $\omega_{max}=1$ rad/m and $\rho=1.947$ m.
