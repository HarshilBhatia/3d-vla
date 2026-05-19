# Checkpoint Configs — Dataset & Noise

Summary of training configuration for each checkpoint in `train_logs/final/`.

---

## Base Models

### `3dfa_multi_cam_G3G4`
| | |
|---|---|
| **Dataset** | `multi_cam_G3G4.zarr` — G3+G4 samples only |
| **Noise** | None |
| **Experiment** | `default` |
| **Epochs** | 3000 |
| **Pretrained from** | Scratch |

---

## Full Multi-Cam Models (trained from scratch on G1–G4)

### `cotrain_medium_12_deltaM`
| | |
|---|---|
| **Dataset** | `multi_cam/train.zarr` — all groups G1–G4 |
| **Noise** | File-based `medium` on G1+G2; G3+G4 clean |
| **Experiment** | `orb_deltaM_full` (`predict_extrinsics=true`, `extrinsics_prediction_mode=delta_m_full`) |
| **Epochs** | 5000 |
| **Pretrained from** | Scratch |

---

## Fine-tuned from `3dfa_multi_cam_G3G4` — No Extrinsics Head

### `3dfa_G3G4_finetune_cotrain_miscal`
| | |
|---|---|
| **Dataset** | `multi_cam/train.zarr` — all groups G1–G4 |
| **Noise** | File-based `medium` on G1+G2; G3+G4 clean |
| **Experiment** | `default` (no extrinsics prediction) |
| **Epochs** | 3000 |
| **Pretrained from** | `3dfa_multi_cam_G3G4/last.pth` |

### `G3G4_finetune_default_miscal`
| | |
|---|---|
| **Dataset** | `multi_cam/train.zarr` — all groups G1–G4 |
| **Noise** | Random on G1+G2: uniformly sampled up to `max_angle=5°`, `max_t=2cm` per sample; G3+G4 clean |
| **Experiment** | `default` (no extrinsics prediction) |
| **Epochs** | 5000 |
| **Pretrained from** | `3dfa_multi_cam_G3G4/last.pth` |

---

## Fine-tuned from `multi_cam_G3G4` — With Extrinsics Head (`orb_deltaM_full`)

### `G3G4_finetune_3dfa`
| | |
|---|---|
| **Dataset** | `multi_cam/train.zarr` — all groups G1–G4 |
| **Noise** | Random on G1+G2: uniformly sampled up to `max_angle=5°`, `max_t=2cm` per sample; G3+G4 clean |
| **Experiment** | `orb_deltaM_full` (`predict_extrinsics=true`, `extrinsics_prediction_mode=delta_m_full`) |
| **Epochs** | 3000 |
| **Pretrained from** | `multi_cam_G3G4/last.pth` |

### `G3G4_finetune_deltaM_eef_final`
| | |
|---|---|
| **Dataset** | `multi_cam/train.zarr` — all groups G1–G4 |
| **Noise** | Random on G1+G2: uniformly sampled up to `max_angle=5°`, `max_t=2cm` per sample; G3+G4 clean |
| **Experiment** | `orb_deltaM_full` (`predict_extrinsics=true`, `extrinsics_prediction_mode=delta_m_full`) |
| **Epochs** | 5000 |
| **Pretrained from** | `multi_cam_G3G4/last.pth` |

---

## Noise Type Reference

| Noise type | Description |
|---|---|
| **None** | All groups see clean data |
| **Random** | Per-sample uniform random rotation+translation noise up to a max magnitude |
| **File-based `medium`** | Fixed pre-computed noise table per group loaded from file — same miscalibration for every sample in that group across all training |
