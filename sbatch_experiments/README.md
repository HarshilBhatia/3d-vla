# ΔM experiment sbatch scripts (RoPE ΔM)

All jobs use `#SBATCH --nodelist=grogu-2-12` and `#SBATCH --time=24:00:00`. Overrides are passed via environment variables; `scripts/rlbench/train_peract2.sh` reads `USE_ROPE_DELTA_M`, `ROPE_LAMBDA_REG`, `WANDB_NAME`, and optionally `ROTATE_PCD`, `ROTATE_ANGLE_DEG`, `ROTATE_AXIS` (ΔM is inside RoPE in position_encodings.py).

## E1 — Does ΔM help?
| Script | Description | Wandb name |
|--------|-------------|------------|
| `e1_m0_baseline.sbatch` | No RoPE ΔM | e1-m0-baseline |
| `e1_m1_deltam_shared.sbatch` | RoPE ΔM shared | e1-m1-deltam-shared |
| `e1_m2_deltam_perhead.sbatch` | RoPE ΔM (shared; per-head needs code) | e1-m2-deltam-perhead |

## E2 — Is SO(d) necessary?
| Script | Description | Note |
|--------|-------------|------|
| `e2_sod.sbatch` | RoPE ΔM = exp(B−B^T) [SO(d)] | Free W / QR / low-rank need code |

## E3 — Synthetic rotation bias
| Script | Description | Note |
|--------|-------------|------|
| `e3_synthetic_bias.sbatch` | With ΔM | Inject R_bias on Q_v (5/15/30/60°) needs code |

## E4 — Calibration perturbation
| Script | Description | Note |
|--------|-------------|------|
| `e4_calibration_perturb.sbatch` | With ΔM | Extrinsic/intrinsic noise in data/eval needs code |

## E5 — Geometry preservation
| Script | Description | Note |
|--------|-------------|------|
| `e5_geometry_logging.sbatch` | ΔM + geometry logs | Trainer already logs frob_A, spectral, det |

## E6 — Attention maps
| Script | Description | Note |
|--------|-------------|------|
| `e6_attention_maps.sbatch` | ΔM run | Entropy/sparsity/KL logging needs code |

## E7 — Ablation where ΔM is applied
| Script | Description | Note |
|--------|-------------|------|
| `e7_ablation_where.sbatch` | Current (cross-modal both) | Q-only / K-only / unimodal need code |

## E8 — Rotation magnitude
| Script | Description | Note |
|--------|-------------|------|
| `e8_rotation_magnitude.sbatch` | ΔM | \|\|A\|\|_F logged as cross_modal_frob_A |

## E9 — Per-head vs shared
| Script | Description | Wandb name |
|--------|-------------|------------|
| `e9_shared.sbatch` | RoPE ΔM shared | e9-shared |
| `e9_perhead.sbatch` | RoPE ΔM (same as shared until per_head wired) | e9-perhead |

## E10 — RoPE + ΔM
| Script | Description | Wandb name |
|--------|-------------|------------|
| `e10_rope_only.sbatch` | No ΔM (RoPE only) | e10-rope-only |
| `e10_rope_and_deltam.sbatch` | RoPE + ΔM | e10-rope-and-deltam |

## E11 — RoPE ΔM regularization λ
| Script | λ | Wandb name |
|--------|---|------------|
| `e11_lambda0.sbatch` | 0 | e11-lambda0 |
| `e11_lambda1e4.sbatch` | 1e-4 | e11-lambda1e-4 |
| `e11_lambda1e3.sbatch` | 1e-3 | e11-lambda1e-3 |
| `e11_lambda1e2.sbatch` | 1e-2 | e11-lambda1e-2 |

## E12 — Generalization
| Script | Description | Note |
|--------|-------------|------|
| `e12_generalization.sbatch` | With ΔM | Train clean; test rotated needs code |

## Run from repo root
```bash
cd /home/lzaceria/mscv/3dvla/3d-vla
sbatch sbatch_experiments/e1_m0_baseline.sbatch
sbatch sbatch_experiments/e1_m1_deltam_shared.sbatch
# etc.
```

Logs go to `logs/<job>-%j.out` and `logs/<job>-%j.err`.
