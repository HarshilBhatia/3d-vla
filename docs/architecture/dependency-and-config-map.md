# Dependency and configuration map

**Snapshot date:** 2026-09-15. This is an as-is map, not the desired target.
The target and migration rules are in
[ADR-001](adr-001-repository-boundaries.md).

## Runtime entry points

| Entry point | Current path | Responsibility | Immediate dependencies |
|---|---|---|---|
| Training / offline validation | `main.py` | Hydra composition, DDP startup, dataset/model/trainer selection | `datasets`, `modeling.policy`, `utils.trainers`, `utils.hydra_utils` |
| Online RLBench evaluation | `evaluation/cli.py` | Compose config, load checkpoint, invoke online runner | `evaluation.checkpoints`, `evaluation.online`, `utils.hydra_utils` |
| Legacy online evaluation | `online_evaluation_rlbench/evaluate_policy.py` | Compatibility CLI forwarder | `evaluation.cli` |
| Offline calibration sweep | `evaluation/offline/sweep_miscal_loss.py` | Dataset-only loss sweep | `datasets`, `modeling`, `utils.data_preprocessors`, `utils.depth2cloud`, `utils.trainers` |
| RLBench data generation/conversion | `data/generation/`, `data/processing/` | Simulator collection and Zarr conversion | `data`, selectively `utils.common_utils` |

## Package ownership today

| Area | What it actually owns | Main incoming users | Architectural finding |
|---|---|---|---|
| `modeling/` | Encoders, policies, noise schedulers, attention/RoPE utilities, model construction | training, evaluation | Mostly coherent; `denoise_actor_3d.py` imports `utils.pytorch3d_transforms`, which should become a dependency-light geometry module. |
| `datasets/` | Zarr datasets, samplers, dataset registry | training, evaluation analysis | Split from the rest of the data domain. |
| `data/` | Generation, orbital simulator support, Zarr conversion/processing | evaluation orbital backends, scripts | Owns both data pipeline and simulator creation utilities. |
| `utils/` | Config, EMA, scheduler, geometry compatibility paths | nearly all runtime paths | Trainer implementations moved to `training/`; remaining shared utilities are the next cleanup area. |
| `evaluation/` | Canonical CLI, checkpoint compatibility, artifacts, protocols, planning, online/offline runners, analysis | legacy wrappers and evaluation commands | Migration is materially underway. |
| `online_evaluation_rlbench/` | Legacy module/CLI import paths | old launchers and docs | Already forwards most backend modules to `evaluation`; keep only as a temporary compatibility package. |
| `scripts/` | Slurm/Sky/local launchers plus plotting, conversion, and diagnostics | operators | Needs a future split between `ops/launch/` and code-owned CLIs. |
| `instructions/` | task text, calibration/noise realizations, task mappings, eval plans | data, eval, launchers | Immutable inputs; future `assets/` owner. |

## Observed internal dependency map

```text
main.py
  ├── datasets
  ├── modeling.policy
  ├── utils.hydra_utils
  └── utils.trainers
        ├── datasets.samplers
        ├── modeling.encoder / modeling.policy.construction
        └── utils.{data_preprocessors,depth2cloud,ema,schedulers}

evaluation.cli
  ├── evaluation.checkpoints ── modeling.policy + utils.hydra_utils
  └── evaluation.online.runner
        └── evaluation.online.rlbench.backends
              ├── modeling.encoder.text
              ├── data.generation.orbital.*       (orbital backends only)
              └── utils.{data_preprocessors,depth2cloud}

evaluation.offline / evaluation.analysis
  ├── datasets
  ├── modeling
  └── utils.{data_preprocessors,depth2cloud,hydra_utils,trainers}

data.processing
  └── utils.common_utils                         (small legacy edge)
```

### Boundary violations to remove

1. **Resolved in the first migration slice:** evaluation analysis/offline code
   previously imported `utils.trainers.base` for `base_collate_fn` and
   `relative_to_absolute`. They now live in `data.batch`; the old trainer path
   is a compatibility re-export only.
2. Evaluation backends import data-generation orbital environment classes.
   This is acceptable only after those classes are explicitly designated as a
   reusable `data.simulator.rlbench.orbital` integration rather than a
   generation-only implementation.
3. Model code imports a utility geometry transform. Move it to `common.geometry`
   (if tensor-only) or `data.geometry` (if observation/data-specific) and make
   the ownership explicit.
4. `utils.hydra_utils` is used by training and evaluation, so it is a shared
   configuration boundary, not a generic utility.

## Current configuration inventory

The root config is a single resolved namespace. Hydra global groups are
normalized in `utils/hydra_utils.py`, so CLI `data=x`, `experiment=x`, and
`miscal=x` select files under `config/` while merging values globally.

| Current group | Files / examples | Current responsibility | Target domain |
|---|---|---|---|
| Root | `config/config.yaml` | Defaults for every concern | Split across all domains |
| Data | `config/data/{full,full_nfs,orbital,orbital_peract2_nfs,...}.yaml` | dataset identity, paths, instructions, some `train_iters` | `data/`; training duration moves to `training/` |
| Experiments | `config/experiment/*.yaml` | model choices, optimizer, data identity, logging, miscalibration, checkpoint evals | minimal composed experiment layer |
| Miscalibration | `config/miscal/*.yaml` | sampled/group/cotrain noise | `calibration/` |
| Cluster | `config/clusters/*.env` | machine paths/environment variables | `runtime/` or `ops/clusters/` |
| Evaluation plans | `instructions/eval_plans/*.json` | method/checkpoint/task/calibration rollout recipes | `assets/protocols/` |

### Root config domains currently interleaved

| Domain | Representative keys | Proposed owner |
|---|---|---|
| Data and loader | `dataset`, `*_data_dir`, `*_instructions`, `batch_size`, `num_workers`, cache/sampler controls | `config/data` + `config/training/loader` |
| Training and logging | `lr`, `wd`, `train_iters`, checkpoint cadence, EMA, W&B, benchmark | `config/training` |
| Model architecture | backbone, token/attention dimensions, FPS, RoPE, denoise settings, Video-DeltaM | `config/model` |
| Geometry/calibration | view-alignment, DeltaM legacy aliases, miscalibration, EE aux | `config/model/view_alignment` + `config/calibration` |
| Online evaluation | task, rollout controls, output paths, calibration registry, videos | `config/evaluation` |
| Runtime/cluster behavior | DDP-visible paths, local-vs-container values, artifact roots | `config/runtime` and `ops/clusters` |

### Legacy/public vocabulary boundary

`utils/hydra_utils.py` currently maps public keys to legacy runtime keys. This
is intentional compatibility logic and should be centralized, not copied into
trainers, evaluators, launchers, or model classes.

| Public name | Legacy runtime name |
|---|---|
| `view_align_mode` | `predict_extrinsics` + `extrinsics_prediction_mode` |
| `view_align_cameras` | `delta_m_camera_ids` |
| `layerwise_view_align` | `dynamic_rope_from_camtoken` |
| `miscal_cameras` | `miscal_camera_ids` |
| `group_miscal_level` | `orbital_miscal_noise_level` |
| `sampled_miscal_max_*` | `miscal_max_*` |
| `ee_aux*` | `predict_ee_aux`, `lambda_aux`, `ee_aux_cam_ids` |
| `causal_cam_history*` | `video_deltam*` |

New work should write the public vocabulary. Old keys remain loadable while
checkpoint and launcher compatibility requires them.

## Existing safety mechanisms worth preserving

- `modeling/policy/construction.py` builds model kwargs through one shared
  path and checks constructor completeness.
- `evaluation/checkpoints.py` overlays only model-owned checkpoint config;
  evaluation-owned fields are protected by `EVALUATION_RUNTIME_KEYS`.
- Evaluation has tests for artifact writes, calibration routing, checkpoint
  loading, planning/protocol validation, runner behavior, and RLBench routing.
- `write_experiment_manifest` and `write_eval_manifest` capture resolved
  public/legacy configuration information beside artifacts.

## Migration backlog, ordered by risk

| Priority | Slice | Risk | Exit criterion |
|---:|---|---|---|
| 0 | Add import-boundary and config-composition tests | low | Tests prevent new `modeling → trainer/evaluation` and `evaluation → trainer-internal` imports. |
| 1 | Complete legacy evaluator forwarding | low | One canonical evaluation CLI/backend implementation; legacy package has no behavior. |
| 2 | Extract public batch/action helpers from trainer | **complete** | `data.batch` owns the helpers; evaluation imports no `utils.trainers.base` symbols. |
| 3 | Move preprocessors/depth geometry to `data/` | **complete** | Implementations now live under `data.preprocessing` and `data.geometry`; old `utils` packages are forwarders. |
| 4 | Move trainer implementation to `training/` | **complete** | `main.py` imports `training`; `utils.trainers` is a compatibility forwarder and metrics live in `common.metrics`. |
| 5 | Introduce domain config groups/schema | medium | New experiment composes domains without modifying root defaults. |
| 6 | Componentize policy variants | high | Fixed-batch and checkpoint loading parity pass for baseline and Video-DeltaM checkpoints. |
| 7 | Relocate ops/assets/generated artifacts | medium | No active launcher/docs use old locations; retention/checksum plan approved. |

## Non-source artifacts

- `eval/interp/` contains committed result reports and figures. Keep these as
  curated research artifacts until a deliberate artifact-retention policy
  moves them.
- `logs/` has tracked `.gitkeep` files; runtime logs, `train_logs/`, and
  `eval_logs/` are operational outputs and should not become library package
  roots.
- `legacy_may/` is currently untracked. Treat it as an external archive
  candidate; do not move or delete it as part of source restructuring.
