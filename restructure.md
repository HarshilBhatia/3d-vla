# Restructuring Notes

This document is a working proposal, not an implementation plan yet. The goal
is a smaller, predictable codebase and a reproducible evaluation system without
interrupting active training runs.

## Current decision

- **Training is frozen** while the current jobs run. Do not change training
  entry points, model/data/trainer code, training configs, or checkpoint paths.
- **Evaluation may be changed now.** The existing evaluation stack can be
  cleaned up, given a stable public interface, and migrated to a consistent
  artifact layout.
- Do not move active artifacts. New evaluation code must keep compatibility
  with existing checkpoint locations and historical result paths until a
  deliberate artifact migration happens.

## Intended repository shape

The end state should have few top-level folders without adding a `src/` wrapper
or a new package-name layer. This is a **move-and-retire** target: it must not
leave a second copy of `modeling/`, `datasets/`, or
`online_evaluation_rlbench/` alongside the new code.

```text
modeling/                       # policy, encoders, schedulers
data/                           # datasets, generation, conversion, processing
training/                       # training runner and trainer implementations
evaluation/                     # online, offline, analysis, protocols
common/                         # small shared utilities only
ops/                            # how code runs, not the code itself
  docker/                       # container definitions and entrypoints
  launch/                       # Slurm/Sky/local thin launchers
  clusters/                     # cluster-specific operational profiles
config/                         # versioned Hydra experiment/config inputs
assets/                         # task definitions, instructions, calibrations
tests/                          # unit, contract, and smoke tests
docs/                           # stable user/developer documentation
```

`online_evaluation_rlbench/` is a historical compatibility path, not the
future evaluation home. Its name conflates an execution mode (online rollout)
with one benchmark (RLBench). The canonical package should be named
`evaluation/`; RLBench belongs beneath it as a backend.

`interp/` is not a top-level area in the target. Its current model-diagnostic
tools move to `evaluation/analysis/` (or `diagnostics/` if that
name proves clearer). Interpretation is evaluation of model behavior; keeping
it beside online and offline evaluation makes inputs, artifact handling, and
testing consistent.

## Whole-repository migration map

Evaluation is only one part of the cleanup. Each current top-level directory
needs one clear responsibility, so a user can tell whether a file is source,
runtime integration, experiment input, research tooling, or generated output.

| Current area | Target | Rule |
|---|---|---|
| `modeling/` | `modeling/` | Policy, encoders, schedulers, and construction only. |
| `datasets/` + `data/` | `data/` | Keep `datasets/`, `generation/`, `processing/`, and conversion as subareas of one data domain. |
| `utils/` | `common/` | Small shared library code only; no command-line entry points. |
| `utils/trainers/` | `training/` | Training runner and trainer implementations. |
| `online_evaluation_rlbench/` | `evaluation/online/rlbench/` | RLBench becomes an online-evaluation backend, not the package name. |
| `interp/` | `evaluation/analysis/` | Model diagnostics and interpretability are evaluation analysis. |
| `scripts/` | `ops/launch/` | Thin local, Slurm, Sky, and helper launchers only. |
| `docker/` | `ops/docker/` | Runtime images and entrypoints only; no copied evaluator implementation. |
| `config/clusters/` | `ops/clusters/` | Operational machine profiles, separated from experiment semantics. |
| remaining `config/` | `config/` | Hydra defaults and reusable experiment/data/model/eval configs. |
| `instructions/` | `assets/` | Immutable tasks, calibration registries, mappings, and benchmark inputs. |
| `tests/`, `docs/` | unchanged | Tests and durable documentation remain obvious project roots. |
| `legacy_may/` | external `archive/` | Move outside the checkout after retention/migration validation. |

### Docker boundary

`ops/docker/online-eval/` should package and invoke the canonical evaluation CLI,
not own a separate evaluation implementation. The container boundary supplies
simulator dependencies; the repository's `evaluation/` package supplies
behavior. This prevents host and container evaluators from drifting.

### Interpretation boundary

Interpretability is `evaluation/analysis/`: it consumes explicit
model/checkpoint/data interfaces and writes artifacts externally, but is never
imported by training or online rollout code. Stable shared primitives belong in
`common/`; exploratory diagnostics live under `evaluation/analysis/`.

### Cluster and launcher boundary

Cluster-specific values belong in `ops/clusters/`. Slurm/Sky launchers under
`ops/launch/` select a cluster profile and call the same Python command; they do
not hard-code distinct checkpoint/result semantics per cluster.

## Evaluation design

### Vocabulary

| Term | Meaning |
|---|---|
| Protocol | Versioned, immutable definition of tasks, conditions, seeds, rollout budget, and metrics. |
| Method | A named policy/checkpoint being evaluated. |
| Condition | A materialized environment/calibration realization. |
| Cell | One protocol × method × condition × task execution. |
| Run | The complete collection of cells produced by one invocation. |
| Backend | The benchmark adapter used for a rollout: PerAct, HiveFormer, Orbital, Bimanual, or Orbital+Bimanual. |

Avoid ambiguous names such as `new`, `final`, `v2`, or date-only run folders.
Use protocol, method, and condition IDs instead.

### Canonical evaluation layout

```text
evaluation/
  cli.py                        # one command-line entry point
  runner.py                     # run/cell orchestration and resume behavior
  checkpoints.py                # checkpoint loading and config validation
  artifacts.py                  # manifests, atomic writes, progress files
  protocols.py                  # protocol loading and validation
  online/
    rlbench/
      runner.py                 # RLBench rollout lifecycle
      environments/             # PerAct, HiveFormer, Orbital, Bimanual adapters
      policies.py               # local and external policy adapters
  offline/
    runner.py                   # dataset-only metrics/loss evaluation
  legacy/                       # forwarding wrappers during migration
```

This is a target layout. The first step is extraction, not a wholesale move:
the existing `utils_with_*_rlbench.py` modules become thin compatibility
adapters only after shared behavior has test coverage. The old
`online_evaluation_rlbench/` path then forwards to `evaluation.online.rlbench`
until all launchers, Docker files, and historical documentation are migrated.

### Canonical flow

```text
protocol + checkpoint + condition
        ↓ validate all inputs
resolve backend and materialized calibration
        ↓ write immutable manifest
execute/resume cells
        ↓ atomically write result files
aggregate/report
```

The evaluator must fail closed when it cannot identify a backend, a
calibration, a protocol, or a compatible checkpoint configuration. It must not
silently fall back to a different benchmark harness.

## Protocols and artifacts

Protocols should be versioned config files, for example:

```text
instructions/eval_protocols/
  peract2_orbital_camera_subset_v1.yaml
  peract2_orbital_checkpoint_ladder_v1.yaml
```

Each protocol defines tasks, variations, seeds, rollout counts, allowed
backends, metrics, and named/materialized conditions. A result ID must never
describe a condition sampled randomly at runtime.

All generated artifacts should ultimately live outside this checkout:

```text
/grogu/datasets/hbhatia/3dfa_artifacts/
  runs/train/<run_id>/
  runs/offline_eval/<run_id>/
  runs/online_eval/<protocol>/<method>/<condition>/<task>/
  jobs/train/<slurm_job_id>/
  jobs/eval/<slurm_job_id>/
  archive/legacy_may/
```

For now, existing paths remain supported. New evaluation launchers can opt in
through `THREEDFA_ARTIFACT_ROOT`; changing defaults waits until the migration
is validated.

## Required manifest fields

Every online-evaluation run should record:

- protocol ID and version;
- method ID and checkpoint path;
- checkpoint checksum and resolved architecture configuration;
- dataset release/path;
- backend identity;
- calibration registry and realization ID, or an explicit clean condition;
- task, variation IDs, seed, and rollout budget;
- resolved Hydra configuration;
- code revision once Git metadata is repaired.

## Migration plan

### Phase 0 — no-risk work now

- Write tests around backend selection, protocol validation, manifests, and
  artifact paths.
- Extract dependency-light helper code from evaluator scripts.
- Add protocol/config schemas and a dry-run validator.
- Leave existing evaluation commands and paths operational.

### Phase 1 — introduce the canonical `evaluation/` package

- Create `evaluation/cli.py` and retain
  `online_evaluation_rlbench/evaluate_policy.py` as a compatibility wrapper.
- Make one current camera-subset evaluation launcher call the new CLI.
- Validate result parity on a small fixed task/seed set.

### Phase 2 — unify launchers

- Replace duplicated Slurm logic with a shared helper under `scripts/helpers/`.
- Make each launcher declare only its protocol, method/checkpoint, resources,
  and optional overrides.
- Route Slurm stdout/stderr through the centralized artifact root for new runs.

### Phase 3 — retire the old name and duplicated harness logic

- Extract common RLBench environment lifecycle, action execution, progress,
  and result writing from the five `utils_with_*_rlbench.py` modules.
- Retain only benchmark-specific adapter behavior in each backend module.
- Update launchers, Docker files, and docs to the canonical entry point.
- Delete the old `online_evaluation_rlbench/` forwarding package only after
  contract and smoke tests pass and no active historical workflow needs it.

### Phase 4 — artifact migration

- Migrate only completed campaigns, with counts/checksums verified first.
- Preserve old paths temporarily with compatibility symlinks if required.
- Remove generated logs/checkpoints from the checkout after the retention
  period, not while any job or watcher references them.

## Explicit non-goals during active training

- No changes to `main.py`, `modeling/`, `datasets/`, data preprocessing, or
  `utils/trainers/`.
- No moving checkpoint directories, active logs, or active result directories.
- No updates to training Slurm scripts or config defaults.
- No Git metadata repair unless separately planned and verified.
