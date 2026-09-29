# ADR-001: Repository boundaries and incremental migration

- **Status:** accepted
- **Date:** 2026-09-15
- **Owners:** 3DFA maintainers
- **Supersedes:** the directory-shape proposal in `restructure.md` where this
  document is more specific

## Context

3DFA has evolved through active research, benchmark integration, and cluster
operations.  The resulting code works, but package names no longer reliably
describe ownership:

- `utils/` contains configuration, trainer implementations, geometry,
  preprocessing, metrics, and general helpers.
- `datasets/` and `data/` jointly own the data domain.
- `evaluation/` is the canonical evaluator being introduced, while
  `online_evaluation_rlbench/` remains a historical import/CLI path.
- `config/config.yaml` is one flat schema consumed by training, offline
  evaluation, online evaluation, model construction, calibration, and runtime
  infrastructure.
- experiment variants are primarily boolean and legacy-key combinations inside
  the main policy classes.

This makes it easy for a new flag or helper to cross a boundary silently.  The
previous `image_space_sampling` train/eval divergence is the concrete failure
mode we must design against.

## Decision

The repository will use the following dependency direction:

```text
                 training
                ↙    ↓
common  ←  modeling  ←  data
  ↑          ↑          ↑
  └──────── evaluation ─┘

ops and assets are runtime inputs; they are not imported as application code.
```

More precisely:

1. `common/` contains small, dependency-light shared primitives only.
2. `modeling/` contains reusable model components and depends only on
   `common/` plus third-party ML libraries. It must not import simulator,
   dataset, trainer, protocol, artifact, or cluster code.
3. `data/` owns dataset loading, schemas, preprocessing, geometric conversion,
   generation, and conversion. It may depend on `common/`, but not on
   `training/` or `evaluation/`.
4. `training/` owns training orchestration, validation, checkpoints, and
   logging. It may depend on `modeling/`, `data/`, and `common/`.
5. `evaluation/` owns protocols, checkpoint compatibility, artifacts, online
   and offline execution, and analysis. It may depend on `modeling/`, `data/`,
   and `common/`; it does not import trainer internals.
6. `ops/` contains Docker images, cluster profiles, and thin launchers. A
   launcher selects a config/protocol and invokes a canonical Python CLI; it
   does not duplicate model or evaluator behavior.
7. `assets/` contains versioned, immutable benchmark inputs: instructions,
   task mappings, camera definitions, calibration registries, and protocols.

The target top-level layout is therefore:

```text
modeling/     data/       training/    evaluation/
common/       config/     assets/      ops/
tests/        docs/
```

This is a move-and-retire migration. We will not keep permanent duplicate
implementations under both old and new names.

## Compatibility policy

Moves use a temporary forwarding module at the old import path. A forwarding
module may re-export a new implementation, but contains no new behavior.

For every migration slice:

1. add contract tests against the existing behavior;
2. move the implementation to the owner package;
3. leave a forwarding module at the old path;
4. migrate internal imports and canonical CLIs;
5. remove the forwarder only after repository search, CI, launchers, Docker,
   and active-job documentation no longer reference it.

Checkpoint format, existing artifact locations, and legacy CLI overrides are
compatibility contracts. They cannot be changed incidentally by a directory
move.

## Configuration decision

New configuration is composed by domain, while the current flat Hydra surface
remains readable during migration:

```text
config/
  model/          # architecture and conditioning components
  data/           # dataset release, split, cameras, loader settings
  training/       # optimization, checkpoint, distributed/logging controls
  evaluation/     # protocol, rollout and artifact controls
  calibration/    # materialized or sampled geometry perturbations
  experiment/     # named, minimal compositions of the above
  runtime/        # local/cluster/container operational defaults
```

An experiment config should name component choices and only contain genuinely
experiment-specific overrides. New public keys must not add another legacy
alias unless checkpoint or CLI compatibility requires one. The normalization
layer in `utils/hydra_utils.py` remains the sole location that translates
legacy vocabulary during the transition.

## Policy composition decision

The stable 3D policy is composed from explicit components:

```text
Encoder → Scene token selection → optional temporal frontend
       → optional view alignment → trajectory denoiser → action/auxiliary heads
```

`Video-DeltaM`, view alignment, and EE auxiliary prediction are optional
components selected by configuration. They must not introduce simulator or
trainer dependencies into `modeling/`. Existing checkpoints may continue to
use current constructors until a checkpoint-compatible component factory is
implemented and tested.

## Migration order

1. **Baseline and guardrails:** maintain this ADR, the dependency/config map,
   import-boundary tests, and fixed-input model/checkpoint parity tests.
2. **Finish evaluation migration:** make `evaluation/` canonical; retain
   `online_evaluation_rlbench/` only as forwarders until no external launcher
   needs it.
3. **Extract data and training:** move preprocessors and depth geometry under
   `data/`, then trainers under `training/`; replace imports from
   `utils.trainers` in evaluation with public data/model interfaces.
4. **Compose configs by domain:** add typed/domain config groups and a
   compatibility adapter for the flat namespace.
5. **Componentize the policy:** extract optional temporal, alignment, and
   auxiliary-head strategies behind tested interfaces.
6. **Move operations and assets:** migrate launchers/Docker/clusters to `ops/`
   and immutable instructions/protocols to `assets/`; then archive generated
   artifacts outside the checkout.
7. **Retire compatibility shims and dead variants.**

No directory-wide rename or `git mv` sweep is permitted. Each step must be
independently testable and releasable.

## Consequences

### Positive

- A contributor can identify where a feature belongs before editing code.
- Training and evaluation can share model/data contracts without sharing each
  other's orchestration internals.
- Configuration and checkpoint ownership become explicit, preventing silent
  runtime overrides.
- Simulator/container dependencies remain at the evaluation boundary.

### Costs and risks

- Forwarding modules temporarily increase file count.
- Some old imports and checkpoint configuration keys must live through the
  migration.
- Model componentization is high-risk and comes after interface/contract
  coverage, not before.
- Active jobs constrain when a behavior-changing migration can land; a move
  itself must preserve commands, paths, and checkpoint loading.

## Definition of done

The cleanup is complete only when `utils/`, `datasets/`, and
`online_evaluation_rlbench/` have no implementation ownership; model/data/
training/evaluation import boundaries are tested; configs compose by domain;
and ops/assets/generated artifacts have the locations described above.
