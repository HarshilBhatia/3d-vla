# 3DFA documentation

Canonical state of the repository, split by how fast it changes.

- **[`handbook/`](handbook/)** — how the system works. Changes rarely; if the
  code contradicts it, the handbook is a bug.
- **[`research/`](research/)** — what we are finding out. Changes weekly.
Figures are not a separate area: each one lives in the `figures/` folder of
whatever owns it, and every such folder has a README naming the script that
produces each file.

## handbook/

| Doc | What it answers |
|---|---|
| [setup.md](handbook/setup.md) | Environment, CoppeliaSim/PyRep/RLBench install, per-user paths |
| [commands.md](handbook/commands.md) | How to train, evaluate, and convert data |

**[model/](handbook/model/)**

| Doc | What it answers |
|---|---|
| [architecture.md](handbook/model/architecture.md) | Pipeline, key modules, non-obvious design details |
| [config.md](handbook/model/config.md) | Hydra config groups and important flags |
| [terminology.md](handbook/model/terminology.md) | Source of truth for paper-facing names, public config keys, view-alignment and history axes |

**[data/](handbook/data/)**

| Doc | What it answers |
|---|---|
| [zarr-format.md](handbook/data/zarr-format.md) | Zarr schema, dataset classes, on-GPU preprocessing |
| [camera-extrinsics.md](handbook/data/camera-extrinsics.md) | PerAct2 extrinsics: source, convention, transforming them |
| [orbital-collection.md](handbook/data/orbital-collection.md) | Generating raw orbital demos end to end, then converting to zarr |
| [orbital-pipeline.md](handbook/data/orbital-pipeline.md) | The `orbital/` package: modules, entry points, dependency graph |

**[eval/](handbook/eval/)**

| Doc | What it answers |
|---|---|
| [protocols.md](handbook/eval/protocols.md) | Calibration realizations, eval plans, campaign naming, output layout |
| [miscalibration.md](handbook/eval/miscalibration.md) | How miscalibration is applied in online eval; built-in levels |
| [checkpoint-triggered-evals.md](handbook/eval/checkpoint-triggered-evals.md) | Submitting online evals automatically from training |
| [peract2-2d-and-camera-subsets.md](handbook/eval/peract2-2d-and-camera-subsets.md) | The 2D baseline run and the camera-subset miscalibration evals |

**[codebase/](handbook/codebase/)**

| Doc | What it answers |
|---|---|
| [repo-boundaries-adr.md](handbook/codebase/repo-boundaries-adr.md) | Target package boundaries and dependency direction (accepted ADR) |
| [dependency-map.md](handbook/codebase/dependency-map.md) | As-is map of entry points, package ownership, boundary violations |
| [migration-playbook.md](handbook/codebase/migration-playbook.md) | How to move a boundary without breaking behavior |
| [figure-conventions.md](handbook/codebase/figure-conventions.md) | `utils.plotting` rules for experiment figures |

## research/

| Area | What it holds |
|---|---|
| [experiments/](research/experiments/) | One page per experiment, filed by the week it started. [Registry](research/experiments/README.md) lists question, status and headline for all of them |
| [results/](research/results/) | Settled numbers. [Summary](research/results/README.md) is the decision-facing page; [artifact-index](research/results/artifact-index.md) maps the raw `/grogu` campaigns; `protocols/` holds per-protocol detail; `data/` holds the machine-readable exports |
| [weekly_logs/](research/weekly_logs/) | One file per week: submissions, results as they landed, bugs, decisions |
| [reproduce.md](research/reproduce.md) | Provenance map for the Aug 2026 campaign: commits, configs, checkpoints, egress bundle |
| [archive/](research/archive/) | Pre-Sept working notes. **Not canonical** |

Figure folders: [handbook/eval/figures/](handbook/eval/figures/) (geometry
illustrations), [research/results/figures/](research/results/figures/)
(cross-model plots, plus the 248 MB `task-gallery/` renders),
[research/experiments/2026-09-29/figures/](research/experiments/2026-09-29/figures/)
(that week's panels and the viewpoint-sweep captures). Bulk renders and `.rrd`
recordings are gitignored and regenerable.

## Where does a new page go?

- A fact about how the system works → `handbook/`, in the matching subfolder.
- A thing you are currently testing → `research/experiments/<week>/`.
- A number you would cite → `research/results/`.
- What happened this week → `research/weekly_logs/`.
- A figure → the `figures/` folder of whichever of those owns it.
