# High-confidence migration playbook

This is the operational companion to
[ADR-001](adr-001-repository-boundaries.md). The goal is to make a structural
cleanup boring: preserve behavior while moving one ownership boundary at a
time, then improve the design only after behavior is protected.

## The rule: one migration slice, one invariant

Do not combine package moves with model changes, config renames, numerical
changes, launcher changes, or artifact moves. Each pull request owns one
slice, has an explicit compatibility story, and can be reverted on its own.

The first slices should be:

1. Finish evaluation forwarding and remove duplicated evaluator behavior.
2. Extract trainer-owned batch/action helpers to a public data interface.
3. Move data preprocessors and depth-to-cloud geometry under `data/`.
4. Move trainer implementations from `utils.trainers` to `training/`.
5. Split the flat Hydra namespace into domain configs.
6. Only then componentize policy variants or retire legacy architecture flags.

Do **not** begin with a repository-wide rename. That creates a large, hard to
review diff and makes it impossible to attribute a parity failure.

## Required checks for every migration slice

### Before editing

1. Choose one checkpoint that represents the behavior affected by the move.
   For Video-DeltaM work, use the current decision-grade K5v/K3p checkpoint;
   for a generic training utility, use a representative baseline checkpoint.
2. Freeze one preprocessed `model.forward` input batch and one eager inference
   output. Put the artifacts outside the checkout, alongside the checkpoint or
   in the durable artifact store.
3. Create a case JSON as below and run the opt-in parity contract once on the
   known-good commit. This proves the case is usable before it becomes a gate.
4. Identify the public API to preserve: imports, CLI, Hydra flags, checkpoint
   keys, output paths, or all of these.

### During the change

1. Add tests for the extracted public interface before moving its implementation.
2. Move the implementation.
3. Leave a behavior-free forwarding import at the old path.
4. Migrate one internal caller at a time.
5. Do not refresh the golden output to make a failure pass. A golden refresh is
   permitted only after an intentional, separately reviewed behavior change.

### Before merging

1. Run focused unit/contract tests for the changed package.
2. Run the manual checkpoint parity contract below.
3. Run one normal import/CLI smoke test for the old compatibility path.
4. Inspect `rg` results for old imports; preserve only documented forwarders or
   external launchers that have not yet migrated.
5. Record the checkpoint, digest, GPU/software environment, and result in the
   migration PR or changelog.

## Manual checkpoint parity contract

`tests/test_migration_parity.py` is intentionally opt-in and GPU-only. It is
not part of routine CI and does not run during ordinary `pytest` execution.
It performs three checks:

1. the checkpoint's SHA-256 matches the pinned case;
2. the canonical evaluation loader can construct it in eager mode;
3. a seeded deterministic `run_inference=True` forward pass exactly matches
   the frozen golden tensor (unless the case explicitly records a reviewed
   tolerance).

Run it immediately before and after a mechanical migration:

```bash
THREEDFA_MIGRATION_PARITY_CASE=/abs/path/peract2_video_case.json \
  pytest -m migration_parity -q
```

The test forces deterministic PyTorch/CuDNN settings and disables TF32 for the
comparison. It uses the canonical evaluation checkpoint loader, which means it
also exercises checkpoint-config overlay and model construction. It does not
invoke `torch.compile`; an accidentally compiled model fails the test.

### Case format

```json
{
  "schema_version": 1,
  "checkpoint": "/durable/checkpoints/model_best.pth",
  "checkpoint_sha256": "<sha256 of exactly that file>",
  "input": "/durable/parity/model_input.pt",
  "expected_output": "/durable/parity/model_output.pt",
  "seed": 20260915,
  "rtol": 0.0,
  "atol": 0.0
}
```

`input` is a `torch.save` dictionary with the exact `model.forward` keyword
arguments: `gt_trajectory`, `trajectory_mask`, `rgb3d`, `rgb2d`, `pcd`,
`instruction`, and `proprio`. It must be captured **after** normal dataset
collation/preprocessing, so this test isolates model/checkpoint behavior from
the Zarr loader and simulator. `expected_output` is the CPU tensor returned by
that input with `run_inference=True`, captured with the same seed and
deterministic settings as the test.

Capture a case from a known-good revision with:

```bash
python scripts/capture_migration_parity.py \
  --checkpoint /durable/checkpoints/model_best.pth \
  --input /durable/parity/model_input.pt \
  --case /durable/parity/peract2_video_case.json
```

The capture command refuses to overwrite existing golden artifacts without
`--force`; use that flag only after a reviewed, intentional behavior change.

Keep the large tensor artifacts out of Git. The JSON case is small enough to
commit only if its paths are portable; otherwise store it beside the artifacts.

### When a mismatch is acceptable

Only a deliberate numerical or behavioral change may update the golden output.
The PR must state why exact parity is no longer expected, retain the old case
until review is complete, and record a new checkpoint digest/case. A package
move, import cleanup, config reorganization, or compatibility wrapper change
is **not** a reason to refresh it.

## Concrete first execution plan

### Slice A — evaluation completion (low risk)

- Treat `evaluation/cli.py` as the only canonical evaluator entry point.
- Inventory launchers still calling `online_evaluation_rlbench/`.
- Add import/CLI contract tests for each legacy wrapper.
- Move any remaining implementation from old evaluator modules into
  `evaluation/online/rlbench/`; old modules remain one-line re-exports.
- Run protocol/artifact tests plus one fixed-seed small rollout.

**Exit condition:** `online_evaluation_rlbench/` contains no behavior, only
explicit deprecation-compatible forwarders.

### Slice B — public batch/action interface (low-to-medium risk) — complete

- Identify `base_collate_fn` and `relative_to_absolute` in
  `utils.trainers.base` as data/action semantics, not training orchestration.
- Move them to a small public module under `data/` with direct unit tests.
- Replace evaluation imports first, then trainer imports.
- Leave an old-path forwarder.
- Run offline evaluation tests and the checkpoint parity contract.

**Exit condition:** `evaluation/` imports no `utils.trainers.*` symbols. This
slice is complete in the current checkout; the old trainer module re-exports
the helpers for compatibility.

### Slice C — preprocessing and depth geometry (medium risk) — complete

- Move `utils.data_preprocessors` to `data/preprocessing` and
  `utils.depth2cloud` to `data/geometry` (or `data/depth` if clearer).
- Preserve registry names and constructor signatures.
- Add deterministic unit tests around depth unprojection and each
  miscalibration composition order.
- Run offline loss sweep smoke, a fixed online eval smoke, and parity.

**Exit condition:** old `utils` modules are pure forwarders and both training
and evaluation use `data/` imports. This slice is complete in the current
checkout; focused registry/forwarding tests cover the compatibility boundary.

### Slice D — training extraction (medium risk) — complete

- Create `training/` with trainer implementation, metrics, schedulers, EMA,
  checkpoint IO, and training CLI glue.
- Keep `main.py` behavior and its `utils.trainers` import via a forwarder for
  the first transition.
- Move `main.py` only after training/resume and offline-validation smoke tests
  match pre-move artifacts.
- Use both a baseline and Video-DeltaM parity case.

**Exit condition:** `training/` owns training implementation; `utils.trainers`
is a forwarder only. This slice is complete in the current checkout; model,
checkpoint, and optimizer behavior were not intentionally changed.

### Slice E — configuration composition (medium risk)

- Introduce domain groups without deleting the flat keys.
- Add a test that old CLI overrides and a named new composition resolve to the
  same public and legacy runtime values.
- Migrate one experiment at a time; checkpoint config overlay remains unchanged.
- Delete a legacy alias only after no checkpoint/launcher requires it.

**Exit condition:** new experiments compose `model`, `data`, `training`,
`evaluation`, `calibration`, and `runtime` intentionally rather than adding
more root-level flags.

## What we should not do yet

- Do not move `legacy_may/`; it is untracked archival material.
- Do not relocate historical result paths or checkpoints.
- Do not delete legacy flags just because a public alias exists.
- Do not refactor the policy while moving its dependencies.
- Do not make the GPU parity test a required default test; it is a manual,
  reproducible migration gate.
