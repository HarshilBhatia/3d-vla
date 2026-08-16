# deltaM Experiment Plan — Orbital Bimanual PerAct2 (Aug 2026)

Scope: decide whether the learned-extrinsics (deltaM) mechanism buys camera
generalization on the **new orbital bimanual** setup, where the no-deltaM
baseline is already known (in-domain 79.2 / OOD-cam 72.3, job 120769).
Target: **2-3 training launches** for the next round, not seven.

---

## a. What deltaM is in this codebase

`RotaryPositionEncoding3D` (`modeling/utils/position_encodings.py`) turns a token's
xyz into a sin/cos stack `[B, N, d//6, 6]`. deltaM right-multiplies that stack by a
learned matrix before it becomes a rotary position code (`_apply_delta_M`, line 83;
applied in `forward` line 482 and `_finalize_from_base` line 553). The matrix is
produced by a linear head on camera features and pushed through
`torch.linalg.matrix_exp` of a skew-symmetrized parameter, so it is orthogonal and
initializes at ≈ I (`base_denoise_actor.py:716-740`, `_predict_from_cam_feat`
:757-784). The intent: a per-camera rotation of the RoPE basis lets the policy
re-register a miscalibrated or unseen viewpoint without explicit calibration.
`predict_extrinsics=false` bypasses the whole path (`ExtrinsicsPredictor` base
class returns `(None, None, None)`).

Three modes, selected by `extrinsics_prediction_mode`
(`head_strategies.make_extrinsics_predictor`):

| mode | head | applied as |
|---|---|---|
| `delta_m` | `Linear(embed, 36)` → 6x6 per camera | mixes the 6 sin/cos components within each frequency block; `(B, ncam, 6, 6)`, scattered to tokens via `fps_cam_ids` |
| `delta_m_full` | `Linear(embed, D*D)`, `D = (embed//6)*6` = 120 | full `D x D` mix **across** frequency blocks — far more parameters (14.4k vs 36 outputs), grouped matmul path in `_apply_delta_M` |
| `rt` | `Linear(embed, 6)` axis-angle + translation | not a RoPE mix at all: transforms the point cloud itself (`_transform_pcd_with_extrinsics`), `delta_M` forced to `None` (`denoise_actor_3d.py:193-201`) |

Source of the camera feature differs by mode and matters: `delta_m*` reads the
**per-image average tokens** (the last `ncam` entries of `fps_scene_feats`, appended
in `base_encoder.py:144-147`) → one matrix per camera, content-dependent. `rt`
reads only the static learnable `camera_token` `nn.Parameter`, so it is
batch-constant and cannot adapt per-scene. `dynamic_rope_from_camtoken=true`
(the default in `config/config.yaml:152`) additionally re-predicts deltaM and
rebuilds the RoPE codes **after every cross-attn and self-attn block**, feeding
back the evolving camera token (`features[:, -1, :]`) and per-image tokens
(`base_denoise_actor.py:907-960`). Sin/cos bases are precomputed once and reused,
so the per-block cost is a matmul, not a recompute.

**Known kwarg bug: FIXED, but a different one is live.** `docs/architecture.md`
claims `TransformerHead.__init__` swallows `extrinsics_prediction_mode` in
`**kwargs` without forwarding it. That is stale — `denoise_actor_3d.py:165-166`
now forwards it (and `dynamic_rope_from_camtoken`) via `kwargs.get(...)`. Delete
that doc note. The construction guardrail (`modeling/policy/construction.py`,
`5777b27..b933802`) would **not** have caught it: the guardrail audits
`DenoiseActor.__init__` against config keys, and `DenoiseActor` always forwarded
the flag correctly; the bug was in the inner head's constructor, which the
guardrail never inspects. Audited: all 33 `DenoiseActor` params have a config key,
so the deltaM flags are covered end to end (train and eval both call
`build_model_kwargs`; eval overlays the ckpt config,
`online_evaluation_rlbench/evaluate_policy.py:49-71`).

**Live blocker, `rt` mode only.** `RTExtrinsicsPredictor.forward(self, batch_size,
device)` takes no `fps_*` kwargs, but the single call site passes them
unconditionally (`base_denoise_actor.py:903-905`):
```python
cam_params_rt, delta_M, self._last_predicted_cam_params = self.extrinsics_predictor(
    batch_size, device, fps_scene_feats=fps_scene_feats, fps_cam_ids=fps_cam_ids)
```
`extrinsics_prediction_mode=rt` therefore raises `TypeError` on the first forward.
Verified by signature inspection (`RTExtrinsicsPredictor` args =
`['self','batch_size','device']`, no `**kwargs`). It is a 1-line fix (accept and
ignore the kwargs), but it means **`camtoken_RT.yaml` has never run since the
strategy refactor** — do not schedule an `rt` run without fixing and smoke-testing
it first.

---

## b. What grogu established, and what is untested here

Established (single-arm, 3-cam, `docs/status/experiments.md`):

1. **6x6 beat full D x D.** open_drawer, unseen cams: `deltaM_6x6` 0.84/0.70 (G1/G5)
   vs `delta_m_full` 0.71/0.60 vs no-deltaM 0.71/0.52. The bigger matrix was worse —
   consistent with over-parameterization given the same data.
2. **deltaM is a miscalibration-robustness mechanism.** turn_tap sweep: no-deltaM
   collapsed 0.91 → 0.00 by 15 deg+15 cm; `deltaM_EEF` stayed 0.86 → 0.77;
   `fixmed_rn` (train-with-noise, no deltaM) was flattest at 0.71 but started
   lower (0.73 at 0 deg).
3. **The mechanisms are partly redundant.** On clean-but-novel G7, `fixmed_rn` was
   the *weakest* of the three (0.199 vs 0.216 / 0.228 mean) — training noise buys
   noise robustness at a cost on clean novel views, whereas deltaM won both.
4. **Effect sizes were confounded by a task-difficulty floor.** Grogu's 16-task
   means were 0.20-0.23, and default_3dfa scored 0.27 on its *own training*
   cameras. Most tasks were at zero regardless of camera. All deltaM wins came
   from ~5 easy tasks (`docs/status/stuls.md`).

Untested in the new setting, and (4) is why it matters: the orbital bimanual
baseline is at 0.79 in-domain / 0.72 OOD, so there is real headroom and the
signal is no longer swamped by a difficulty floor. Specifically untested:
bimanual `nhand=2`; 4 cameras rather than 3; siglip2 rather than CLIP RN50;
`nhist=3` **with real visual history**; 3-train-groups-per-task viewpoint
diversity (grogu trained on far fewer camera poses, so its OOD gap was much
larger than our -6.9 — deltaM has less to fix here); and the whole
`dynamic_rope_from_camtoken` path at this scale.

**nhist=3 x deltaM: no interaction. Checked.** The camera token does *not* see
history. `base_encoder.py:103-106` takes `rgb3d_feats[:, -1]` (latest frame) before
computing FPS tokens, `cam_ids_full`, and per-image averages; `base_denoise_actor.py`
:124-127 drops history again before the head. So deltaM is predicted from
current-frame features only, exactly as on grogu. **Cut this rung** — the code
answers it for free.

---

## c. Experiment ladder

Costs: one 100k-iter orbital run = **4x B200 for ~5h20m ≈ 21 B200-hours**
(job 120769 actual). One eval condition = **13 L40S jobs** (1/task, 10 rollouts).
Fixed reference points, already paid for: baseline in-domain 0.792, OOD 0.723,
plus the concurrent miscal sweep (0/2/5/10/15 deg+cm on OOD cams) now running.

### R0 — `rt`-mode fix + deltaM smoke test (no cluster job)

Fix the `RTExtrinsicsPredictor` signature; add a construction/forward test that
instantiates all three modes x `dynamic_rope_from_camtoken` on/off and runs one
forward with `nhand=2, ncam=4, nhist=3` synthetic tensors. Extend
`tests/test_model_construction.py`.
**Decision rule:** all six combinations forward without error before any launch.
**Cost:** local, ~0 GPU-hours. Non-negotiable gate — the `rt` crash proves this
path has zero coverage.

### R1 — deltaM 6x6 on orbital bimanual *(LAUNCH)*

**Hypothesis:** the 6x6 per-camera RoPE mix narrows the -6.9 OOD gap without
costing in-domain accuracy.
**Config delta** vs job 120769: `experiment=camtoken_deltaM`, i.e.
`predict_extrinsics=true extrinsics_prediction_mode=delta_m
dynamic_rope_from_camtoken=true`. Everything else identical (siglip2,
`num_history=3`, `bimanual=true`, `dataset=OrbitalPeract2`, bs256, lr3e-4, 100k,
`image_space_sampling=false`).
**Compared against:** 0.792 / 0.723 (same protocol, same harness).
**Decision rule:** OOD gap shrinks to ≤ -3.0 pts *and* in-domain ≥ 0.77 → deltaM
carries forward. In-domain drop > 3 pts → the mechanism costs more than it buys;
stop the line. Watch `put_item_in_drawer` and `pick_laptop` (-0.30 each) — they
carry most of the gap, so most of any win must appear there. Ignore
`pick_plate`/`straighten_rope` (0.20 in both conditions; not viewpoint-limited).
**Cost:** 21 B200-h + 26 L40S jobs (in-domain + OOD).

### R2 — deltaM 6x6 under eval-time miscalibration *(free rider on R1)*

**Hypothesis:** deltaM's real value is miscal robustness (grogu's largest effect),
not clean-novel-view generalization.
**Config delta:** none — the *same R1 checkpoint*, evaluated with
`miscal_rot_level`/`miscal_trans_level` at 0/2/5/10/15 deg+cm on OOD cams,
`eval_use_depth2cloud=true`. Mirrors the concurrent baseline sweep exactly.
**Compared against:** the concurrent sweep's no-deltaM curve.
**Decision rule:** deltaM's curve stays above baseline's at ≥ 10 deg by ≥ 10 pts
(grogu-scale effect). If the curves overlap, deltaM does not do on bimanual what
it did single-arm and the program stops after R1/R2.
**Cost:** 0 training hours, 4 extra eval conditions ≈ 52 L40S jobs. **Highest
information-per-cost rung in the plan** — it reuses R1's checkpoint and tests the
hypothesis grogu supports most strongly.

### R3 — deltaM + train-time miscal noise *(LAUNCH, conditional)*

**Hypothesis:** the two robustness mechanisms compose. Grogu could not answer
this — `fixmed_rn` and `deltaM_EEF` were separate models, never combined, and
`fixmed_rn` was *worse* on clean novel views, hinting the mechanisms trade off
rather than add.
**Config delta** vs R1: `+ orbital_miscal_noise_level=medium
miscal_max_angle_deg=3 miscal_max_translation_m=0.01` (the grogu `fixmed_rn`
recipe: per-group fixed medium + random noise on top).
**Compared against:** R1 (deltaM, clean training) and the concurrent baseline
sweep, on the same 5-point miscal curve.
**Decision rule:** launch **only if R2 shows deltaM degrading by > 15 pts at
15 deg**. If deltaM is already flat, training noise adds nothing and costs clean
accuracy — skip. If launched: keep if it flattens the curve *without* dropping
clean OOD below R1's.
**Cost:** 21 B200-h + ~78 L40S jobs (in-domain, OOD, 4 miscal points).

### R4 — G7 held-out, winner only *(LAUNCH later, 1 eval only)*

**Hypothesis:** the winning config generalizes to a camera pose whose *group was
never collected*, not merely held out from a collected set.
**Config delta:** none; `spawn_camera_group=G7`.
**Why it stays untouched:** G7 is the only condition in this program that is not
contaminated by selection. G1-G6 all appear in some task's training set, so an
"OOD" group is OOD per-task but in-distribution for the camera-pose manifold. If
G7 is used to choose between R1/R3, it stops being a test and becomes a
validation set, and the headline number becomes unpublishable. Touch it once, at
the end, with one config.
**Decision rule:** report-only, no gating.
**Cost:** 13 L40S jobs, 0 training hours.

### Cuts

- **`delta_m_full` (D x D) — CUT.** Grogu already answered it: 0.71/0.60 vs 6x6's
  0.84/0.70, i.e. the larger matrix lost on both unseen cameras. The mechanism is
  identical in this codebase, and our dataset has *more* viewpoint diversity, not
  less, so the over-parameterization argument only strengthens. Not worth 21
  B200-hours to re-lose.
- **`rt` mode — CUT** (beyond the R0 fix). It cannot work as intended: it predicts
  from the static learnable `camera_token`, so its 6-DoF transform is identical for
  every sample in every scene — it can only learn a global dataset-level offset,
  which the orbital dataset (3 groups/task) does not have. Grogu never ran it
  either. Fix it so it does not rot; do not spend a run on it.
- **`dynamic_rope_from_camtoken` on/off — CUT for now.** It is `true` in every
  grogu deltaM result, so on/off is unmeasured — but it is an *efficiency/depth*
  ablation, not a go/no-go. It only earns a run if R1 succeeds and we want the
  cheaper static-RoPE variant. Defer to a post-R1 round.
- **nhist=3 x deltaM interaction — CUT.** Answered by reading the code (see b):
  the camera token never sees history frames.
- **Branch archaeology — CUT.** `DroPE`, `camera_token`, `comRoPE`, `flex-pred`
  are all **0 commits ahead of `main`** (fully merged, 107-131 behind); nothing to
  recover. Of the `lz-*` branches only `origin/lz-dmself` (vision-token-only deltaM
  via a threaded `vision_mask`, ~45 lines, no `vision_mask` on `main`) and
  `origin/lz-deltam1`'s Björck-orthogonalization parameterization are genuinely
  absent from `main`. Both are alternative parameterizations of a mechanism whose
  *basic* value on bimanual is still unmeasured — revisit only if R1 succeeds.
  Worth taking now regardless, as free instrumentation: `lz-deltam1`'s deltaM
  orthogonality diagnostic (`||WᵀW - I||_F` to wandb).

### Next round: launch R1 now, R2 off its checkpoint, R3 only if R2 says so.

That is 21-42 B200-hours and a decision on deltaM's value on bimanual within one
training cycle.

---

## d. Risks and unknowns

**Untested code paths predating this week's refactors.** The construction
guardrail covers `DenoiseActor`'s 33 params (audited: all have config keys), but
not the inner `TransformerHead`, which reads deltaM flags through
`kwargs.get(...)` defaults — the same shape as the bug the docs describe. The
`rt` crash is proof this family of paths has no forward-pass coverage. R0 exists
for exactly this; do not skip it.

**`ncam=4` vs grogu's 3.** Every grogu deltaM number came from a 3-camera setup.
The orbital dataset has 4 cams/episode (orbital pair + 2 over-shoulder), and deltaM
allocates one matrix per camera by construction (`(B, ncam, 6, 6)` scattered via
`fps_cam_ids`), so 4 cams should be transparent. Two things to check at R0:
`recursive_set_encoder_ncam` defaults to **3** and must never be enabled without
being set to 4 (it is `false` by default — keep it that way, and note that
enabling it silently *replaces* the head's deltaM with a precomputed one,
`base_denoise_actor.py:898-901`); and `ee_aux_cam_ids: [0, 1]` presumes a specific
camera ordering, which matters only if `predict_ee_aux` is turned on. Grogu's
winning `deltaM_EEF` *did* use the EE aux head — it is **not** in R1's config.
That is deliberate (one variable per run), but it means R1 is not a strict replica
of the grogu winner; if R1 underperforms, the aux head is the first thing to add.

**Bimanual `nhand=2` x camera token.** The camera token is a single parameter at
sequence position `-1` and is hand-agnostic; `nhand` only lengthens the trajectory
block (`get_sa_feature_sequence`). The index arithmetic that extracts per-image
tokens in the dynamic path derives `ncam` as
`fps_scene_feats.shape[1] - M` and slices `features[:, traj_seq_len + M : ... ]`
(`base_denoise_actor.py:955-958`) — `traj_seq_len` is read from `traj_feats.shape[1]`
so it should absorb `nhand=2`, but this exact combination
(`dynamic_rope_from_camtoken` + `nhand=2` + `ncam=4`) has never executed. R0's
forward test must use `nhand=2, ncam=4`, not defaults.

**Weaker expected effect than grogu.** Our baseline OOD gap is -6.9 pts; grogu's
was ~13-19 pts on comparable comparisons. With 3 training camera groups per task
the baseline already generalizes well, so deltaM has much less to fix, and at 10
rollouts/task per condition the per-task resolution is 0.1 — a 3-pt mean shift is
near the noise floor. Consequence: **R1 alone may be underpowered.** This is the
real argument for R2 being the primary rung — the miscal sweep manufactures the
large degradation that makes the effect measurable. If R1 comes back within
±3 pts, do not conclude "no effect" from it; conclude "underpowered" and read R2.

**Norm-stats / checkpoint provenance.** Reuse the exact staged orbital zarr at
`/k8s-nfs/harsvbha/3dfa/data/orbital_peract2/` and the same
`instructions/peract2_orbital_task_group_mapping.json` as job 120769, or the
79.2/72.3 reference stops being a valid control.

**Eval-time flag drift.** The `image_space_sampling` incident cost ~9-31 points
silently. R1's eval must pass `ISS=false` explicitly and the harness must be the
new bimanual+orbital one (`e042de3`). Confirm the deltaM flags actually arrive in
the rollout process by reading the `loaded_from_ckpt` dump in the eval log
(`evaluate_policy.py:58`) rather than assuming the overlay worked.
