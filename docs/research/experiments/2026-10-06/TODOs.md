# TODOs: what have exp 3 and 1b learned?

A working queue of analyses. A subagent works through it top to bottom; new items are appended while it runs. Each item gets its results written directly under it.

## How to work this list (read first)

1. **Re-read this file before starting each item.** Items are added while you work.
2. Take the **first item whose status is `open`**. Set it to `in progress (agent)`, commit, then do it.
3. When done: write **Results** (numbers in tables, where files and figures live) and a short **Reading** (what it means, what it rules out) under the item. Set status to `done`, then commit.
4. If an item is blocked or ambiguous, set it to `blocked: <reason>`, write what you tried, and move on to the next one.
5. When no `open` items remain, stop and report.
6. **Several agents work this file at once.** Only take an item whose status is exactly `open`. Items reserved for another agent say `claimed (<name>)` or `in progress (<name>)`; skip them.
   - Edit **only your own item's section**, and re-read the file immediately before every edit, so you never overwrite another agent's text.
   - Commit only your own files (`git add <paths>`, never `git add -A`). If `git commit` hits an index lock, wait a few seconds and retry.

**Rules**
- Follow `CLAUDE.md`: one-line comments, docs are canonical, and figures go in `figures/` next to this file with a README entry naming the script.
- New scripts go in `scripts/eval/` or `scripts/analysis/`. Reuse `scripts/eval/diagnose_delta_m.py` (load, preprocess and record) where it fits.
- GPU work goes through Slurm: `--partition=shubhamlong,all`, 1 GPU, and `--exclude=grogu-4-3,grogu-4-13,grogu-3-5,grogu-3-10,grogu-3-15,grogu-3-20,grogu-3-25,grogu-3-30,grogu-4-18`. Those nodes are slow Turing cards, broken or mislabelled.
- **Never cancel a RUNNING job.** Don't touch anyone else's jobs or any training.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`. Don't push.
- Online evals: `sbatch --array=0-<N-1> scripts/eval/online_eval_plan.slurm <plan.json>`. Put `eval_weights` in `overrides`, never at the top level of `runtime`.

## Context

The week's runs ([camera-adaln](camera-adaln.md), [randhist-deltam](randhist-deltam.md)) are all from scratch, 200k steps, random pooled history, EMA, no history register.

| Model | Mechanism | Checkpoint (EMA weights) | IDLT calibrated / 60-60 at 200k |
|---|---|---|---|
| **exp 3** | layer-wise delta_M (`rope_6d`) predicted from the C-tokens before every SA layer, cameras [0,1] | `train_logs/PerAct2/peract2_orbital_vid_pooled_randhist_deltam_scratch/interm_step_200000.pth` | 77.5 / 71.6 |
| **1b** | pooled layer-wise C-token adaLN (`camera_adaln: pooled`), no delta_M | `train_logs/PerAct2/peract2_orbital_vid_pooled_randhist_adaln_pooled_scratch/interm_step_200000.pth` | 80.8 / 70.4 |

**References** (raw weights unless noted):
- VideoPooled 84k: `train_logs/PerAct2/peract2_orbital_video_deltam_external_warmstart_k5v_k3p_a5000_resume/best.pth`
- Warm random history, 50k: `train_logs/PerAct2/peract2_orbital_vid_deltam_pooled_randhist_notime/interm_step_50000.pth`
- Warm random history + EMA fine-tune, 15k: `train_logs/PerAct2/peract2_orbital_ema20k_randhist/interm_step_15000.pth`, EMA weights; 76.5 at 60/60
- Control A (no delta_M, no adaLN): `train_logs/PerAct2/peract2_orbital_vid_pooled_noalign_randhist_scratch/interm_step_{160000,200000}.pth`

**Already known:**
- **delta_M's input can't see miscalibration.** C-tokens are mean-pooled appearance with no xyz, so delta_M before cross-attention is identical under clean and 60/60 cameras.
- **Averaged over all calls, the size of delta_M barely changes under miscalibration.** It is ‖ΔM − I‖_F per camera: exp 3 0.77 → 0.89 from 110k to 200k; VideoPooled about 0.38; warm random history about 0.35.
  - **Not yet measured:** how delta_M varies per layer and per sample.
  - Source: `results/diagnose_delta_m/wk1006.json`.
- **Offline action error is unreliable for the older warm checkpoints.** VideoPooled shows 7 cm, probably a proprio-order mismatch. Treat those models' action errors as invalid until fixed, but their delta_M values are fine.

**Code pointers:**
- delta_M: `TransformerHead._predict_from_cam_feat` (head) and `HistoryFeatureExtractor._delta_m_from` (Direct), plus the layer-wise loop in `TransformerHead.forward` (`modeling/policy/base_denoise_actor.py`).
- adaLN: `_forward_camera_adaln` and `_camera_adaln_signal`.

---

## TODO 1: delta_M per layer and per sample (exp 3, with VideoPooled and warm random history as references)

**Status:** done

**Why:** we reported delta_M as roughly constant, but we only measured an average over every call. It is re-predicted before each SA layer, from C-tokens that change layer to layer, so it can vary with depth and across scenes.

**Do:** extend `scripts/eval/diagnose_delta_m.py`, or write a sibling, to record every delta_M tagged with **(sample, denoising step, layer index, camera)**.
- In the layer-wise path, the first prediction is before cross-attention; then one prediction before each of the 4 SA layers.
- Run on exp 3 at 200k (also 110k if cheap), VideoPooled 84k and warm random history 50k, under clean and heavy 60°/60 cm on cameras [0,1].
- Use the same samples and seeds as the existing script, about 20 validation batches covering several tasks.

**Report, per model:**
1. Per layer: mean and std over samples of ‖ΔM − I‖_F, for cameras 0 and 1.
2. How much delta_M changes with depth: ‖ΔM_l − ΔM_{l−1}‖_F between consecutive layers, and the rotation angle of ΔM_l·ΔM_{l−1}ᵀ (use the norm of its matrix log, or the principal angles).
3. Across-sample variation at each layer: std over samples, and the share of that variance explained by task. Group by `sample["task"]`.
4. Clean vs 60/60 per layer: ‖ΔM_clean − ΔM_heavy‖_F on the same sample.
5. Variation across denoising steps: does delta_M differ between the 5 flow steps?
6. Camera 0 vs camera 1: are their delta_Ms close to each other?

Save the raw tensors (`results/analysis/delta_m_per_layer/<model>.pt`) and one figure per metric (per-layer curves, models overlaid).

**Results:**

Scripts: `scripts/analysis/delta_m_per_layer.py` (records, Slurm job 3993242), `scripts/analysis/plot_delta_m_per_layer.py` (stats and figures), `scripts/analysis/delta_m_trunk_activity.py` (C-token spread and live trunk units, job 3993253). Raw tensors: `results/analysis/delta_m_per_layer/<model>.pt`, shaped (853 samples, 5 flow steps, 6 calls, 4 cams, 6, 6), with tasks and per-batch pos_l2. Stats: `results/analysis/delta_m_per_layer/summary.json` and `trunk_activity.json`. Figures: `figures/dm_layer_*.png` (see `figures/README.md`).

Setup: all 853 val samples (13 tasks), clean and heavy60 on cameras [0,1]. The first 320 samples are the old script's 20 batches; on those, the mean over all calls reproduces `wk1006.json` exactly (exp 3 200k 0.886, 110k 0.765, VideoPooled 0.381, warm 0.354). One deviation from the old script: I reseed after preprocessing, because heavy60 draws its rotation axes from the torch RNG. In `diagnose_delta_m.py` that means **clean and heavy60 did not share flow noise**, so its action-error comparison carries noise mismatch. Its pre-CA delta_M comparison is unaffected.

**Call structure (all four models are layer-wise "refine").** Each flow step makes 6 head calls: `init` (from `view_align_predictor`, computed but unused in the layer-wise branch), CA, then SA0–SA3. init, CA and SA0 read the same C-tokens and are bit-identical. The old "average over every call" therefore triple-counted the CA value and included one unused prediction. Wrist cameras are exactly identity.

1. ‖ΔM − I‖_F per layer, camera 0, mean (std over samples). Camera 1 agrees within 0.01.

| Model | CA = SA0 | SA1 | SA2 | SA3 |
|---|---|---|---|---|
| exp 3 200k | 0.701 (0) | **2.662** (0.0007) | 0.288 (0.037) | 0.276 (0) |
| exp 3 110k | 0.673 (0) | 2.096 (0.121) | 0.245 (0.039) | 0.231 (0) |
| VideoPooled 84k | 0.681 (0) | 0.078 (0.006) | 0.082 (0) | 0.082 (0) |
| warm randhist 50k | 0.258 (0) | 1.162 (0.104) | 0.068 (0.006) | 0.067 (0) |

At SA1 in exp 3 200k, ‖log ΔM‖_F = 3.000 for every sample. That is the predictor's clamp (`max_norm=3.0` on the skew generator), so the SA1 delta_M is pinned at the maximum allowed rotation. The 110k → 200k growth (0.76 → 0.89 averaged) comes almost entirely from SA1 running into this clamp.

2. Depth change, ‖log(ΔM_l ΔM_{l−1}ᵀ)‖_F in radians, with the largest principal angle (camera 0). Frobenius differences are in summary.json.

| Model | CA→SA0 | SA0→SA1 | SA1→SA2 | SA2→SA3 |
|---|---|---|---|---|
| exp 3 200k | 0 | 2.52 (91°) | 2.95 (106°) | 0.02 (0.7°) |
| exp 3 110k | 0 | 1.71 (63°) | 2.17 (79°) | 0.02 (0.9°) |
| VideoPooled | 0 | 0.63 (25°) | 0.02 (0.7°) | 0.0001 |
| warm randhist | 0 | 1.26 (47°) | 1.21 (46°) | 0.002 |

3. Across-sample RMS spread of the step-mean delta_M (Frobenius), with task η² in brackets. All 853 samples; chance η² ≈ 0.014.

| Model | CA = SA0 | SA1 | SA2 | SA3 |
|---|---|---|---|---|
| exp 3 200k | 0.0000 [–] | 0.026 [0.52] | 0.060 [0.39] | 0.0000 [–] |
| exp 3 110k | 0.0000 [–] | 0.154 [0.68] | 0.061 [0.44] | 0.0006 [–] |
| VideoPooled | 0.0000 [–] | 0.025 [0.84] | 0.0004 [–] | 0.0000 [–] |
| warm randhist | 0.0000 [–] | 0.112 [0.66] | 0.013 [0.09] | 0.0000 [–] |

"–" means there is no spread (RMS < 1e-3), so η² is undefined. On the first-320 subset (6 tasks) the SA1 η² values are similar (0.38–0.77).

4. ‖ΔM_clean − ΔM_60/60‖_F with the same sample and same flow noise, camera 0, mean (std).

| Model | CA = SA0 | SA1 | SA2 | SA3 |
|---|---|---|---|---|
| exp 3 200k | 0 | 0.009 (0.004) | 0.010 (0.037) | 0.0001 |
| exp 3 110k | 0 | 0.017 (0.021) | 0.026 (0.058) | 0.0005 |
| VideoPooled | 0 | 0.0006 | 0.0000 | 0 |
| warm randhist | 0 | 0.009 (0.013) | 0.0005 | 0 |

The largest of these is about 1°, against 60° of injected rotation.

5. Across the 5 flow steps: CA and SA0 are constant by construction. The RMS deviation from the step mean is 0.009 at SA1 and 0.002 at SA2 for exp 3 200k; 0.017 / 0.004 for 110k; 0.004 / 0.0001 for VideoPooled; 0.022 / 0.0006 for warm. SA3 is 0 for all four.

6. Camera 0 vs camera 1, ‖log(ΔM_0 ΔM_1ᵀ)‖_F: exactly 0 at CA and SA0 in every model. At SA1 and SA2 it is ≤ 0.017 (≤ 0.6°); at SA3 it is 0.

**Why delta_M is this static** (`trunk_activity.json`, 5 clean batches):
- **The C-tokens entering the head are collapsed.** The HistoryFeatureExtractor's current-frame input varies across samples (per-channel std 0.025–0.048; camera 0 vs 1 distance 0.4–0.8). Its output, the head's C-tokens, has per-channel std about 2e-5 to 8e-5, and cameras 0 and 1 are equal to within 1e-3, in all four models. The pre-CA delta_M is therefore one learned constant matrix, shared by both cameras and every sample.
- **The delta_M trunk (Linear + ReLU, 120 units) is almost dead.** About 1 unit is ever active at CA/SA0, 2 at SA1, fewer than 1 at SA2, and none at SA3. At SA3 delta_M equals exp(skew(bias)) of `camera_predictor` exactly (difference 0.0). Each layer's delta_M is thus confined to a family with at most two parameters around a bias rotation.

**Reading:**
- **delta_M is a learned, per-layer fixed basis change, not a per-scene or per-camera estimate.** It does vary with depth, and strongly: exp 3 applies a different large rotation of the RoPE basis at SA1 (the clamp-limited maximum) than at CA/SA0 and SA2/SA3. But at any given layer it is close to constant across samples, flow steps, cameras and calibration. This replaces the earlier "roughly constant": delta_M is constant within a layer, not across layers.
- **The "C-tokens are appearance only" explanation was too generous.** The C-tokens reaching the head carry essentially no per-sample information at all; the HistoryFeatureExtractor collapses them in every pooled model checked. The only per-sample signal in delta_M enters through SA mixing into the C-token positions (SA1, SA2). It is small: at most 0.15 RMS against rotations of size 2–3. It is mostly task identity (η² 0.5–0.8), and it is close to blind to 60/60 miscalibration (≤ 1°).
- **This rules out delta_M doing camera-specific geometric correction in any of these models.** exp 3's robustness gain over VideoPooled cannot come from delta_M tracking the miscalibration. If delta_M matters, it is as a learned RoPE re-mixing per layer; TODO 4 should look at the SA1 matrix in particular. Next checks: why the extractor output collapses (post-LN blocks with large shared camera/current-frame embeddings are a candidate), and whether the dead trunk is a training artifact.
- **Caveat.** The η² values describe very small spreads at SA2 and are noisy there; read SA1 as the meaningful one. The offline pos_l2 values recorded with each model are not re-validated (see Context on the older warm checkpoints).

---

## TODO 2: what the 1b adaLN signal encodes

**Status:** in progress (agent)

**Why:** 1b conditions every head layer on `signal = time_embs + MLP_c(mean of C-tokens)`, re-read after each SA layer (`camera_adaln: pooled`). We don't know whether it carries sample-specific information (task, scene, progress, miscalibration) or settles into a learned near-constant, as delta_M appears to.

**Do:** hook `_camera_adaln_signal`, plus the `AdaLN` modules in `cross_attn`, `self_attn` and the position/rotation heads, on 1b at 200k. Use the same validation samples, seeds and conditions as TODO 1 (clean / train / heavy60). Record per (sample, denoising step, layer):
- the camera term `MLP_c(mean C)` on its own, and `time_embs` on its own;
- each AdaLN block's `scale` and `shift`, from its `modulation` on the full signal, and the same with the camera term removed (`time_embs` only).

**Report:**
1. **Strength of the camera conditioning per layer:** ‖scale_full − scale_time‖ and ‖shift_full − shift_time‖, relative to ‖scale_time‖ and ‖shift_time‖. Is the camera term a small nudge or a dominant part of the modulation?
2. **Variation across samples of the camera term per layer:** std over samples relative to its mean norm, and the share of variance explained by task (`sample["task"]`). Also cosine similarity between samples, within a task and across tasks.
3. **Sensitivity to miscalibration per layer:** ‖camera term clean − camera term heavy60‖ on the same sample. Layer 0 should be about 0, because C-tokens are appearance-only; later layers may react through attention to miscalibrated scene tokens.
4. **Change with depth:** cosine between consecutive layers' camera terms.
5. **Dependence on denoising step:** the camera term should not depend on it (it doesn't take `time_embs`), but check that the full modulation's mix of time and camera changes across steps.
6. A low-dimensional view: PCA of the layer-0 and last-layer camera terms, coloured by task, and a figure.

**Results:**

**Reading:**

---

## TODO 3: where trajectory queries attend (external vs wrist cameras), clean vs 60/60

**Status:** in progress (attention agent)

**Why:** hypothesis: robust models move attention away from the miscalibrated external cameras (0, 1) onto the wrist cameras (2, 3, always calibrated) when geometry is inconsistent. Last week's probe showed blacking out the external cameras is catastrophic (VideoPooled 17), so they still matter; this measures how the balance shifts.

**Do:** record attention weights in the head for **exp 3 (200k), 1b (200k), control A (200k, raw)** and the warm random-history model with EMA (`ema20k_randhist` 15k). Use the same samples, seeds and conditions as TODO 1.
- **Cross-attention:** the trajectory queries over the dense scene tokens, both layers. Map keys to cameras: the dense scene tokens are camera-major, with `Np // ncam` tokens per camera.
- **Shared self-attention:** the trajectory query rows over the FPS scene tokens (camera from `fps_cam_ids`), the C-tokens, the registers and the camera token, each of the 4 layers.
- **Output heads:** the 2 self-attention layers each of the position and rotation heads.

`MultiheadCustomAttention` may not return weights. Add an opt-in hook or flag that changes nothing when off, and keep the test suite green.

**Report**, per model, layer and condition: the share of trajectory-query attention mass on camera 0+1 vs camera 2+3 tokens (also on C-tokens and registers in self-attention), averaged over samples, heads and denoising steps.
- Main number: **Δ(external share) from clean to heavy60.**
- Also attention entropy per layer (does attention get more diffuse or more focused under heavy miscalibration?).
- One figure: external share per layer, clean vs heavy, models overlaid.

**Results:**

**Reading:**

---

## TODO 4: what exp 3's delta_M does to the RoPE kernel

**Status:** open

**Why:** delta_M is a 6×6 orthogonal matrix that mixes the sin/cos RoPE basis of a camera's scene tokens (`RotaryPositionEncoding3D._finalize_from_base`). It doesn't track miscalibration (see Context), so its effect is mostly a learned transform of how attention depends on 3D position. We want to know what that transform is.

**Do:** pure linear algebra on the weights. Take exp 3's delta_M per camera from TODO 1's saved tensors: per layer, sample mean, and a few individual samples. Also VideoPooled's for reference. For a query at the origin and a key displaced by d along x, y or z (and along random directions), at |d| = 0, 1, 2, 5, 10, 20, 40, 60 cm, compute the RoPE inner-product factor that the relative rotation contributes to the attention logit, averaged over frequency bands.
- Do it with delta_M = identity (baseline geometry) and with the learned delta_M.
- Note that delta_M is applied only to scene/FPS tokens, not trajectory tokens. Model the actual asymmetry: trajectory query uses the plain basis, scene key uses the mixed one.
- Read `modeling/utils/position_encodings.py` to get the frequencies, the basis layout and exactly where delta_M enters.

**Report:**
1. The kernel versus displacement, per axis, identity vs learned. Does delta_M flatten (blur) the kernel, sharpen it, or shift its peak away from zero displacement (an effective offset)?
2. Which basis components delta_M mixes: rotation angles and invariant subspaces of the 6×6 (eigen-decomposition), and whether it swaps or mixes x/y/z or sin/cos.
3. What a rigid miscalibration of 60°/60 cm does to that kernel for a fixed scene point, with and without delta_M. Does the learned delta_M make the logits less sensitive to the miscalibration?
4. Figures: kernel curves per axis, and the 6×6 matrices as heatmaps.

**Results:**

**Reading:**

---

## TODO 5 (larger experiment): linear probes — do the C-tokens know the true camera pose, and the injected miscalibration?

**Status:** in progress (probing agent)

**Why:** delta_M was designed to correct camera error, and 1b's adaLN reads the same C-tokens. If the representation never encodes where the camera is or how wrong its extrinsics are, then neither mechanism can be correcting geometry, and robustness must come from somewhere else. Two targets only:
- **(A)** true camera pose with respect to the robot (world) frame;
- **(B)** the injected miscalibration.

**Features** (per external camera 0 and 1; the wrist cameras 2 and 3 serve as a control):
1. Encoder C-tokens: mean-pooled appearance, `camera_summaries` before the history extractor.
2. C-tokens after the history extractor (`refined_frames[:, -1]`).
3. C-tokens after each of the 4 head SA layers, read from the feature sequence at `traj_len + M : traj_len + M + ncam`. Use denoising step 0 and the last step, since they may differ.
4. What each model actually uses:
   - **exp 3:** delta_M per layer (the 6×6, flattened, or its skew generator).
   - **1b:** the adaLN camera term `MLP_c(mean C)` per layer. It is global, so probe it against both external cameras' targets concatenated.

**Models:** exp 3 at 200k (EMA), 1b at 200k (EMA), control A at 200k (raw), and the warm random history + EMA model (`ema20k_randhist` 15k). Add a **randomly initialised** copy of the exp 3 architecture as the floor.

**Targets:**
- **(A) True camera pose,** from the **clean** stored extrinsics (camera-to-world; see `docs/handbook/data/camera-extrinsics.md` for the convention and what "robot frame" means here):
  - camera position xyz;
  - viewing direction (unit forward axis);
  - azimuth, elevation and distance about the workspace centre (≈ (0.25, 0, 0.95); verify).
  - Also report camera-group (G1–G6) classification.
  - **Check first how many distinct poses the data contains.** If it is a handful of fixed groups, pose regression collapses into group classification. In that case also run a **leave-one-group-out** probe: train on 5 groups, test on the 6th. That measures whether pose is encoded continuously or just memorised per group.
- **(B) Injected miscalibration:** hook the preprocessor's `_get_miscal_noise` to record the applied (B, ncam, 4, 4) error per camera. Targets:
  - rotation error as axis-angle (3 values) and its magnitude in degrees;
  - translation error (3 values) and its magnitude in cm;
  - clean vs miscalibrated (binary).
  - Training-style noise (medium bias + 3°/1 cm) has too little spread for regression. Add a **probe-only sweep**: fixed-magnitude random directions with rotation uniform in [0°, 60°] and translation in [0, 60 cm], drawn independently per sample, on cameras [0, 1].
  - **Expectation:** (B) is impossible at depths 1–2 and for the first delta_M or adaLN term, because those inputs are appearance-only. The real question is whether it becomes decodable at head depth 3, where C-tokens attend to scene tokens placed at miscalibrated 3D positions.

**Protocol:**
- Fit on **training-split** samples (several thousand, spread across tasks and episodes) and test on the **validation split**. Never let a probe see test episodes.
- Standardise features. Use ridge regression for continuous targets (pick α by CV on the training set) and logistic regression for classes.
- Baselines: shuffled labels, the random-init model, and a "mean C-token of the wrist cameras" feature for the external-camera pose (should be weak).
- Report per model and depth: test R² (and position error in cm, angle error in degrees, for the pose and miscalibration regressions) and classification accuracy.
- One figure per target: probe score against depth, models overlaid.
- Save features and labels under `results/analysis/probes/` so new probes can be fit without re-running the GPU.

**Results:**

**Reading:**

---

## TODO 6: under heavy miscalibration, does the model follow the miscalibrated cameras or ignore them?

**Status:** in progress (intervention agent)

**Why:** a miscalibration error N_j on external camera j (observed extrinsics = N_j · true, camera-to-world) moves every world point that camera sees by the rigid transform N_j: x → N_j x. If a model trusts camera j's geometry, its predicted keypose should move the same way. If it relies on the wrist cameras (always calibrated), proprioception or appearance, the prediction should stay at the true target. This splits the 60/60 error into "followed the wrong geometry" and "other". Check `data/preprocessing/rlbench.py` (`apply_miscalibration`) and confirm the direction of the composition.

**Do (offline):** use exp 3 at 200k, 1b at 200k and control A at 200k (raw), plus exp 3 at 110k to see whether following decreases with training. Use the same validation samples and seeds as TODO 1. For each sample, hand and keypose, take:
- the prediction p̂ (robot/world frame, after unnormalisation);
- the ground truth p;
- the "followed" targets N_0 p and N_1 p.

Conditions:
- heavy, both external cameras;
- **only camera 0** perturbed;
- **only camera 1** perturbed;
- a magnitude sweep: 10/10, 20/20, 40/40 and 60/60, fixed magnitude and random direction.

**Report:**
1. Following coefficient α_j: the least-squares fit of (p̂ − p) ≈ α_j (N_j p − p), per model, condition and magnitude. α = 0 means ignore; α = 1 means follow fully. With both cameras perturbed, fit α_0 and α_1 jointly.
2. Distances |p̂ − p|, |p̂ − N_0 p| and |p̂ − N_1 p| (cm) and the share of samples closest to each.
3. The same for rotation, if cheap: does the predicted gripper orientation rotate with N_j's rotation?
4. Per task, where the sample count allows.
5. Clean-condition error as the floor.

Offline action error is reliable for exp 3, 1b and A. It is not for the older warm checkpoints (see Context), so leave those out.

**Results:**

**Reading:**

---

## TODO 7: activation patching — which path carries the damage from miscalibration?

**Status:** claimed (intervention agent)

**Why:** under 60/60, offline action error rises. Patching one component from a **clean** run into the **heavy** run of the *same* sample, with the same seeds and the same flow noise, shows how much of the damage flows through that component.

**Do (offline):** use exp 3 at 200k, 1b at 200k and control A at 200k (raw), with the same validation samples and seeds as TODO 1. Run `encode_inputs` for clean and heavy60 (cameras [0, 1]), then assemble mixed inputs and run `conditional_sample` with identical initial noise. Components to patch, clean into heavy, one at a time and then in useful combinations:
- **(a) Scene-token positions** in the head: `pcd` / `rgb3d_pos`, `fps_scene_pos`, `camera_summary_pos`. Geometry is restored, features stay heavy.
- **(b) Scene-token features:** `rgb3d_feats`, plus the FPS features without C-tokens. The encoder may already mix positions into these through its own 3D RoPE; check `modeling/encoder/multimodal/encoder3d.py` and say so.
- **(c) The C-tokens entering the head:** the camera-summary part of `fps_scene_feats`. These should be about identical clean vs heavy; it's a sanity check.
- **(d) The mechanism's outputs:**
  - **exp 3:** replay the clean delta_M at every layer call (hook `_predict_from_cam_feat` and replay the recorded values in call order);
  - **1b:** replay the clean adaLN camera term per layer (hook `_camera_adaln_signal`).
- **(e) Proprio and language:** these should be identical; a sanity check.

Also run the **reverse** direction (heavy into clean) for (a), (b) and (d): how much damage each component causes on its own.

**Report** per model: clean error, heavy error, and for each patch the patched error and the **fraction of the degradation recovered**, (heavy − patched) / (heavy − clean), using position L2, accuracy within 1 cm and rotation L1. A figure: bars per component, models side by side. Note that (a) and (b) together should recover close to 100%; check it as a consistency test.

**Results:**

**Reading:**
