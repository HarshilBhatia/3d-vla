# TODOs: what have exp 3 and 1b learned?

A working queue of analyses. A subagent works through it top to bottom; new items are appended while it runs. Each item gets its results written directly under it.

## How to work this list (read first)

1. **Re-read this file before starting each item.** Items are added while you work.
2. Take the **first item whose status is `open`**. Set it to `in progress (agent)`, commit, then do it.
3. When done: write **Results** (numbers in tables, where files and figures live) and a short **Reading** (what it means, what it rules out) under the item. Set status to `done`, then commit.
4. If an item is blocked or ambiguous, set it to `blocked: <reason>`, write what you tried, and move on to the next one.
5. When no `open` items remain, stop and report.

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

**Status:** in progress (agent)

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

**Reading:**

---

## TODO 2: what the 1b adaLN signal encodes

**Status:** open

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

**Why:** hypothesis: robust models move attention away from the miscalibrated external cameras (0, 1) onto the wrist cameras (2, 3, always calibrated) when geometry is inconsistent. Last week's probe showed blacking out the external cameras is catastrophic (VideoPooled 17), so they still matter; this measures how the balance shifts.

**Status:** open

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
