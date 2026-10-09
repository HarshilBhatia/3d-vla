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
- GPU work goes through Slurm: `--partition=shubhamlong,all`, 1 GPU, and `--exclude=grogu-4-3,grogu-4-13,grogu-3-5,grogu-3-10,grogu-3-15,grogu-3-20,grogu-3-25,grogu-3-30,grogu-4-18,grogu-1-20,grogu-4-23`. Those nodes are slow Turing cards, broken or mislabelled.
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

**⚠ Loader bug, fixed 7 Oct in `5ef5cf3`.** `load()` in `scripts/eval/diagnose_delta_m.py`, which most analysis scripts import or copy, ran every history model at **K=1** instead of its trained K=5. `visual_num_history` is eval-owned, so the checkpoint overlay skipped it. The loader also kept `view_align_cameras=[0,1]` for old-schema checkpoints (VideoPooled 84k, DeltaM 140k), which online eval runs on all cameras. **Every offline result produced before the fix is suspect**:
- TODO 1, including the "history extractor collapses the C-tokens" finding;
- TODO 3's history-model numbers;
- `results/diagnose_delta_m/wk1006.json`;
- any in-progress run.

Re-run with the fixed loader. If you copied `load()`, apply the same two fixes. Check the `[load] ... visual_num_history=` line it now prints.

**Code pointers:**
- delta_M: `TransformerHead._predict_from_cam_feat` (head) and `HistoryFeatureExtractor._delta_m_from` (Direct), plus the layer-wise loop in `TransformerHead.forward` (`modeling/policy/base_denoise_actor.py`).
- adaLN: `_forward_camera_adaln` and `_camera_adaln_signal`.

---

## TODO 1: delta_M per layer and per sample (exp 3, with VideoPooled and warm random history as references)

**Status:** done (re-run at K=5 after `5ef5cf3`)

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

**Re-run at K=5 with the fixed loader (`5ef5cf3`); every `[load]` line reads `visual_num_history=5`, and the extractor saw 5 frames.** Every delta_M statistic below matches the earlier K=1 run to within 1e-3, which is expected if the C-tokens are collapsed (see below). Three things did change. (a) VideoPooled now applies its delta_M to all 4 cameras, as online eval does; its wrist cameras get the same constant as cameras 0 and 1 (0.68 from identity at CA). (b) Offline pos_l2 is now sane for the old warm checkpoints: VideoPooled 0.065 → 0.012 and warm random history 0.033 → 0.013 (clean), so the "7 cm VideoPooled" action error in Context was this loader bug. (c) The C-token collapse is slightly less extreme (see below).

Scripts: `scripts/analysis/delta_m_per_layer.py` (records, Slurm job 3993409), `scripts/analysis/plot_delta_m_per_layer.py` (stats and figures), `scripts/analysis/delta_m_trunk_activity.py` (C-token spread and live trunk units, job 3993434). Raw tensors: `results/analysis/delta_m_per_layer/<model>.pt`, shaped (853 samples, 5 flow steps, 6 calls, 4 cams, 6, 6), with tasks and per-batch pos_l2. Stats: `results/analysis/delta_m_per_layer/summary.json` and `trunk_activity.json`. Figures: `figures/dm_layer_*.png` (see `figures/README.md`).

Setup: all 853 val samples (13 tasks), clean and heavy60 on cameras [0,1]. The first 320 samples are the old script's 20 batches; on those, the mean over all calls reproduces `wk1006.json` exactly (exp 3 200k 0.886, 110k 0.765, VideoPooled 0.381, warm 0.354). One deviation from the old script: I reseed after preprocessing, because heavy60 draws its rotation axes from the torch RNG. In `diagnose_delta_m.py` that means **clean and heavy60 did not share flow noise**, so its action-error comparison carries noise mismatch. Its pre-CA delta_M comparison is unaffected.

**Call structure (all four models are layer-wise "refine").** Each flow step makes 6 head calls: `init` (from `view_align_predictor`, computed but unused in the layer-wise branch), CA, then SA0–SA3. init, CA and SA0 read the same C-tokens and are bit-identical. The old "average over every call" therefore triple-counted the CA value and included one unused prediction. Wrist cameras are exactly identity, except in VideoPooled, whose old-schema config corrects all cameras.

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
- **The C-tokens entering the head are collapsed.** The HistoryFeatureExtractor's current-frame input varies across samples (per-channel std 0.025–0.048; camera 0 vs 1 distance 0.4–0.8). Its output, the head's C-tokens, has per-channel std about 1e-5 to 2e-4, and cameras 0 and 1 are equal to within 4e-3, in all four models, at K=5 as at K=1. The pre-CA delta_M is therefore one learned constant matrix, shared by both cameras and every sample.
- **The delta_M trunk (Linear + ReLU, 120 units) is almost dead.** About 1 unit is ever active at CA/SA0, 2 at SA1, fewer than 1 at SA2, and none at SA3. At SA3 delta_M equals exp(skew(bias)) of `camera_predictor` exactly (difference 0.0). Each layer's delta_M is thus confined to a family with at most two parameters around a bias rotation.

**Reading:**
- **delta_M is a learned, per-layer fixed basis change, not a per-scene or per-camera estimate.** It does vary with depth, and strongly: exp 3 applies a different large rotation of the RoPE basis at SA1 (the clamp-limited maximum) than at CA/SA0 and SA2/SA3. But at any given layer it is close to constant across samples, flow steps, cameras and calibration. This replaces the earlier "roughly constant": delta_M is constant within a layer, not across layers.
- **The "C-tokens are appearance only" explanation was too generous.** The C-tokens reaching the head carry essentially no per-sample information at all; the HistoryFeatureExtractor collapses them in every pooled model checked. The only per-sample signal in delta_M enters through SA mixing into the C-token positions (SA1, SA2). It is small: at most 0.15 RMS against rotations of size 2–3. It is mostly task identity (η² 0.5–0.8), and it is close to blind to 60/60 miscalibration (≤ 1°).
- **This rules out delta_M doing camera-specific geometric correction in any of these models.** exp 3's robustness gain over VideoPooled cannot come from delta_M tracking the miscalibration. If delta_M matters, it is as a learned RoPE re-mixing per layer; TODO 4 should look at the SA1 matrix in particular. Next checks: why the extractor output collapses (post-LN blocks with large shared camera/current-frame embeddings are a candidate), and whether the dead trunk is a training artifact.
- **Caveat.** The η² values describe very small spreads at SA2 and are noisy there; read SA1 as the meaningful one. The offline pos_l2 values recorded with each model are not re-validated (see Context on the older warm checkpoints).

---

## TODO 2: what the 1b adaLN signal encodes

**Status:** done (re-run at K=5 after `5ef5cf3`)

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

**Re-run at K=5 with the fixed loader (`5ef5cf3`, `[load] visual_num_history=5`).** Every number below matches the earlier K=1 run to within about 1e-3, which is expected because the C-tokens entering the head are collapsed.

Scripts: `scripts/analysis/adaln_signal.py` (records, Slurm job 3993408), `scripts/analysis/plot_adaln_signal.py` (stats and figures). Raw: `results/analysis/adaln_signal/exp1b_200k.pt`, holding the camera term, time_embs and mean C-token per (sample, step, signal), plus per-AdaLN scale/shift norms. Stats: `results/analysis/adaln_signal/exp1b_200k.json`. Figures: `figures/adaln_1b_*.png`.

Setup: 1b at 200k (EMA), all 853 val samples (13 tasks), clean / train / heavy60 on cameras [0,1], with flow noise reseeded so it matches across conditions. Offline pos_l2: clean 0.0100, train 0.0099, heavy60 0.0152. Each flow step computes 5 signals: one before CA (used by both CA layers and SA0), one after each of SA0–SA2 (used by SA1–SA3), and one after SA3 (used by the position and rotation heads). Every one of the 20 AdaLN calls received exactly the recorded signal (0 mismatches).

The camera term is recomputed in fp32. A first pass took it as `full − time_embs` from bf16 tensors, and the rounding error on time_embs (norm 10–16) appeared as a fake step dependence and miscalibration response at layer 0.

Note that `time_embs` is the denoising-time embedding **plus the current-gripper proprio embedding**. It is therefore sample-specific: its across-sample spread is 0.58 of its norm, with task η² 0.54. "time-only" below means "without the camera term", not "sample-independent".

1. Strength of the camera conditioning, ‖full − time-only‖ / ‖time-only‖, mean over samples and steps:

| AdaLN group (signal) | scale | shift |
|---|---|---|
| CA attn/ffw (pre-CA signal) | 0.07–0.08 | 0.09–0.14 |
| SA0 attn / ffw (pre-CA signal) | 0.13 / 0.07 | 0.27 / 0.08 |
| SA1–SA3 attn | 0.15–0.17 | 0.31–0.38 |
| SA1–SA3 ffw | 0.12–0.14 | 0.18–0.21 |
| rotation / position heads | 0.07–0.10 | 0.05–0.14 |

The ratio grows over the flow steps, because time_embs shrinks from norm 15.9 at step 0 to 10.2 at step 4 while the camera term stays fixed. Example: SA1 ffw scale goes 0.09 → 0.29.

2. Camera term MLP_c(mean C) per signal:

| Signal | ‖term‖ | across-sample RMS / ‖term‖ | task η² | centred cosine, within / across task |
|---|---|---|---|---|
| pre-CA (CA, SA0) | 1.91 | **0.000** (spread 6e-6) | – | – |
| after SA0 (SA1) | 3.95 | 0.16 | 0.71 | 0.53 / −0.02 |
| after SA1 (SA2) | 3.78 | 0.21 | 0.61 | 0.45 / −0.02 |
| after SA2 (SA3) | 3.24 | 0.35 | 0.44 | 0.56 / −0.03 |
| after SA3 (heads) | 1.21 | 0.94 | 0.33 | 0.62 / 0.21 |

Raw cosine between samples is 0.93–1.0 at SA1–SA3: a large shared component with a sample-specific part on top. The mean C-token entering the head has across-sample spread 1e-4 of its norm, the same collapse found for exp 3 and the delta_M models in TODO 1.

3. Miscalibration: ‖term_clean − term_miscal‖ on the same sample, as a fraction of the term's norm [and of the across-sample spread]:

| Signal | train | heavy60 |
|---|---|---|
| pre-CA | 0 (exact) | 0 (exact) |
| SA1 | 0.002 [0.009] | 0.013 [0.08] |
| SA2 | 0.003 [0.013] | 0.029 [0.13] |
| SA3 | 0.005 [0.013] | 0.045 [0.12] |
| heads | 0.006 [0.006] | 0.054 [0.05] |

4. Depth, cosine between consecutive camera terms: pre-CA→SA1 0.50; SA1→SA2 0.99; SA2→SA3 0.98; SA3→heads 0.72.

5. Flow steps: the pre-CA term is exactly constant across steps. At SA1–SA3 the RMS deviation across the 5 steps is 12–14% of the term's norm, and 7% at the heads; its norm does not trend with step. The camera term's share of the modulation does grow with step, as in item 1.

6. PCA (`figures/adaln_1b_camera_term_pca.png`, SA1 and heads): tasks form separate clusters. At the heads, a few tasks (handover_item, handover_item_easy, push_box) sit far from a dense common cluster. Layer 0 is not plotted because it has no spread: its "PCA" is float noise of order 1e-6.

**Reading:**
- **1b's first camera signal is a learned constant bias, not conditioning.** The C-tokens entering the head are collapsed, as in exp 3, so the pre-CA term is identical for every sample, step and calibration. All sample-specific camera conditioning in 1b comes from the C-token slots *after* they have been rewritten by SA. By then they function as extra learned summary tokens of the whole SA sequence, not as camera descriptors.
- **After SA, the term is a moderate, mostly-task signal.** It shifts the SA modulation by 15–38% of the non-camera part (more in shift than in scale). It varies across samples by 16–35% of its norm, and 45–70% of that variation is task identity; within-task samples also share direction (centred cosine about 0.5). It is a nudge on top of time+proprio, not the dominant part of the modulation.
- **It barely sees miscalibration.** Heavy 60/60 moves the SA terms by 1–5% of their norm, about 8–13% of their across-sample spread. Training-level miscalibration moves them by about 1% of the spread. The reaction is real (the pre-CA term is exactly 0) and must come through SA attention to miscalibrated scene positions, but it is small. As with exp 3's delta_M, this argues against 1b's robustness coming from an explicit miscalibration signal in the adaLN path.
- **Pattern shared with exp 3.** In both models the first conditioning is a constant, and the per-sample information enters at SA1 with a large change of direction (cosine 0.50 here; the delta_M rotation in TODO 1). Both point to the HistoryFeatureExtractor collapse as the common bottleneck worth checking next.

---

## TODO 3: where trajectory queries attend (external vs wrist cameras), clean vs 60/60

**Status:** done

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

**Re-run at K=5 (7 Oct):** the first run (job 3993279) loaded every history model at K=1, because `visual_num_history` is an eval-runtime key that the offline loader didn't overlay. The numbers below are the re-run at each checkpoint's own K=5, as online eval uses (job 3993406, `--faithful`, via TODO 8). What changed: exp 3 and 1b match K=1 to ≤0.01 pp. Control A and ema20k shift by ≤1.4 pp in clean share and ≤0.2 pp in Δ. ema20k's offline pos_l2 drops from 2.95 to 1.06 cm. No conclusion changes.

Script `scripts/analysis/camera_attention.py` (GPU; K=5 run via `scripts/analysis/todo8.slurm PART=attn`; first K=1 run `camera_attention.slurm`); summary and figures from `scripts/analysis/plot_camera_attention.py`. Raw per-(sample, step, layer, head, query, key-group) masses: `results/analysis/todo8/camera_attention/<model>.pt` (K=5; aggregates in `results/analysis/camera_attention_k5/summary.json` and `summary_table.txt`). The K=1 run is kept in `results/analysis/camera_attention/`. No model code changed: the script wraps `multi_head_attention_forward` and recomputes softmax(QKᵀ/√d) with the same RoPE in fp32 for the trajectory-query rows. Check: the recomputed layer output matches the module's own fp32 output to ≤1.5e-6 relative on every layer and model (≈1–3% against the bf16 autocast output, which is bf16 rounding).

Data: all 853 val samples, same seeds as TODO 1 (seed 1000+i before preprocessing, 5000+i after), so FPS draws and flow noise are identical across conditions. Averaged over 5 flow steps, 6 heads and the 2 trajectory queries (one per arm). Weights: exp 3 and 1b EMA, control A raw, ema20k_randhist EMA. "External share" = attention on cam 0+1 scene tokens / attention on all scene tokens (cams 0–3). In cross-attention that is all of the attention; in SA and the output heads scene tokens get 88–97% of the trajectory-query mass (C-tokens 1–4%, registers 1–4%, camera token and other trajectory tokens <1%).

**FPS keys are mostly wrist tokens.** Of the 204 FPS scene tokens in SA, only these come from cameras 0+1 (identical clean vs heavy60, since `scene_sampling: fps` samples by feature distance):

| Model | cam0 / cam1 / cam2 / cam3 FPS tokens | ext token share | samples with 0 external FPS tokens |
|---|---|---|---|
| exp 3 200k | 20.1 / 18.3 / 90.3 / 75.3 | 18.8% | 0% |
| 1b 200k | 12.5 / 7.8 / 101.2 / 82.5 | 10.0% | 4.8% |
| control A 200k | 12.5 / 8.5 / 105.2 / 77.9 | 10.3% | 2.8% |
| ema20k_randhist 15k | 14.9 / 10.4 / 99.0 / 79.7 | 12.4% | 0.7% |

**External share, clean → heavy60, Δ in percentage points (± SE over the 853 paired samples):**

| Layer | exp 3 | 1b | control A | ema20k_randhist |
|---|---|---|---|---|
| ca0 (dense) | 54.5 → 54.8, **+0.32** ±0.04 | 51.7 → 51.8, +0.07 ±0.06 | 48.8 → 49.5, **+0.73** ±0.10 | 46.5 → 47.0, +0.50 ±0.06 |
| ca1 (dense) | 52.6 → 53.2, +0.58 ±0.05 | 49.6 → 49.7, +0.11 ±0.06 | 48.9 → 50.7, **+1.74** ±0.11 | 44.4 → 45.0, +0.64 ±0.07 |
| sa0 | 26.4 → 26.6, +0.25 | 11.9 → 12.2, +0.33 | 11.4 → 11.5, +0.09 | 15.5 → 15.5, +0.04 |
| sa1 | 29.7 → 29.6, −0.01 | 14.1 → 14.1, +0.01 | 11.0 → 11.0, +0.07 | 16.6 → 16.6, +0.03 |
| sa2 | 30.3 → 29.8, −0.44 | 11.7 → 11.9, +0.23 | 12.9 → 12.8, −0.08 | 14.4 → 13.2, **−1.14** |
| sa3 | 25.0 → 24.6, −0.47 | 11.3 → 10.8, −0.52 | 13.2 → 12.9, −0.31 | 12.6 → 11.6, −1.01 |
| pos0 | 19.9 → 18.6, **−1.25** | 8.7 → 7.7, −1.06 | 5.9 → 4.7, −1.11 | 8.7 → 7.3, −1.46 |
| pos1 | 13.4 → 11.7, **−1.73** | 4.4 → 3.2, −1.14 | 4.4 → 3.2, −1.24 | 6.7 → 4.9, **−1.76** |
| rot0 | 20.5 → 20.5, −0.01 | 10.3 → 10.3, +0.05 | 9.6 → 9.6, +0.04 | 13.1 → 13.0, −0.07 |
| rot1 | 21.6 → 21.6, +0.05 | 10.4 → 10.4, −0.00 | 9.8 → 9.7, −0.16 | 11.9 → 11.9, +0.02 |

SE is 0.02–0.11 pp everywhere, so every |Δ| ≥ 0.3 pp is well resolved. Train-style noise (medium bias + 3°/1 cm) moves every entry by ≤0.14 pp.

Other measurements (all in `summary_table.txt`):
- **Attention per external key, relative to uniform** (share ÷ token share), clean: exp 3 attends to external FPS tokens 1.3–1.6× their count in SA, against 1.0–1.4× for the others. In pos1 every model under-weights them (0.43–0.71×), and heavy60 lowers that further (exp 3 0.71 → 0.62; 1b 0.44 → 0.32; A 0.43 → 0.31; ema20k 0.54 → 0.40).
- **Entropy** (nats; clean CA ≈6.1–6.6, SA ≈5.0–5.2, pos1 ≈4.0–4.4): heavy60 makes attention slightly *more focused*, never more diffuse: Δ −0.02 to −0.07 nats in CA, ≤ −0.06 in SA, −0.02 to −0.05 in the position head, ≈0 in the rotation head. The largest drops are exp 3 sa2 (−0.063) and control A ca1 (−0.065).
- **C-tokens, registers and camera token** together get 3–8% of the mass and barely move (|Δ| ≤ 0.6 pp everywhere except exp 3 sa2, where C-tokens 0+1 go 1.0% → 2.4%, and ema20k sa2 +0.5 pp).
- **Per head:** the shift lives in the even heads (0, 2, 4), up to −4.7 pp in the position head and up to +4.3 pp in control A's ca1. Heads 1, 3 and 5 move ≤1.2 pp in every model and layer. With `rope3d`, head h of 20 dims covers one axis (x, x, y, y, z, z) and the odd heads hold the lower-frequency half of the RoPE bands, so a 60 cm shift barely changes their logits.
- **Across flow steps:** the CA increase grows from the first step to the last, e.g. control A ca1 +1.2 → +2.4 pp and exp 3 ca1 +0.35 → +0.90 pp. Position-head drops are roughly constant across steps.
- Offline pos_l2 clean → heavy60 (cm), K=5: exp 3 1.03 → 1.40, 1b 0.99 → 1.53, A 1.19 → 2.03, ema20k 1.06 → 1.57. At K=1 ema20k showed 2.95 cm clean, and A 1.37 cm. That earlier "suspect" ema20k error was the K=1 loading.

Figures (`figures/README.md`): `cam_attn_external_share.{png,svg}` shows external share per layer (clean solid, heavy60 dashed, dotted = token share), Δ ± SE, and Δ entropy, with the 4 models overlaid. `cam_attn_sa_mass_breakdown.{png,svg}` stacks trajectory-query mass by key group for the SA and output-head layers, clean vs heavy60.

**Reading:**
- **The hypothesis does not hold at the level of attention mass.** No model moves meaningful attention from the external to the wrist cameras under 60°/60 cm. Shifts are ≤2 pp absolute everywhere. In cross-attention they go the *wrong* way (slightly more attention to the miscalibrated cameras), and the largest of those is in control A, the least robust model offline. The only consistent move towards the wrist cameras is in the position head (−1.1 to −1.9 pp, i.e. 15–30% relative of an already small external share). It is the same size in control A, which has neither delta_M nor adaLN, so it is generic RoPE-geometry behaviour, not something exp 3's or 1b's mechanism learned. Exp 3 and 1b don't reweight more than A, and robustness rankings are not explained by attention balance.
- **The trajectory queries already mostly see wrist cameras, by construction rather than in reaction to miscalibration.** The FPS sampler keeps only 10–19% of its 204 tokens from cameras 0+1, and in up to 4.6% of samples none at all. So in SA and the output heads, 80–95% of the scene attention is on wrist tokens under clean cameras too. Only the two cross-attention layers see the external cameras at 50%. That is a structural source of robustness that all four models share. Combined with last week's blackout result (external cameras still matter), the external information that matters most likely enters through cross-attention and the C-tokens, not through the SA trunk's attention to external FPS tokens.
- **Exp 3 does differ, but in the clean condition, not in its response.** Its SA and output heads put about twice the share on external tokens (26–30% vs 11–17% in SA; 21% vs 10–13% in the rotation head). That is mostly because its sampler keeps more external tokens (18.8% vs 10–12.5%); per token it attends to them only a little more (1.3–1.6× vs 1.0–1.4×). That fits a model whose layer-wise delta_M rewrites the external cameras' RoPE kernel (TODO 4) and so leans on those tokens more. Its clean-to-heavy60 shift is still no larger than control A's.
- **Rules out:** "robustness via re-routing attention to calibrated cameras" as the mechanism separating exp 3 / 1b from control A. Heavy miscalibration makes attention slightly sharper, not more diffuse, and only in the high-frequency RoPE heads, which suggests the 60 cm shift moves external keys out of the high-frequency kernel's peak rather than the model deciding to ignore them.
- **Caveats:**
  - Attention mass is not influence. Attention × ‖value‖ or activation patching (TODO on patching) is needed to say how much the external tokens contribute to the output.
  - The SA external share is confounded by how many external tokens the FPS sampler keeps, which differs per model. The per-token enrichment controls for that.
  - Averaging over heads hides the larger even-head effects (up to ±5 pp).

---

## TODO 4: what exp 3's delta_M does to the RoPE kernel

**Status:** done

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

Script: `scripts/analysis/rope_kernel.py` (CPU, pure linear algebra). It reads TODO 1's K=5 tensors (`results/analysis/delta_m_per_layer/{exp3_200k,videopooled84k}.pt`, camera 0; camera 1 is identical) and takes, per layer, the sample-and-step mean delta_M projected back onto SO(6). Outputs: `results/analysis/rope_kernel/summary.json` and `kernels.pt`. Figures: `figures/rope_kernel_displacement.png` and `figures/rope_delta_m_heatmaps.png`.

Model of the logit, from `RotaryPositionEncoding3D` and `_apply_delta_M`:
- RoPE runs on the full 120-dim q/k before the head split: 40 dims (20 frequency bands) per axis.
- The trajectory query uses the plain basis [cos, sin](ω·q). Each scene/FPS key uses delta_M·[cos x, cos y, cos z, sin x, sin y, sin z](ω·p), with the same 6×6 for all 20 bands.
- Per band and axis block, the relative operator on a q/k channel pair is M(α, β) with α = c_q·c_k + s_q·s_k and β = c_q·s_k − s_q·c_k. "Kernel" below is the even part α, averaged over bands (and blocks where stated).

**0. The frequencies matter more than delta_M.** Positions are in metres and ω_k = 10000^(−k/20) rad/m, so the highest band is 1 rad/m (wavelength 6.3 m) and the others go down to 1.6e-4 rad/m. Over the whole ±60 cm range, plain RoPE's kernel (all bands and blocks) moves only from 1.000 to 0.995 (0.9999 at 10 cm, 0.998 at 40 cm). Only the top ~3 bands rotate by more than 0.1 rad across the workspace. In this model RoPE is a very coarse, almost linear position signal, and most bands are effectively constant.

1. Kernel vs displacement. Kernel at d = 0 and its change from 0 to ±60 cm, for a query at the origin (a query averaged over the workspace gives the same picture):

| delta_M | K(0) | K(0) − K(±60 cm), x / y / z | top band (1 rad/m) at d=0: phase, gain |
|---|---|---|---|
| identity | 1.000 | 0.005 / 0.005 / 0.005 | 0, 1.00 |
| exp 3 CA (= SA0) | 0.971 | 0.005 / 0.004 / 0.005 | −0.21 rad, 0.98 |
| exp 3 SA1 | **0.541** | 0.003 / −0.000 / 0.005 (monotone ramp, no peak) | **−0.81 rad, 0.87** |
| exp 3 SA2 / SA3 | 0.9997 / 0.9998 | 0.005 | −0.02 / −0.01 rad, 1.00 |
| VideoPooled CA | 0.911 | 0.005 / 0.005 / 0.003 | −0.07 rad, 0.75 |
| VideoPooled SA1 | 0.999 | 0.005 | 0.00, 0.98 |

delta_M does not blur or sharpen the kernel in any meaningful way, because there is almost no kernel shape to change. What it does is add a **constant offset and attenuation**. The low bands of every scene key are about [1,1,1,0,0,0] at every position, and delta_M rotates that constant vector partly into the sin slots. Against a plain query this lowers α equally for all displacements: to 0.54 at exp 3's SA1, 0.97 at CA, 0.91 at VideoPooled CA.

At SA1 the even kernel becomes a gentle linear ramp across ±60 cm (0.52 → 0.56 along x and y), not a shifted peak. The phase offset in the top band (−0.81 rad) would correspond to 81 cm in that band but to metres-to-kilometres in the lower bands. **It is not a coherent spatial offset**; it is a fixed relative rotation of each scene key's channel pairs. Sample-to-sample delta_M variation is irrelevant here: individual SA1 samples change the kernel by at most 0.002.

2. Structure (`figures/rope_delta_m_heatmaps.png`; rotation planes and loadings are in summary.json):
- **exp 3 CA** (‖ΔM−I‖ 0.70): two rotation planes, 25° and 13.5°. Mostly cos↔sin mixing (energy 0.62, vs 0.19 cos→cos and 0.26 sin→sin), with weak cross-axis terms (off-diagonal 2×2 axis blocks 0.19–0.30 vs 1.36–1.38 on the diagonal).
- **exp 3 SA1** (2.66, clamp-saturated): planes of 106°, 58° and 14°. Cross-axis blocks (0.42–0.96) are as large as the diagonal ones (0.68–0.99), so it scrambles x/y/z and cos/sin together. Examples: cos x ← 0.82 sin y; cos y ← −0.69 sin z; sin z ← 0.63 cos y.
- **exp 3 SA2/SA3** (0.28): planes of 11° and 3°, mainly cos z ↔ sin x and cos x ↔ sin x (entries ±0.12). Nearly identity.
- **VideoPooled CA**: planes of 27.5° and 4°, loading on cos x, y, z ↔ sin z (cos y ← −0.31 sin z). **VideoPooled SA1–SA3** are close to identity (0.07–0.08).

3. Rigid 60°/60 cm miscalibration, modelled as the preprocessor applies it (T·E, so world points move as P' = R·P + t; mean point displacement 1.15 m in this workspace). 128 scene points × 128 queries × 128 random perturbations:

| delta_M | RoPE operator change, all blocks (RMS) | per block x / y / z | top-3-band change | even-kernel change, abs |
|---|---|---|---|---|
| identity | 0.340 | 0.207 / 0.200 / 0.182 | 0.849 | 0.017 |
| exp 3 CA | 0.340 | 0.201 / 0.197 / 0.191 | 0.849 | 0.015 |
| exp 3 SA1 | 0.340 | 0.218 / 0.212 / 0.153 | 0.849 | 0.029 |
| exp 3 SA2/SA3 | 0.340 | 0.207 / 0.198 / 0.184 | 0.849 | 0.017 |
| VideoPooled CA | 0.340 | 0.205 / 0.196 / 0.188 | 0.849 | 0.014 |

The total change is identical for every delta_M, and **this holds exactly, not approximately.** delta_M is orthogonal and the same for both P and P', so ‖ΔM(f(P')) − ΔM(f(P))‖ = ‖f(P') − f(P)‖ for every point, query and perturbation. delta_M can only move the perturbation between axis blocks and between the α and β parts. It cannot shrink it.

**Reading:**
- **Its learned delta_M does not make exp 3's attention less sensitive to miscalibration.** By orthogonality it cannot reduce the total positional perturbation a scene key receives. At SA1 it even roughly doubles the change in the even (content-free) kernel term (0.029 vs 0.017). Any robustness from delta_M would have to come from where it routes the perturbation, relative to channels that q/k content then down-weights. That cannot be read from the 6×6 alone: it needs q/k content (TODO 7's patching is the right test).
- **What delta_M actually does is content-like, not geometric.** Most RoPE bands are effectively constant over the workspace (wavelengths 6 m to 40 km). On those bands delta_M is a fixed per-camera rotation of each scene key's 120-dim channel pairs relative to the queries. That is a learned key-side linear map, the same for every scene token of a camera. At SA1 in exp 3 it is a large rotation that cuts the scene keys' content-free alignment with trajectory queries to 0.54 and mixes x/y/z. It acts more like a learned layer-specific key projection than a camera correction.
- **The coarse RoPE frequencies are a finding in their own right.** With ω_max = 1 rad/m, plain 3D RoPE barely separates points 10 cm apart (kernel 0.9999 vs 1). A 60°/60 cm miscalibration, by contrast, moves the top bands by about 1 rad. Any precise geometric use of position has to happen through q/k content riding on the top 2–3 bands, which suggests checking higher-frequency RoPE as a model change.
- **Caveat.** The "kernel" averages out content. The odd (β) part, which is roughly linear in displacement in the low-frequency regime and is what learned q/k can exploit, is summarised here only for the top band (in summary.json, `top_band_x`).

---

## TODO 5 (larger experiment): linear probes — do the C-tokens know the true camera pose, and the injected miscalibration?

**Status:** done

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

*Setup.*
- **Data.** 3000 random training-split rows (1053 episodes, all 13 tasks), seed 0, plus all 853 validation rows (117 episodes). Train and val episodes are disjoint. The α/C grid is picked by 5-fold CV grouped by episode on the training split.
- **Inference.** Each model runs at its trained K = 5 with the fixed loader (`[load] … visual_num_history=5` for all five; extraction was redone after `5ef5cf3`). Flow noise and history draws are the same across conditions.
- **Features.** Per camera, stored in fp16. Depths:
  - `enc`: encoder `camera_summaries`;
  - `hist`: `refined_frames[:, -1]`;
  - `SA1`–`SA4`: C-tokens after each shared SA layer, at the first and last of the 5 flow steps;
  - delta_M: the four distinct per-layer predictions (input depth hist, SA1, SA2, SA3), flattened `ΔM − I`;
  - 1b: the adaLN camera term `MLP_c(mean C)`, five calls (hist, after SA1–SA4), global, probed against both external cameras' targets concatenated.
- **Random-init floor.** The exp 3 config with seed-0 weights. The SigLIP2 vision and text towers are frozen pretrained weights in every model (they are not in the checkpoints), so the floor has those too.
- **Probe-only sweep.** `_sample_random_perturbation` is replaced in a dedicated preprocessor by a sampler with angle ~ U[0°, 60°] and translation length ~ U[0, 60 cm], random directions, drawn independently per (sample, camera) on cameras [0, 1].
  - Wrapping `_get_miscal_noise` records the applied 4×4 that left-multiplies the extrinsics. Targets are its world-frame rotation vector and translation.
  - A third condition (training style: medium group bias + 3°/1 cm) is used only for the binary probe. Its draws are 7.5°/5.6 cm on average.
- **Pose data check.**
  - The external cameras take exactly **6 distinct poses** each (G1–G6, one per episode). The wrist cameras vary continuously (2830/2229 distinct positions).
  - The workspace centre is the least-squares intersection of the 12 optical axes, **(0.248, 0.000, 0.954)** (every axis passes within 0.3 cm). Every G1–G6 camera sits **1.31 m** from it, so the distance target is constant and is dropped.
  - Poses form 3 elevations (10°/35°/60°) × 2 azimuths per side. So within the training data, (A) is a 6-way classification, and the leave-one-group-out (LOGO) probe is the real test.
  - LOGO: train on the 5 other groups (training split), test on the held-out group's val episodes, pooled over the 6 folds.
- **Feature check (fp32).** The `hist` collapse was re-checked from fp32 features for exp 3 and control A (`results/analysis/probes/fp32/`). Relative std across samples is 3e-4 and 1.5e-4, the group probe is at chance (0.21, 0.20), and exp 3's first delta_M has relative std 2e-5. So the collapse is not fp16 quantisation.

*(A) True camera pose.* Mean over cameras 0 and 1; C-tokens at the last flow step. Chance for the group probe is 0.17. A task one-hot gets 0.33, because each task uses 3 groups.

| Model | Group acc: enc / hist / SA1 / SA2 / SA3 / SA4 | LOGO pos err (cm): enc / hist / SA4 | LOGO view-dir err (°): enc / SA4 | Wrist-mean `enc` → ext. group | Best delta_M / adaLN group acc |
|---|---|---|---|---|---|
| exp 3 | 0.93 / **0.18** / 0.69 / 0.69 / 0.69 / 0.68 | 25 / 50 / 41 | 10 / 18 | 0.35 | 0.24 (delta_M) |
| 1b | 0.94 / **0.21** / 0.44 / 0.61 / 0.58 / 0.62 | 24 / 50 / 44 | 10 / 20 | 0.34 | 0.33 (adaLN) |
| control A | 0.94 / **0.20** / 0.46 / 0.56 / 0.58 / 0.77 | 22 / 50 / 30 | 9 / 13 | 0.34 | – |
| ema20k randhist | 0.95 / **0.23** / 0.59 / 0.61 / 0.71 / 0.77 | 21 / 50 / 32 | 9 / 14 | 0.33 | 0.23 (delta_M) |
| random-init | 0.97 / 0.98 / 0.97 / 0.98 / 0.96 / 0.96 | 20 / 20 / 22 | 8 / 9 | 0.31 | 0.70 (delta_M) |

- **LOGO references.** Predicting the mean of the 5 training poses: 50 cm / 22°. The nearest other training pose is 18 cm / 7.9° from the held-out one. LOGO R² for position at `enc` is 0.51–0.70; it is ≤ 0.31 everywhere past `hist` in trained models.
- **iid regression just mirrors classification.** Position error on held-out episodes is 8–12 cm at `enc` and 41 cm at `hist`; azimuth/elevation at `enc` is 2.4–3.6° (R² 0.70–0.82).
- **Wrist cameras' own pose (a continuous control).** Position R² is 0.72–0.74 at `enc` and 0.76–0.79 at SA4 (about 10–11 cm, 7–9°) in trained models, against 0.65/0.60 for random-init.
- **Shuffled labels.** Max over all models and features: group acc 0.27, pose R² 0.03.

*(B) Injected miscalibration* (probe-only sweep). Mean over cameras 0 and 1, last flow step. "Predict the mean" MAE: 15.5° for rotation magnitude, 15.3 cm for translation magnitude, 29.4° for the rotation vector and 29.7 cm for the translation vector.

| Model | Rotation-vector R²: SA1 / SA2 / SA3 / SA4 | Translation-vector R², SA4 | Rotation-magnitude R², SA4 | Translation-magnitude R², SA4 | Clean vs sweep acc: hist / SA2 / SA4 | Clean vs train-style acc, SA4 | Wrist-mean SA4 → ext. rotation vector | Best delta_M / adaLN, rotation vector |
|---|---|---|---|---|---|---|---|---|
| exp 3 | 0.06 / 0.16 / 0.17 / 0.16 | 0.12 | 0.04 | 0.00 | 0.50 / 0.58 / 0.60 | 0.53 | 0.00 | 0.00 |
| 1b | 0.01 / 0.06 / 0.08 / 0.11 | 0.06 | 0.00 | 0.00 | 0.50 / 0.57 / 0.56 | 0.52 | 0.00 | 0.00 |
| control A | 0.04 / 0.11 / 0.13 / 0.22 | 0.19 | 0.11 | 0.01 | 0.50 / 0.54 / 0.62 | 0.57 | 0.00 | – |
| ema20k randhist | 0.06 / 0.10 / 0.12 / 0.20 | 0.18 | 0.09 | 0.00 | 0.50 / 0.53 / 0.60 | 0.55 | 0.00 | 0.00 |
| random-init | 0.00 / 0.00 / 0.00 / 0.00 | 0.00 | 0.00 | 0.00 | 0.50 / 0.51 / 0.51 | 0.50 | 0.00 | 0.00 |

- **`enc` and `hist` are identical** under clean and miscalibrated cameras (max diff 0), so every (B) probe there is at chance by construction.
- **Best SA errors are small.** The best rotation-vector MAE is 26.8° (control A, SA4) against 29.4° for the mean, and the best rotation-magnitude MAE is 14.3° against 15.5°.
- **The first and last flow steps differ little** (dotted lines in the figures): within 0.06 in R² or accuracy.
- **Refits on a different node** (8 Oct, same features) move individual cells by ≤ 0.02 in accuracy/R² and ≤ 1.2 cm/°. This is BLAS non-determinism, largest on the collapsed `hist`/delta_M features; the figures and tables above are from the 7 Oct fit.

*Files.*
- Scripts:
  - `scripts/analysis/probe_features.py` (GPU, via `probe_features.slurm`; `--fp32` for full-precision features);
  - `scripts/analysis/fit_probes.py` (CPU, via `fit_probes.slurm`);
  - `scripts/analysis/plot_probes.py`.
- Saved features and labels: `results/analysis/probes/<model>.pt` (train/val × clean/train/sweep features, noise transforms, extrinsics, task, demo, group, row).
- All probe scores: `results/analysis/probes/fits/probe_results_<model>.json`. Full table: `results/analysis/probes/summary_table.md`.
- Figures: [`figures/probe_pose.png`](figures/probe_pose.png) and [`figures/probe_miscal.png`](figures/probe_miscal.png).

**Reading:**
- **The C-tokens that delta_M and adaLN read from know neither the camera pose nor the miscalibration.** In every trained model the history extractor's output (`hist`) is a constant: the group probe is at chance (0.18–0.23) and position error equals the predict-the-mean baseline. This confirms TODO 1's collapse at K=5 and fp32 from a different angle. The random-init model at `hist` still gets 0.98, so the collapse is learned, not architectural.
  - delta_M (exp 3, ema20k) and the 1b adaLN term are at or near chance for pose at every layer (≤ 0.24 and ≤ 0.33, the latter about task level), and at exactly 0 R² for miscalibration.
  - So neither mechanism has the information it would need to correct geometry. Robustness must come from elsewhere.
- **The encoder's C-tokens do encode pose. That comes from the frozen SigLIP features, and the model then discards it.**
  - `enc` classifies the group at 0.93–0.95. The random-init floor (same frozen SigLIP, random projection) is *better* at 0.97, so training adds nothing here.
  - Under LOGO, `enc` lands about 20–25 cm / 9–10° from the unseen pose: far better than the mean (50 cm / 22°), but no better than snapping to a neighbouring seen pose (18 cm / 8°). That is evidence of a smooth but coarse appearance-to-viewpoint map, not of interpolation to new poses.
  - The wrist-camera control is weak, as expected (0.31–0.35, the task level). Wrist tokens do encode their *own* continuous pose (R² ≈ 0.75).
- **Pose information partially returns in head SA, from attention to scene tokens.** C-tokens re-acquire camera identity (0.44–0.77 group acc by SA4) because they attend to FPS and scene tokens.
  - That is still well below `enc`. Under LOGO it is near the mean baseline (exp 3: 41 cm), except at control A's and ema20k's SA4 (30–32 cm, R² ≈ 0.3).
  - exp 3's delta_M is re-predicted from exactly these SA tokens, yet stays at chance: its predictor maps them to a near-constant.
- **(B) is not decodable before SA and only weakly decodable inside it.** The expected pattern holds: zero through `hist`, and a rise from SA1, where C-tokens attend to scene tokens at miscalibrated positions.
  - At its best (SA4), a linear probe explains 16–22% of the variance of the signed rotation and translation vectors in exp 3, control A and ema20k (11%/6% in 1b). The magnitude is essentially not linearly decodable (R² ≤ 0.11).
  - Clean vs a 0–60° / 0–60 cm sweep is only 56–62% accurate, and clean vs training-style noise 52–57%.
  - Control A, which has no delta_M and no adaLN, is the most decodable at SA4. So this signal is a passive side effect of RoPE attention over displaced points, not something either mechanism cultivates. Neither mechanism's own output carries any of it (R² 0.00).
- **Ruled out:** delta_M or the 1b adaLN term acting as a camera-pose or miscalibration estimator (they have neither signal), and pose being encoded continuously enough to generalise beyond the six training viewpoints.
- **Caveats.**
  - Pose has only 6 values per external camera, confounded with task (3 groups per task). The iid scores are classification scores; only LOGO speaks to continuous encoding, and it has 6 folds.
  - The probes are linear. A nonlinear probe could read the magnitude from the signed-vector components the linear probe finds (R² 0.16–0.22).
  - The miscalibration target is the world-frame left-multiplied error. A rotation about the world origin also moves the camera centre, so "rotation" and "translation" are not separate camera-frame quantities.
  - The random-init floor shares the frozen SigLIP towers, so it measures "SigLIP + random head", not a featureless model.

---

## TODO 6: under heavy miscalibration, does the model follow the miscalibrated cameras or ignore them?

**Status:** done

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

Scripts: `scripts/analysis/follow_miscal.py` (records predictions, GT and the applied N per camera; Slurm job 3993363 via `follow_miscal.slurm`) and `scripts/analysis/plot_follow_miscal.py` (fits and figures, CPU). Raw outputs: `results/analysis/follow_miscal/<model>.pt`. Fits with 95% bootstrap CIs over samples: `summary.json`. Printed table: `summary_table.txt`. Figures: `figures/follow_alpha_pos.png`, `figures/follow_beta_rot.png`, `figures/follow_closest_share.png`.

Setup: all 853 val samples, K = 5 history (run after `5ef5cf3`; the `[load]` line shows `visual_num_history=5` for every model). Seeds are as in TODO 1: torch reseeded per batch before preprocessing (1000+i) and again before sampling (5000+i). Every condition therefore shares the same history draw, the same flow noise, and the same N direction for a given camera. The camera-0-only and camera-1-only N are exactly the corresponding N from the both-cameras run.

Checks:
- **Frame and units.** Actions are absolute (`relative_action=False`), keypose only, quat xyzw. `compute_trajectory` unnormalises to world-frame metres, and `process_actions` passes the GT through unchanged in the same frame.
- **Composition direction.** `apply_miscalibration` does T_new = N · T (camera-to-world), so a world point x → N x. Checked on the first batch at 60/60: median |pcd_heavy − N x| is 1.2 / 1.3 cm for cameras 0 / 1 (bf16 unprojection), against 207 / 214 cm for N⁻¹x and 117 / 121 cm for x. Wrist cameras: 0.
- **Offset scale.** N rotates about the **world origin**, which is about 1 m from the workspace. So even 10°/10 cm moves the targets a lot: mean |N_j p − p| is 17.5 cm at 10/10, 34.8 at 20/20, 68.9 at 40/40 and 101.7 at 60/60.

Clean floor (pos L2 cm / rotation error °): exp 3 200k 1.03 / 3.1; exp 3 110k 1.15 / 3.5; 1b 1.00 / 3.0; A 1.20 / 3.6.

1. Following coefficient α, from (p̂ − p) ≈ α (N_j p − p), pooled over samples, hands and xyz. With both cameras perturbed it is a joint fit; 95% CI in brackets.

| Model | both 60: α₀ | both 60: α₁ | only cam 0 60: α₀ | only cam 1 60: α₁ | both 10: α₀ | **all 4 cams, same N, 10/10** | **all 4 cams, 60/60** |
|---|---|---|---|---|---|---|---|
| exp 3 200k | 0.0028 [0.0014, 0.0042] | 0.0016 [0.0004, 0.0029] | 0.0025 | 0.0018 | 0.0067 [0.002, 0.012] | **0.63** [0.61, 0.65] | **0.41** [0.40, 0.43] |
| exp 3 110k | 0.0037 | 0.0029 | 0.0034 | 0.0035 | 0.0066 | 0.62 | 0.41 |
| 1b 200k | 0.0020 | 0.0007 [−0.0005, 0.0021] | 0.0018 | 0.0003 | 0.0068 | 0.59 | 0.39 |
| A 200k | 0.0036 | 0.0003 [−0.0012, 0.0021] | 0.0035 | 0.0002 | 0.0062 | 0.68 | 0.39 |

The last two columns are an extra control I added: every camera, wrists included, gets the same N (camera 0's draw). This moves the whole scene rigidly while proprioception stays in the true frame. Over the sweep, α for this control is 0.63 → 0.56 → 0.48 → 0.41 for exp 3 200k, and similar for the other models (`summary.json`). Fitting against the prediction shift from the clean prediction (p̂_heavy − p̂_clean), instead of against the GT, gives the same α to within 0.002.

2. Distances at 60/60 with both cameras perturbed, in cm: |p̂ − p|, then |p̂ − N₀p| and |p̂ − N₁p|. The last column is the share of (sample, hand) pairs closest to the true p.

| Model | \|p̂ − p\| | \|p̂ − N₀p\| | \|p̂ − N₁p\| | mean shift from the clean prediction | closest to true p |
|---|---|---|---|---|---|
| exp 3 200k | 1.41 | 101.7 | 101.0 | 0.77 | 99.9% |
| exp 3 110k | 1.73 | 101.6 | 100.9 | 1.09 | 100% |
| 1b 200k | 1.54 | 101.8 | 101.1 | 0.87 | 99.9% |
| A 200k | 2.05 | 101.7 | 101.2 | 1.21 | 99.7% |

At 10/10 the share closest to the true p is still ≥ 98.4% for every model. Single-camera 60/60 pos L2 (camera 0 only / camera 1 only): exp 3 200k 1.24 / 1.23; 110k 1.44 / 1.51; 1b 1.34 / 1.27; **A 2.01 / 1.21**. Control A is hurt almost only through camera 0; exp 3 and 1b are symmetric.

3. Rotation following β, from log(R̂ Rᵀ) ≈ β log(R_Nj):
- External cameras only: |β| ≤ 0.005 at 60/60 in every model and condition, and every CI includes 0. At 10/10, |β| ≤ 0.04 with CIs of about ±0.05.
- All four cameras, same N: β is only 0.02–0.08. Even when the whole scene rotates, the predicted gripper orientation barely rotates with it.

4. Per task, α against the clean prediction at both-60: every task is below 0.011 for exp 3 200k. The largest is push_box, which has only 18 samples. Across all models the maximum is 0.023 (exp 3 110k push_box). Full table in `summary.json` (`per_task_alpha_vs_clean_pred`).

5. Clean floor: listed above. Degradation at 60/60 relative to clean: +0.38 cm for exp 3 200k, +0.58 for 110k, +0.54 for 1b and +0.85 for A.

**Reading:**
- **None of the models follow miscalibrated external cameras.** α ≤ 0.004 at 60/60 and ≤ 0.007 at 10/10. The prediction stays at the true target in more than 99.7% of cases, even though N_j p is 17–100 cm away.
  - The small positive α is real (the CIs exclude 0 for camera 0) but tiny. The directed component α·|N p − p| is about 0.3 cm of the 0.8–1.2 cm shift.
  - Most of the 60/60 degradation is therefore not "following the wrong geometry" but undirected error. TODO 7 traces where it comes from.
- **This is not because the models are blind to geometry.** When all four cameras agree on the wrong frame, the models do move: α ≈ 0.6 at 10/10, falling to 0.4 at 60/60, and 1b and A follow about as much as exp 3. So scene geometry matters when it is consistent. When the two external cameras disagree with the wrist cameras and with proprioception, all models take the calibrated side.
  - The rigid α < 1 is itself a compromise with proprioception, which stays in the true frame (see TODO 7: proprioception features are built by attending to scene xyz).
- **Orientation is anchored.** β ≈ 0 for external-only and ≤ 0.08 even for a rigid whole-scene rotation. Gripper orientation comes almost entirely from proprioception and the task prior, not from scene geometry.
- **Following does not separate the models, and changes little with training.** exp 3 110k → 200k: α 0.0037 → 0.0028, against a larger drop in the undirected shift (1.09 → 0.77 cm). exp 3's robustness edge over A is not "ignoring the cameras better"; all three ignore them. It is a smaller undirected error, plus A's lopsided dependence on camera 0.
- **Caveats.**
  - N rotates about the world origin, so the "followed" target for large rotations sits far outside the workspace. A model trained only on in-workspace targets could not follow fully even if it wanted to; the 10/10 column is the cleanest test.
  - The rigid control conflicts with proprioception by construction, so its α measures a trade-off, not pure trust in the scene.
  - Offline, one N draw per sample, keypose only.
  - Slurm note: `grogu-4-23` (RTX PRO 6000 Blackwell, sm_120) is not supported by the env's torch build, and `grogu-1-20` gave "CUDA device busy" twice. I excluded both in my launchers.

---

## TODO 7: activation patching — which path carries the damage from miscalibration?

**Status:** done

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

Scripts: `scripts/analysis/activation_patching.py` (Slurm job 3993425 via `activation_patching.slurm`) and `scripts/analysis/plot_activation_patching.py` (fractions, bootstrap CIs, figure). Raw per-sample metrics for every patch: `results/analysis/activation_patching/<model>.pt`. Aggregates: `summary.json` and `summary_table.txt`. Figure: `figures/patch_fraction_recovered.png`.

Setup:
- **Samples and preprocessing.** All 853 val samples at K = 5 (`[load]` shows 5 for exp 3, 1b and A). Each loaded sample is preprocessed twice, clean and heavy60 on cameras [0, 1], with the same torch seed, so the history draw is shared.
- **Encoding.** `encode_inputs` runs once per condition under bf16 autocast.
- **Flow noise.** One initial noise per batch from a dedicated generator (seed 5000+i). Every run calls `conditional_sample` on a copy of that same noise.
- **Mechanism replay.** Hooks record or replay, in call order:
  - exp 3: every `_predict_from_cam_feat` output, 6 calls × 5 steps;
  - 1b: every `camera_adaln_proj` output, i.e. the camera term of `_camera_adaln_signal`, 5 calls × 5 steps.
- **Exactness passed for all three models.** Max |Δ prediction| is **0.0** for patch-nothing vs the heavy run and for patch-every-input vs the clean run.

**Which encoded inputs actually differ, clean vs heavy60.** Max abs / relative L2 difference over all batches:

| fixed input | exp 3 | 1b | A |
|---|---|---|---|
| `rgb3d_feats`, `fps_scene_feats` (FPS tokens **and** C-tokens), `fps_cam_ids`, `instr_*`, `query_trajectory` | 0 | 0 | 0 |
| `pcd` (dense scene xyz, CA keys) | 4.96 m / 0.69 | same | same |
| `fps_scene_pos` (FPS xyz + C-token xyz) | 4.59 / 0.44 | 4.01 / 0.30 | 4.55 / 0.34 |
| **`proprio_feats`** | 1.39 / **0.14** | 1.39 / **0.17** | 1.51 / **0.14** |

Reading the code:
- **(b) and (c) are zero by construction.** The CLIP features and `vl_attention` take no positions. FPS is the feature-density sampler (`scene_sampling='fps'`, not `fps_xyz`), so the same tokens are selected. The HistoryFeatureExtractor runs with `patch_rope3d=False`.
- **(e) is not a sanity check: proprioception carries miscalibration.** The encoder's `gripper_context_head` (`encoder3d.encode_proprio`) builds `proprio_feats` from 3 RoPE attention layers. The query is the gripper xyz (true frame); the keys are the current-frame scene tokens, at miscalibrated xyz for cameras 0 and 1. In the head, `proprio_feats` enters only through `time_embs = time_emb(t) + curr_gripper_emb(proprio_feats)`, which is the adaLN signal of every CA, SA, FFW and output-head layer.

**Clean → heavy60** (pos L2 cm / acc < 1 cm / rot L1):
- exp 3: 1.04 / 85.5% / 0.0415 → 1.41 / 78.1% / 0.0488
- 1b: 1.00 / 85.4% / 0.0402 → 1.54 / 78.3% / 0.0618
- A: 1.20 / 79.9% / 0.0482 → 2.05 / 74.7% / 0.0715

**Fraction of the degradation recovered, patching clean into heavy.** Each cell is pos L2 / acc < 1 cm / rot L1; the 95% paired bootstrap CI on pos L2 is in brackets.

| Patch | exp 3 | 1b | A |
|---|---|---|---|
| (a) dense xyz `pcd` | 0.16 / 0.17 / 0.02 | 0.09 / 0.22 / 0.01 | 0.10 / 0.30 / 0.00 |
| (a) FPS xyz | 0.28 / 0.38 / 0.09 | 0.12 / 0.23 / −0.02 | 0.06 / 0.27 / 0.02 |
| (a) C-token xyz | 0.19 / 0.33 / **0.43** | 0.01 / 0.00 / 0.00 | 0.00 / 0.08 / 0.00 |
| **(a) all positions** | **0.51** [0.34, 0.77] / 0.82 / 0.51 | 0.18 [0.11, 0.27] / 0.40 / −0.01 | 0.13 [0.08, 0.20] / 0.27 / 0.02 |
| **proprio_feats** | **0.45** [0.14, 0.64] / 0.08 / 0.52 | **0.79** [0.66, 0.86] / 0.43 / **1.00** | **0.88** [0.81, 0.93] / 0.75 / **1.00** |
| all positions + proprio | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| (d) mechanism replay | −0.00 [−0.01, 0.01] / −0.02 / −0.01 | −0.00 [−0.01, 0.01] / −0.02 / 0.01 | – |
| all positions + mechanism | 0.51 / 0.83 / 0.51 | 0.17 / 0.35 / 0.00 | – |
| (b) scene features, (c) C-token features, language + query | 0 (exact) | 0 | 0 |

**Reverse: heavy into clean, fraction of the degradation caused**, same metric order:

| Patch | exp 3 | 1b | A |
|---|---|---|---|
| all positions | 0.55 / 0.92 / 0.48 | 0.21 / 0.57 / 0.00 | 0.12 / 0.25 / 0.00 |
| proprio_feats | 0.49 / 0.18 / 0.49 | 0.82 / 0.60 / 1.01 | 0.87 / 0.73 / 0.98 |
| mechanism | −0.01 / 0.02 / 0.00 | 0.01 / 0.03 / 0.03 | – |

The individual position pieces in reverse are close to their forward values (`summary_table.txt`).

**Where the damage is.** It is concentrated in a few samples. Only 55 / 67 / 66 samples (exp 3 / 1b / A) get more than 1 cm worse. On those, pos L2 goes 2.3 → 7.6, 1.4 → 7.6 and 2.1 → 13.0 cm. On this subset the pos-L2 recovery is 0.36 (positions) and 0.61 (proprio) for exp 3, 0.11 and 0.87 for 1b, and 0.11 and 0.91 for A.

**Consistency test.** The item asked for (a) + (b) ≈ 100%. Since (b) is identically zero, the real test is (a) + proprio = 100%, which holds exactly by construction (they are all the inputs that differ). The single components are close to additive: 0.51 + 0.45, 0.18 + 0.79 and 0.13 + 0.88.

**Reading:**
- **In 1b and control A, the damage flows mainly through proprioception, not the head's scene geometry.** Patching only `proprio_feats` recovers 79–88% of the position degradation and about 100% of the rotation degradation. All scene-token positions together recover only 13–18%.
  - The path: the miscalibrated external tokens corrupt the encoder's gripper-context summary. That summary is broadcast through `time_embs` into the adaLN of every head layer.
  - This fits TODO 6's undirected error. A corrupted conditioning vector shifts predictions, but not along N_j.
- **exp 3 splits the damage about evenly**: positions 0.51 and proprio 0.45 for pos L2. The C-token positions alone account for 0.43 of exp 3's rotation degradation; they matter for neither 1b nor A. exp 3's head is therefore more sensitive to scene positions (through its RoPE kernel; see TODO 4). Its total degradation is nevertheless the smallest (+0.38 cm vs +0.54 and +0.85), so exp 3 is less damaged on the proprio route, not more robust on the position route.
- **The mechanisms carry none of the damage.** Replaying the clean delta_M (exp 3) or the clean adaLN camera term (1b) into the heavy run recovers 0 ± 0.01. Replaying the heavy values into the clean run causes 0 ± 0.03. Whatever these outputs do, it is not the channel through which miscalibration reaches the actions, and they do not correct it either. This is measured directly at K = 5, so it does not depend on TODO 1's K = 1 collapse finding.
- **Implication.** The cheapest robustness lever these runs point to is the proprioception encoder: it attends to scene tokens at their miscalibrated xyz. Candidates: let `gripper_context_head` attend only to wrist-camera tokens, or drop its RoPE on external tokens. Then re-test 60/60 and IDLT. This is a hypothesis; it is not tested here.
- **Caveats.**
  - The gaps are small in absolute terms (0.4–0.9 cm of mean pos L2), so the fractions have wide CIs (for example exp 3 proprio 0.45 [0.14, 0.64]).
  - The damaged-subset numbers rest on 55–67 samples.
  - Offline, keypose only, one heavy60 draw per sample. The fractions say nothing directly about online success.

---

## TODO 8: redo TODOs 1 and 3 on the non-history models (DeltaM 140k, Base 160k)

**Status:** done

**Why:** the history models reach about 70 at IDLT 60/60, the non-history ones far less. TODO 1 found that in every *history* model the C-tokens reaching the head are collapsed by the history extractor, so delta_M is a per-layer constant. TODO 3 found no attention rerouting. The non-history delta_M model predicts delta_M from the **encoder's** per-camera C-tokens directly, with no extractor. So its delta_M *can* vary per scene and per camera. Does it, and does that explain why it is less robust?

**Models** (raw weights; neither was trained with EMA; K=1 visual history, which is these models' own setting — check the saved config):
- **DeltaM 140k:** `train_logs/PerAct2/peract2_orbital_new_external_only_miscal_deltam_external_200k/interm_step_140000.pth`. Layer-wise `rope_6d` delta_M from the encoder C-tokens, cameras [0,1], no history. IDLT 60/60 = 37.2.
- **Base 160k:** `train_logs/PerAct2/peract2_orbital_new_external_only_miscal_base_200k/interm_step_160000.pth`. No delta_M, no history. IDLT 60/60 = 27.1.

**Do:** the same samples, seeds and conditions as TODOs 1 and 3; reuse their scripts (`scripts/analysis/delta_m_per_layer.py`, `delta_m_trunk_activity.py`, `camera_attention.py` and the plotting scripts).
1. **TODO 1 metrics on DeltaM 140k:**
   - ‖ΔM − I‖ per layer;
   - change between layers;
   - spread across samples, with task η²;
   - clean vs 60/60 per layer;
   - change across flow steps;
   - camera 0 vs 1;
   - C-token spread entering the head and live trunk units.

   The key contrast: are its C-tokens and delta_M **non-collapsed**? Does its delta_M **respond to the miscalibration**? (It might, since the encoder's C-tokens see the point cloud through the encoder's 3D RoPE; check `encoder3d.py` to see whether that is true.)
2. **TODO 3 metrics on DeltaM 140k and Base 160k:**
   - external vs wrist attention share per layer, clean vs 60/60;
   - FPS external-token share;
   - per-token ratio and entropy.
3. **Comparison tables** placing these next to the history models' numbers already in TODOs 1 and 3.

**Results:**

**Setup.** Scripts: `scripts/analysis/todo8.slurm` runs two parts. `PART=dm` (job 3993369) runs `delta_m_per_layer.py` and `delta_m_trunk_activity.py`. `PART=attn` (job 3993406) runs `camera_attention.py`. All three run with a new `--faithful` option that loads through `scripts/analysis/eval_faithful_load.py`, a mirror of `load_model_for_evaluation`. Data: `results/analysis/todo8/{delta_m_per_layer,camera_attention}/`. Stats and figures come from `plot_delta_m_per_layer.py --fig-prefix todo8_dm_layer` and `plot_camera_attention.py --fig-prefix todo8_cam_attn`. Same 853 val samples, seeds and clean/train/heavy60 conditions as TODOs 1 and 3. History models are re-run at their saved K=5.

**Load settings, checked against the saved configs:**
- Both checkpoints are old-schema. They save `num_history: 3` (migrated to visual history 3), `video_deltam: False`, `proprio_num_history: 3`, `scene_sampling` → fps, and `head_positional_encoding` → rope3d.
- Their IDLT plans pass no `visual_num_history`, so online eval ran them at K=1. I use K=1 too.
- K doesn't matter for these models: without the video extractor only the current frame reaches the head. On one batch, K=1 vs K=3 gives the same pos_l2 and attention masses within bf16 noise.
- **DeltaM 140k is a legacy checkpoint** (`dynamic_rope_from_camtoken` + `delta_m_camera_ids` saved). Online eval therefore applies its delta_M to **all four cameras** (`view_align_cameras=None`), and so does this analysis.
- The encoder can't see the point cloud. `encode_siglip2` is SigLIP2 → projection → `vl_attention` (vision-to-language, no RoPE), and the pcd is only interpolated alongside. So the encoder's C-tokens are pure appearance, and the pre-CA delta_M can't respond to miscalibration.

**1. delta_M, DeltaM 140k vs the history models.** Rows are the history models at K=5, numbers identical to TODO 1, plus DeltaM 140k at K=1. Camera 0 unless noted.

‖ΔM − I‖_F per layer, mean (std over samples):

| Model | CA = SA0 | SA1 | SA2 | SA3 |
|---|---|---|---|---|
| **DeltaM 140k**, cam 0 | 0.552 (0.033) | 0.550 (0.081) | 0.551 (0.151) | 0.566 (0.226) |
| DeltaM 140k, cam 1 | 0.546 (0.033) | 0.528 (0.043) | 0.501 (0.056) | 0.481 (0.052) |
| DeltaM 140k, wrist cams 2 / 3 | 0.53 / 0.55 | 0.58 / 0.67 | 0.59 / 0.66 | 0.62 / 0.68 |
| exp 3 200k | 0.701 (0) | 2.662 (0.0007) | 0.288 (0.037) | 0.276 (0) |
| VideoPooled 84k | 0.681 (0) | 0.077 (0.006) | 0.082 (0) | 0.082 (0) |
| warm randhist 50k | 0.258 (0) | 1.162 (0.104) | 0.068 (0.006) | 0.067 (0) |

Across-sample RMS spread of the step-mean delta_M, with task η² in brackets (chance ≈ 0.014):

| Model | CA = SA0 | SA1 | SA2 | SA3 |
|---|---|---|---|---|
| **DeltaM 140k** | **0.077** [0.80] | **0.228** [0.64] | **0.296** [0.64] | **0.398** [0.77] |
| exp 3 200k | 0 [–] | 0.026 [0.52] | 0.060 [0.39] | 0 [–] |
| VideoPooled 84k | 0 [–] | 0.024 [0.85] | 0.0005 [–] | 0 [–] |
| warm randhist 50k | 0 [–] | 0.112 [0.66] | 0.013 [0.09] | 0 [–] |

Response to miscalibration, ‖ΔM_clean − ΔM_60/60‖_F on the same sample with the same noise:

| Model | CA = SA0 | SA1 | SA2 | SA3 |
|---|---|---|---|---|
| **DeltaM 140k**, cam 0 | 0 | 0.013 | 0.020 | 0.018 |
| DeltaM 140k, angle of ΔM_clean·ΔM_heavyᵀ: mean / p99 over samples and steps, cam 0 | 0° | 0.5° / 6.6° | 0.7° / 12° | 0.7° / 11° |
| DeltaM 140k, same angle on wrist cam 3, which is not miscalibrated | 0° | 1.1° / 20° | 1.1° / 24° | 1.3° / 26° |
| exp 3 200k | 0 | 0.009 | 0.010 | 0.0001 |

Change with depth, across flow steps, and between cameras:

| | DeltaM 140k | exp 3 200k |
|---|---|---|
| depth change ‖log(ΔM_l ΔM_{l−1}ᵀ)‖ (max angle), SA0→1 / 1→2 / 2→3 | 0.17 (6.3°) / 0.11 (3.9°) / 0.15 (5.5°) | 2.52 (91°) / 2.95 (106°) / 0.02 (0.7°) |
| RMS change across the 5 flow steps, SA1 / SA2 / SA3 | 0.011 / 0.015 / 0.020 | 0.009 / 0.002 / 0 |
| cam 0 vs cam 1, ‖log(ΔM_0 ΔM_1ᵀ)‖ (max angle), CA / SA1 / SA2 / SA3 | 0.063 (2.3°) / 0.16 (6.1°) / 0.19 (7.1°) / 0.18 (6.7°) | 0 / 0.009 / 0.010 / 0 |

**C-tokens entering the head, and the delta_M trunk** (`trunk_activity.json`, 5 clean batches; per-channel std over samples of the external C-tokens fed to `camera_proj`):

| Model | C-token spread at CA | live trunk units (of 120), CA / SA1 / SA2 / SA3 | ‖ΔM − exp(skew(bias))‖, CA / SA1 / SA2 / SA3 |
|---|---|---|---|
| **DeltaM 140k** (encoder C-tokens, no extractor) | **0.037** | 7.6 / 5.0 / 4.9 / 4.0 | 0.20 / 0.37 / 0.52 / 0.52 |
| exp 3 200k, K=5 | 0.00015 (extractor input 0.025) | 1.0 / 2.0 / 0.6 / 0 | 0.56 / 2.63 / 0.05 / 0 |
| exp 3 110k, K=5 | 0.00001 (input 0.048) | 1.0 / 2.0 / 0.8 / 0 | 0.56 / 2.20 / 0.05 / 0 |
| VideoPooled 84k, K=5 | 0.00001 (input 0.039) | 1.0 / 1.2 / 0.4 / 0 | 0.61 / 0.05 / 0.003 / 0 |
| warm randhist 50k, K=5 | 0.00005 (input 0.035) | 1.0 / 1.0 / 0.6 / 0 | 0.23 / 1.35 / 0.05 / 0 |

So the C-token collapse in the history extractor is real at K=5, not a K=1 artifact. The extractor reduces a 0.025–0.048 input spread to ≤1.5e-4.

**2. Attention.** External share = attention on cam 0+1 scene tokens / attention on all scene tokens. Values are clean → heavy60, Δ in pp ± SE over 853 paired samples. History models at K=5 are in TODO 3.

| Layer | **DeltaM 140k** | **Base 160k** | exp 3 (K=5) Δ | 1b Δ | control A Δ | ema20k Δ |
|---|---|---|---|---|---|---|
| ca0 | 48.3 → 48.6, +0.34 ±0.07 | 48.8 → 49.8, +1.02 ±0.11 | +0.32 | +0.07 | +0.73 | +0.50 |
| ca1 | 46.6 → 49.1, **+2.51** ±0.12 | 43.6 → 46.0, **+2.33** ±0.14 | +0.58 | +0.11 | +1.74 | +0.64 |
| sa0 | 32.8 → 33.6, +0.76 ±0.06 | 34.6 → 34.5, −0.08 ±0.04 | +0.25 | +0.33 | +0.09 | +0.04 |
| sa1 | 32.7 → 33.4, +0.71 ±0.07 | 35.8 → 36.0, +0.23 ±0.04 | −0.01 | +0.01 | +0.07 | +0.03 |
| sa2 | 32.1 → 32.4, +0.34 ±0.05 | 35.0 → 35.5, +0.52 ±0.07 | −0.44 | +0.23 | −0.08 | −1.14 |
| sa3 | 33.4 → 35.7, **+2.31** ±0.16 | 34.0 → 35.0, +0.95 ±0.11 | −0.47 | −0.52 | −0.31 | −1.01 |
| pos0 | 32.4 → 34.6, **+2.18** ±0.15 | 32.1 → 30.3, −1.83 ±0.16 | −1.25 | −1.06 | −1.11 | −1.46 |
| pos1 | 32.2 → 33.3, +1.13 ±0.14 | 31.1 → 27.8, **−3.21** ±0.16 | −1.73 | −1.14 | −1.24 | −1.76 |
| rot0 | 31.6 → 31.7, +0.07 ±0.02 | 33.4 → 33.5, +0.13 ±0.05 | −0.01 | +0.05 | +0.04 | −0.07 |
| rot1 | 31.7 → 31.7, +0.07 ±0.05 | 32.8 → 33.2, +0.37 ±0.07 | +0.05 | −0.00 | −0.16 | +0.02 |

Other attention measurements:
- **FPS token share** (204 tokens, cam 0 / 1 / 2 / 3), identical clean vs heavy60:
  - DeltaM 140k: 30.9 / 30.4 / 72.2 / 70.5, so **30.0% external**.
  - Base 160k: 36.0 / 29.8 / 69.1 / 69.0, so **32.3% external**.
  - History models keep only 10–19% external, and 1b has no external FPS token at all in 4.8% of samples.
- **Per-token ratio** (external share ÷ token share), clean → heavy60:
  - DeltaM 140k: 1.07–1.11 in SA and the heads, rising under heavy60 (sa3 1.11 → 1.19, pos1 1.07 → 1.11).
  - Base 160k: 1.05–1.11 in SA; pos1 0.96 → 0.86.
  - History models: pos1 0.43–0.71, falling under heavy60.
- **Entropy:** clean values are similar across all models (CA ≈ 6.5, SA ≈ 5.2 nats). The non-history models focus more under heavy60:
  - DeltaM 140k: −0.11 (ca1), −0.12 (sa3), −0.09 (pos0/pos1).
  - Base 160k: −0.10 (ca1), −0.07 (sa3), −0.09 (pos0).
  - History models: at most −0.07 anywhere.
- **Per head:** as in TODO 3, the shift sits in the high-frequency heads 0, 2 and 4. DeltaM's largest moves are +8.5 pp (ca1, head 2), +11.2 pp (sa3, head 2) and +9.8 pp (pos0, head 0). Base's are −8.1 pp (pos0, head 4) and −5.7 pp (pos1, head 4). The CA increase grows over the flow steps (DeltaM ca1 +1.9 → +3.1 pp; Base +1.6 → +3.1 pp).
- **Offline pos_l2 clean → heavy60 (cm):**

  | Model | clean → heavy60 | ratio |
  |---|---|---|
  | DeltaM 140k | 1.32 → 2.68 | ×2.0 |
  | Base 160k | 1.49 → 3.33 | ×2.2 |
  | exp 3 | 1.03 → 1.40 | ×1.4 |
  | 1b | 0.99 → 1.53 | ×1.5 |
  | ema20k | 1.06 → 1.57 | ×1.5 |
  | control A | 1.19 → 2.03 | ×1.7 |

  This ordering matches IDLT 60/60.

Figures (`figures/README.md`):
- `todo8_dm_layer_*.{png,svg}`: the six TODO 1 metrics with DeltaM 140k added. History models are at K=5, and VideoPooled's wrist cameras now carry delta_M under the legacy rule.
- `todo8_cam_attn_external_share.{png,svg}` and `todo8_cam_attn_sa_mass_breakdown.{png,svg}`: the six models overlaid.

**Reading:**
- **DeltaM 140k's delta_M is not collapsed.** Its C-tokens come straight from the encoder and vary across samples (0.037 per-channel std, against ≤1.5e-4 after the history extractor). So its delta_M varies per sample from the first layer (RMS 0.08 at CA, up to 0.40 at SA3, 64–80% explained by task) and per camera (2–7° between cameras 0 and 1). It drifts gently with depth (4–6° per layer) instead of jumping between a few fixed rotations, and it keeps about 5–8 live trunk units against 1–2 in the history models. Its size, ‖ΔM − I‖ ≈ 0.55, is similar on all four cameras, including the wrist cameras that the legacy rule also rotates.
- **It barely responds to the miscalibration.** The pre-CA delta_M is exactly unchanged under 60/60, because the encoder's C-tokens are appearance-only (no point cloud, no RoPE). In SA it changes by 0.5–0.7° on average against 60° injected, with a p99 of 7–12°. The non-miscalibrated wrist camera 3 changes *more* (1.1–1.3° mean, p99 20–26°). So these changes come from SA mixing, not from estimating camera 0/1 error. A delta_M that varies per scene is not a delta_M that corrects geometry.
- **This doesn't explain the robustness gap; if anything it points the other way.** The less robust model has the richer, per-sample delta_M. The robust history models have a per-layer constant one. Delta_M's per-scene variation is mostly task identity, and robustness doesn't track it.
- **The clearest structural difference is which tokens the head sees.** The non-history models keep 30–32% external FPS tokens against 10–19% in the history models. They spend about a third of SA and output-head attention on the external cameras, against 11–30%. Under 60/60 they shift attention *more* (CA +2.3–2.5 pp, against ≤1.7). DeltaM even pushes more attention onto the miscalibrated cameras in SA3 and the position head (+2.2–2.3 pp). Base pulls attention off them in the position head (−1.8/−3.2 pp) and is still the least robust. So, as in TODO 3, the direction of the attention shift doesn't track robustness. What separates the groups is how exposed the trunk is to external tokens in the first place. The history models' feature-density sampler may help simply by sampling few external tokens.
- **Caveats.**
  - The two groups differ in more than history: training recipe, scratch vs warm start, EMA, miscalibration curriculum, and when they were trained. These comparisons are correlations.
  - The FPS-share explanation is a hypothesis. A direct test is to evaluate a history model with `scene_sampling: image_space`, which forces 50% external tokens, or a non-history model with external FPS tokens capped. Activation patching (TODO 7) would also test it.
  - Attention mass is not influence (see TODO 3).
- **Loader bug found on the way (fixed centrally in 5ef5cf3).** `visual_num_history` is an eval-runtime key, so the offline `load()` had run every history model at K=1. Legacy delta_M checkpoints (DeltaM 140k, VideoPooled) also lost their all-camera rule. At K=5, TODO 1's delta_M numbers and the extractor collapse are unchanged, and TODO 3 changes by ≤1.4 pp. The earlier "suspect" offline errors were this bug: ema20k drops 2.95 → 1.06 cm, and VideoPooled reads 1.19 cm, not 7 cm.

---

## TODO 9: activation patching on the non-history models (Base 160k, DeltaM 140k)

**Status:** done

**Why:** every history model's extractor is dead within 10k–30k steps:

| Model | Extractor dead by |
|---|---|
| VideoPooled stage 1 (from scratch, ordered history) | 20k |
| A, C, 1b | 10k |
| exp 3 | 30k |

(CPU check: each checkpoint's `HistoryFeatureExtractor` fed random 5-frame inputs; output spread across samples and cameras is 0.) Yet history models reach about 60–72 at IDLT 60/60 and non-history models 27–37. The gap can't come from history content. Candidates:
- **The C-tokens:** non-history models have per-camera appearance tokens placed at each camera's **miscalibrated** mean xyz; history models have one learned constant.
- **The FPS sampler's external share:** 30% vs 10–19%.
- **The proprio encoder.**

TODO 7 measured where the damage enters for exp 3, 1b and A; this does the same for the models that are *not* robust.

**Do:** run TODO 7's activation patching (`scripts/analysis/activation_patching.py`), clean into heavy60 and the reverse, on **Base 160k** (`train_logs/PerAct2/peract2_orbital_new_external_only_miscal_base_200k/interm_step_160000.pth`) and **DeltaM 140k** (`train_logs/PerAct2/peract2_orbital_new_external_only_miscal_deltam_external_200k/interm_step_140000.pth`). Raw weights, with the load settings TODO 8 used (`--faithful`, legacy all-camera rule for DeltaM). Same samples and seeds. Patch separately:
1. **C-token positions only** (`camera_summary_pos`, i.e. the last ncam entries of `fps_scene_pos`);
2. **FPS scene-token positions only;**
3. **dense scene-token positions** (cross-attention keys);
4. **proprio features;**
5. **DeltaM's delta_M**, replayed per call;
6. useful combinations.

Check that positions plus proprio recover 100%. Report each component's share of the damage next to TODO 7's numbers for exp 3, 1b and A. The key question: does C-token position carry a large share in Base/DeltaM but not in the history models?

**Results:**

Scripts: `scripts/analysis/activation_patching.py` with a new `--faithful` option, which loads via TODO 8's `eval_faithful_load.py` (spec `label=ckpt:raw|ema:K|ckpt`), and three new combinations (below). Launcher: `scripts/analysis/activation_patching_todo9.slurm` (job 3994003). Plotting: `scripts/analysis/plot_activation_patching.py --in-dir results/analysis/todo9/activation_patching --fig-stem todo9_patch_fraction_recovered`. Raw per-sample metrics: `results/analysis/todo9/activation_patching/<model>.pt`. Aggregates: `summary.json` and `summary_table.txt`. Figure: `figures/todo9_patch_fraction_recovered.png`.

Setup: the same 853 val samples, seeds and protocol as TODO 7: shared history draw, one flow noise per batch, heavy60 on cameras [0, 1].
- **DeltaM 140k and Base 160k:** raw weights, K=1. DeltaM 140k uses the legacy all-camera delta_M (`[load] ... legacy_all_cams=True view_align_cameras=None`).
- **exp 3, 1b and A:** re-run in the same job at their saved K=5. They reproduce TODO 7's fractions exactly, so the five models sit on one footing.
- **Exactness:** patch-nothing = heavy and patch-every-input = clean, max |Δ| = **0.0** for all five models.
- **Inputs that differ between clean and heavy60:** in all five models only the dense xyz, the FPS and C-token xyz, and `proprio_feats` differ. Scene, C-token and language features are bit-identical. The relative change in `proprio_feats` is 0.14 for DeltaM, 0.19 for Base and 0.07–0.09 for the history models at K=5 on the smoke batches.

New combinations:
- `pos_dense+pos_fps`: scene-token positions without the C-tokens.
- `pos_ctok+proprio`.
- `pos_ctok+mech`.

**Clean → heavy60 (pos L2 cm / acc < 1 cm / rot L1).** The last column counts samples that get more than 1 cm worse.

| Model | clean | heavy60 | samples >1 cm worse |
|---|---|---|---|
| DeltaM 140k | 1.32 / 67.5% / 0.0465 | 2.69 / 50.8% / 0.0641 | **244** |
| Base 160k | 1.49 / 65.6% / 0.0482 | 3.30 / 48.2% / 0.0865 | **264** |
| exp 3 200k | 1.04 / 85.5% / 0.0415 | 1.41 / 78.1% / 0.0488 | 55 |
| 1b 200k | 1.00 / 85.4% / 0.0402 | 1.54 / 78.3% / 0.0618 | 67 |
| A 200k | 1.20 / 79.9% / 0.0482 | 2.05 / 74.7% / 0.0715 | 66 |

**Share of the position-L2 degradation, clean into heavy (fraction recovered).** The 95% paired bootstrap CI is in brackets.

| Patch | **DeltaM 140k** | **Base 160k** | exp 3 | 1b | A |
|---|---|---|---|---|---|
| C-token xyz | **0.00** [0.00, 0.00] | **−0.00** [0.00, 0.00] | 0.19 [0.12, 0.33] | 0.01 | 0.00 |
| FPS scene xyz | 0.26 [0.21, 0.31] | 0.20 [0.16, 0.25] | 0.28 | 0.12 | 0.06 |
| dense xyz (CA keys) | 0.27 [0.21, 0.34] | 0.17 [0.13, 0.22] | 0.16 | 0.09 | 0.10 |
| dense + FPS | 0.32 | 0.23 | 0.38 | 0.17 | 0.13 |
| all positions | 0.32 [0.24, 0.41] | 0.23 [0.17, 0.29] | 0.51 | 0.18 | 0.13 |
| proprio_feats | 0.17 [0.04, 0.28] | 0.49 [0.40, 0.57] | 0.45 | 0.79 | 0.88 |
| C-token xyz + proprio | 0.17 | 0.49 | 0.63 | 0.80 | 0.89 |
| all positions + proprio | **1.00** | **1.00** | 1.00 | 1.00 | 1.00 |
| delta_M replay | 0.02 [0.00, 0.04] | – | −0.00 | −0.00 (adaLN) | – |
| all positions + delta_M | 0.34 | – | 0.51 | 0.17 | – |

**Reverse, heavy into clean (fraction caused), pos L2:**

| Patch | DeltaM 140k | Base 160k | exp 3 | 1b | A |
|---|---|---|---|---|---|
| C-token xyz | −0.00 | −0.00 | 0.13 | 0.00 | 0.00 |
| FPS scene xyz | 0.32 | 0.19 | 0.20 | 0.11 | 0.03 |
| dense xyz | 0.34 | 0.22 | 0.11 | 0.06 | 0.05 |
| all positions | **0.83** [0.73, 0.96] | **0.51** [0.43, 0.61] | 0.55 | 0.21 | 0.12 |
| proprio_feats | 0.68 [0.60, 0.76] | 0.77 [0.70, 0.82] | 0.49 | 0.82 | 0.87 |
| delta_M | 0.01 | – | −0.01 | 0.01 | – |

**Rotation L1:**
- Proprio recovers 0.95 (DeltaM) and 0.97 (Base) of the rotation damage, and causes 1.01 and 0.97 of it.
- Positions recover ≤ 0.03.
- C-token xyz recovers 0.00 for both. exp 3 is the only model where it matters for rotation (0.43).

**Accuracy within 1 cm** behaves oddly in DeltaM: patching only clean proprio into heavy *lowers* it further (−0.36 of the gap), and heavy positions alone cause 1.36 of it. The paths interact strongly here (next section).

**Additivity.**
- In the history models, the forward shares of positions and proprio add up to about 1: 0.51+0.45 for exp 3, 0.18+0.79 for 1b, 0.13+0.88 for A.
- In the non-history models they don't. Forward: 0.32+0.17 = 0.49 for DeltaM and 0.23+0.49 = 0.72 for Base. Reverse: 0.83+0.68 = 1.51 and 0.51+0.77 = 1.28.
- So each path alone causes most of the damage, and fixing either alone leaves most of it. Damage happens when *either* the head's scene geometry *or* the proprio summary is corrupted. Only the joint patch gives 100%.

**Reading:**
- **No: C-token positions carry none of the damage in the non-history models.** The share is 0.00 [0.00, 0.00] in both directions for DeltaM 140k and Base 160k. Among the history models, C-token positions matter only for exp 3 (0.19 of pos L2, 0.43 of rot), whose delta_M re-mixes the C-tokens' RoPE; they matter for neither 1b nor A. TODO 8 found external C-tokens draw only about 1% of SA attention in every model, consistent with this. The hypothesis that per-camera C-tokens at miscalibrated mean xyz explain the history/non-history gap is ruled out.
- **What differs is how much of the damage comes through scene-token geometry, and that the paths are redundant.**
  - Non-history models are damaged far more often: 244–264 samples more than 1 cm worse, against 55–67.
  - Their heavy60 positions alone cause 0.51–0.83 of the position damage, against 0.12–0.21 for 1b and A. The dense and FPS tokens contribute about equally.
  - The FPS share fits TODO 8's 30–32% external FPS tokens (against 10–19%). The dense CA-key share (0.17–0.27 recovered, 0.22–0.34 caused, against 0.05–0.16) also implicates the cross-attention keys, which every model sees at the same 50% external share. So the non-history heads are more geometry-sensitive per token as well, not only more exposed.
- **Proprio is a shared failure route.** It carries about 100% of the rotation damage in all four non-exp-3 models, 1b and A as well as DeltaM and Base. In Base it is also about half the position damage. The `gripper_context_head` point from TODO 7 applies to the non-robust models too.
- **DeltaM's delta_M carries nothing:** 0.02 recovered, 0.01 caused. This matches TODO 8, where its delta_M varies per sample but barely responds to 60/60.
- **Caveats.**
  - The two groups differ in many training factors (TODO 8 caveat), so this localises the damage; it does not identify a cause.
  - The non-additivity means single-component shares for DeltaM and Base depend on direction. Read the forward and reverse columns together.
  - Offline, keypose only, one heavy60 draw per sample.

---

## TODO 10: what the C-tokens do in the non-history models (Base 160k, DeltaM 140k), with 1b as contrast

**Status:** done

**Why:** in Base and DeltaM, each C-token is camera j's real mean appearance feature, placed at camera j's mean point-cloud xyz (`camera_summary_pos`), which is miscalibrated for cameras 0 and 1 under heavy miscalibration. In the history models it is one learned constant, copied to all four cameras, placed at the same kind of miscalibrated position. We want to see how these tokens behave and whether they are a route for miscalibration into the head.

**Models:**
- Base 160k and DeltaM 140k: raw weights, TODO 8's load settings.
- 1b 200k (EMA) as the collapsed-C-token contrast. Add exp 3 200k if cheap.

Same validation samples, seeds and conditions as before (clean, train, heavy60).

**Measure** per camera (0 and 1 external, 2 and 3 wrist), at head input and after each SA layer, plus the output-head SA layers:
1. **Content at head input:**
   - spread across samples;
   - distance between cameras (all pairs);
   - share of variance explained by task, and by camera group.
2. **Position shift:** ‖camera_summary_pos_heavy − camera_summary_pos_clean‖ in cm, per camera. External cameras should move; wrist cameras should not.
3. **How each C-token changes under heavy, layer by layer:** ‖C_heavy − C_clean‖ / ‖C_clean‖ on the same sample. Compare with the same relative change for FPS scene tokens and trajectory tokens, so we see whether the C-tokens change more or less than the rest.
4. **Attention involving the C-tokens, clean vs heavy:**
   - how much the trajectory queries attend to each C-token, per layer;
   - what each C-token attends to (share on external vs wrist FPS tokens, other C-tokens, registers).
5. **How much the head relies on the C-tokens** (attention × value norm, or a quick zero-ablation of the C-token slots in the clean condition, with offline position error).

One summary table comparing Base, DeltaM and 1b, plus a figure of item 3 per layer.

**Results:**

**Setup.** Script `scripts/analysis/ctoken_behaviour.py` (job 3994034 via `scripts/analysis/ctoken_behaviour.slurm`); summary and figure from `scripts/analysis/plot_ctoken_behaviour.py`. Data in `results/analysis/ctoken_behaviour/`: `<model>.pt`, `summary.json`, `summary_table.txt`.
- Models and loading: Base 160k and DeltaM 140k raw at K=1 with TODO 8's `--faithful` settings (legacy all-camera delta_M for DeltaM); 1b 200k and exp 3 200k EMA at their saved K=5.
- Samples: the same 853 val samples, with seeds per batch as in TODOs 1/3/8. Offline pos_l2 reproduces TODO 8 to ≤0.03 cm.
- Each batch runs clean, train, heavy60 and the ablations back-to-back on the same sample and the same flow noise.
- The FPS selection is asserted identical across conditions, so relative changes are per token.
- Attention is recomputed in fp32 as in TODO 3, for the trajectory rows and the four C-token rows of every SA and output-head layer.
- Two ablations: `mask` removes the C-token slots as attention keys in every SA and output-head layer; `mean` replaces the C-token content at head input with the dataset mean per camera.
  - I checked that masking cuts every route. With the mask on, also zeroing the C-tokens changes the prediction by exactly 0.0 (Base, one batch).
  - The C-tokens can still act outside attention: through delta_M in DeltaM and exp 3, and through the adaLN signal in 1b. `mean` tests those routes.

**Summary table.** "ext" = cameras 0/1, "wr" = cameras 2/3. Relative change = ‖h_heavy60 − h_clean‖/‖h_clean‖ on the same sample, averaged over samples and the 5 flow steps.

| | **Base 160k** | **DeltaM 140k** | 1b 200k (collapsed) | exp 3 200k (collapsed) |
|---|---|---|---|---|
| **1. Content at head input** | | | | |
| spread across samples (per-channel std), ext / wr | 0.104, 0.106 / 0.144, 0.135 | 0.109, 0.112 / 0.156, 0.154 | ≤1e-4 | ≤2e-4 |
| token norm | 1.7–2.0 | 1.8–2.4 | 11.3 | 9.6 |
| distance between cameras, 0–1 / 0–2 / 2–3 | 0.88 / 2.95 / 1.65 | 0.83 / 2.84 / 1.73 | 0.000 / 0.002 / 0.002 | 0.003 / 0.002 / 0.003 |
| η² task, ext / wr | 0.82, 0.89 / 0.46, 0.46 | 0.76, 0.75 / 0.51, 0.58 | ≈0 (no spread) | ≈0 (no spread) |
| η² camera group (G1–G6), ext / wr | 0.13, 0.16 / 0.04, 0.04 | 0.13, 0.14 / 0.06, 0.06 | ≈0 | ≈0 |
| clean vs heavy60 content at head input | identical (max diff 0) | identical | identical | identical |
| after sa3: spread ext; η² task ext; distance cam 0–1 | 0.35; 0.74–0.82; 2.1 | 0.44; 0.77–0.78; 3.3 | 0.18; 0.48; 0.38 (copies stay alike) | 0.28; 0.46–0.47; 0.81 |
| **2. C-token position shift** (cm), cams 0 / 1 / 2 / 3 | heavy60: 98.1 / 93.7 / 0 / 0; train: 13.6 / 12.8 / 0 / 0 (same data for all models) | same | same | same |
| **3. Relative change under heavy60** | | | | |
| at sa3: C ext / C wr / FPS ext / FPS wr / traj | 0.057 / 0.059 / 0.109 / 0.054 / 0.087 | 0.054 / 0.064 / 0.080 / 0.051 / 0.076 | **0.072** / 0.031 / 0.128 / 0.036 / 0.035 | **0.123** / 0.035 / 0.096 / 0.063 / 0.038 |
| at pos1 (position head output): C ext / C wr / FPS ext / FPS wr / traj | 0.117 / 0.062 / 0.146 / 0.063 / 0.048 | 0.154 / 0.085 / 0.201 / 0.079 / 0.066 | **0.31** / 0.033 / 0.311 / 0.039 / 0.027 | **0.25** / 0.036 / 0.257 / 0.066 / 0.034 |
| **4. Attention involving C-tokens** (clean; heavy60 moves every entry ≤0.2 pp unless noted) | | | | |
| traj → all 4 C-tokens, per layer (SA / pos / rot) | 1.6–2.0% / 2.3–2.7% / 2.0% | 1.7–2.1% / 2.1–2.2% / 1.9% | 2.7–4.4% / 1.8–4.5% / 2.1–3.9% | 2.6–4.4% / 1.2–2.8% / 2.1–2.2%; sa2 rises 2.7 → 4.0% under heavy60 |
| C0/C1 rows → ext FPS / wr FPS | 0.31–0.34 / 0.61–0.64 | 0.27–0.31 / 0.64–0.68 | 0.05–0.12 / 0.82–0.92 | 0.14–0.25 / 0.68–0.81 |
| C2/C3 rows → ext FPS | 0.25–0.31 | 0.27–0.31 | 0.05–0.12 | 0.16–0.25 |
| external share of FPS tokens (for reference) | 32% | 30% | 10% | 19% |
| C rows → other C / registers / traj | 1.3–2.3% / 1.1–1.9% / ≈1% | 1.3–1.9% / 1.5–1.9% / ≈1% | 0.9–6.6% / 1.5–2.5% / ≈1% | 1.4–2.7% / 1.3–2.2% / ≈1% |
| **5. Ablations, offline pos_l2** (cm) | | | | |
| clean | 1.49 | 1.32 | 1.00 | 1.04 |
| clean, mask all C / mask external C | 1.50 / 1.49 | 1.33 / 1.32 | 1.04 / 1.02 | 1.07 / 1.04 |
| clean, mean C / zero C | 1.68 / 1.83 | **2.06 / 6.12** | 1.00 / 2.48 | 1.04 / 1.07 |
| heavy60 | 3.30 | 2.69 | 1.54 | 1.41 |
| heavy60, mask all C / mask external C | 3.32 / 3.30 | 2.70 / 2.69 | 1.57 / 1.54 | 1.36 / **1.34** |
| heavy60, mean C | 3.42 | 3.26 | 1.53 | 1.41 |

Figure: `figures/ctoken_rel_change.{png,svg}`. Item 3 per layer, one panel per model, for the C-tokens per camera, external and wrist FPS tokens, and trajectory tokens.

**Reading:**
- **In Base and DeltaM the C-tokens carry real content; in the history models they don't.** Base and DeltaM have per-camera appearance tokens: external C-tokens are 75–89% explained by task and 13–16% by camera group, and external differs strongly from wrist (distance ≈ 3). In 1b and exp 3 the four C-tokens are one constant at head input. Self-attention makes them sample-specific (about half explained by task), but in 1b the four copies stay nearly identical even after sa3.
- **The miscalibration enters the C-tokens only through their position.** Content at head input is bit-identical clean vs heavy60. The external C-tokens' positions jump about 1 m (98 / 94 cm) under 60°/60 cm; train-style noise moves them about 13 cm; wrist positions don't move.
- **Base and DeltaM's C-tokens barely react to that 1 m jump; the robust models' react more.**
  - Base/DeltaM: in self-attention the external C-tokens change no more than the wrist ones (≈0.05–0.06 relative at sa3), and less than the external FPS tokens. Their content dominates their position.
  - 1b and exp 3: the external C-tokens change 2–4× more than the wrist C-tokens, and as much as the external FPS tokens, peaking at 0.25–0.33 in the position head. With no content to tell them apart, position is the only thing that distinguishes the external copies.
- **Yet the C-tokens are not a route for the damage.**
  - Trajectory queries put only about 2% of their attention on all four C-tokens in Base and DeltaM, and 2–4.5% in the history models. Heavy60 doesn't change that.
  - Removing the C-tokens as keys everywhere changes heavy60 error by ≤0.02 cm in Base/DeltaM, out of 1.8 and 1.4 cm of damage.
  - The one exception is exp 3: masking the external C-tokens cuts its (small) heavy60 damage from 0.37 to 0.30 cm, about a fifth.
- **The C-tokens act as generic pooling tokens, not camera-specific summaries.** Each C-token row attends to external vs wrist FPS tokens roughly in proportion to how many of each the sampler kept (Base 31–34% vs 32% of tokens; 1b 5–12% vs 10%). C0 reads its own camera no more than C2 does, and none of this moves under heavy60.
- **Reliance on content.**
  - DeltaM needs its C-token content, through delta_M: mean-ablation costs +0.74 cm clean and +0.57 cm at heavy60; zero-ablation costs +4.8 cm, which is out of distribution.
  - Base is mildly sensitive to *wrong* content (+0.19 cm for the mean), but not to removing it (+0.01 cm). A mismatched C-token perturbs the tokens that read it; an absent one is renormalised away.
  - 1b only reacts to zeroing (+1.5 cm, via the adaLN signal); exp 3 doesn't react at all.
- **What this rules out.** The non-history models' weakness at 60/60 does not come from their per-camera C-tokens or from those tokens' miscalibrated positions. That leaves the scene-token positions, i.e. the FPS and dense keys, and the larger external FPS share (TODO 8) as the main candidates. TODO 9's patching is the direct test.
- **Caveats.**
  - Masking removes C-tokens as attention keys only. In DeltaM and exp 3 their content still drives delta_M; in 1b it still drives the adaLN signal.
  - "Relative change" mixes direct position effects with downstream mixing.
  - The η² values for 1b and exp 3 at head input are over an essentially zero spread and mean nothing.
  - Ablation effects ≤0.03 cm are at noise level (offline pos_l2 matches TODO 8 to 0.03 cm).

---

## TODO 11: TODO 5's probes on the full-patch history models, including inside the extractor

**Status:** done

**Why:** a CPU check (random 5-frame patch grids and random per-sample xyz through each checkpoint's real extractor) shows every full-patch extractor outputs a constant too:
- FullPatch (`fullpatch_finetune_300k_a6000_b64`, the "best" in the results tables): constant from 10k (warm-started); it loads with legacy compat.
- FullPatch depth 8: constant from 10k.
- **Refine FullPatch rope3d**, whose patches carry miscalibrated xyz via 3D RoPE: spread 0.0003 at 10k, constant from 50k.
- **Direct FullPatch rope3d:** delta_M identical for every sample; ‖ΔM − I‖ grows only 0.06 → 0.12 over training.

The full-patch path keeps every patch and, in the rope3d variants, sees geometry, which is the information delta_M would need. We want to know whether that information exists **inside** the extractor before it collapses, and whether anything downstream keeps it.

**Models:**
- **Refine FullPatch rope3d** (best) and **Direct FullPatch rope3d** (best): these see geometry.
- **FullPatch a6000_b64** (best): no geometry, the reference. Add depth 8 if cheap.
- **Random-init** full-patch rope3d: the floor.

Use each run's saved config (K = 5, `video_deltam_full_image`, `patch_rope3d`, role), the fixed loader (check the `[load]` line), and legacy compat where `load_model_for_evaluation` would apply it.

**Features**, per external camera:
- encoder per-image tokens (mean over patches) and the C-tokens entering the extractor;
- **inside the extractor:** the per-image token, and the patch-mean, after each same-time and temporal block (hook `_forward_patch_rope3d` / `_forward_legacy`);
- the extractor output;
- delta_M (Direct);
- the head's C-tokens after each SA layer (Refine).

**Targets:** exactly TODO 5's: (A) true camera pose, including the leave-one-group-out probe; (B) the injected miscalibration, using TODO 5's probe-only 0–60° / 0–60 cm sweep.

**Key questions:**
1. In the rope3d variants, is the miscalibration decodable at any depth inside the extractor? It sees patch xyz, so cross-camera geometric inconsistency is in principle visible there.
2. If yes, at which block is it lost?
3. Does the random-init rope3d extractor carry it (an architecture ceiling vs a learned loss)?

Reuse `scripts/analysis/probe_features.py` and `fit_probes.py` (extend them with options; keep TODO 5's defaults unchanged). Save features under `results/analysis/probes_fullpatch/`.

**Results:**

*Setup.*
- **Protocol and samples are TODO 5's:** 3000 training rows to fit and all 853 val rows to test, the same seeds, the same clean / train-style / 0–60° · 0–60 cm sweep conditions on cameras [0, 1], ridge/logistic with CV grouped by episode, and mean over cameras 0 and 1.
- **Checkpoints:** `best.pth`, raw weights, for:
  - Refine FullPatch rope3d (160k)
  - Direct FullPatch rope3d (200k)
  - FullPatch a6000_b64 (244k)
  - FullPatch d8 (276k)
  - random-init Refine rope3d (seed 0; frozen SigLIP towers as always)
- **Loading.** Every `[load]` line reads `visual_num_history=5`. `[arch]` lines confirm `full_image=True`, `patch_rope3d` True/True/False/False, roles refine/predict_delta_m/refine/refine, depths 4/4/4/8, and consecutive history. a6000_b64 loads with both legacy rules from `load_model_for_evaluation`: delta_M on all cameras, and the pre-readout full-image forward (`[checkpoint compat] legacy full-image extractor forward`).
- **Features are stored in fp32**, because the extractor's later blocks vary by about 1e-4 of their norm. For the current frame, per camera:
  - the patch mean after every same-time (ST) and temporal (TP) block;
  - the per-image token after every TP block;
  - the extractor output (= image token after the last TP);
  - the head's C-tokens after SA1–SA4;
  - Direct's delta_M.
- **Two scale columns.** *Spread* is across-sample std / mean norm (clean val). *Sweep shift* is mean ‖clean − sweep‖ / norm on the same sample.

Selected rows (full table: `results/analysis/probes_fullpatch/summary_table.md`):

| Model | Stage | Spread | Sweep shift | Group acc | LOGO pos (cm) | Rotation-vector R² | Translation-vector R² | Rotation-magnitude R² | Clean vs sweep acc |
|---|---|---|---|---|---|---|---|---|---|
| Refine rope3d | enc | 0.61 | 0 | 0.95 | 21 | 0.00 | 0.00 | 0.00 | 0.50 |
| Refine rope3d | TP1 patch mean | 0.24 | 1e-2 | 0.63 | 35 | 0.03 | 0.00 | 0.06 | 0.58 |
| Refine rope3d | TP3 patch mean | 2.3e-4 | 2.0e-4 | 0.92 | 27 | **0.30** | 0.00 | **0.47** | **0.81** |
| Refine rope3d | TP4 patch mean | 2.6e-4 | 1.4e-3 | 0.71 | 63 | 0.06 | 0.00 | 0.13 | 0.80 |
| Refine rope3d | extractor out (TP4 image token) | 2.6e-4 | 2.2e-3 | 0.27 | 47 | 0.05 | 0.00 | 0.09 | 0.60 |
| Refine rope3d | SA4 | 0.30 | 6.5e-2 | 0.77 | 30 | 0.22 | 0.19 | 0.12 | 0.62 |
| Direct rope3d | TP1 image token | 0.20 | 1.3e-2 | 0.83 | 29 | 0.19 | 0.02 | 0.14 | 0.61 |
| Direct rope3d | ST3–ST4 patch mean | 4e-4–1e-3 | 3e-4 | 0.24–0.32 | 50 | 0.00 | 0.00 | 0.00 | 0.50 |
| Direct rope3d | TP4 patch mean | 4.0e-4 | 2.9e-4 | 0.95 | 21 | **0.41** | 0.00 | **0.60** | **0.82** |
| Direct rope3d | extractor out | 1.0e-3 | 1.0e-3 | 0.70 | 36 | 0.17 | 0.00 | 0.15 | 0.63 |
| Direct rope3d | **delta_M** | **0** | **0** | 0.20 | 50 | 0.00 | 0.00 | 0.00 | 0.50 |
| Direct rope3d | SA4 (C-tokens never refined) | 0.24 | 3.9e-2 | 0.81 | 29 | 0.15 | 0.12 | 0.05 | 0.59 |
| a6000_b64 | TP2–extractor out | 3e-5–1e-3 | 0 | 0.20–0.23 | 50 | 0.00 | 0.00 | 0.00 | 0.50 |
| a6000_b64 | SA4 | 0.32 | 6.2e-2 | 0.75 | 32 | 0.17 | 0.16 | 0.06 | 0.59 |
| d8 | TP3–extractor out | 1e-6–4e-4 | 0 | 0.19–0.24 | 50 | 0.00 | 0.00 | 0.00 | 0.50 |
| d8 | SA4 | 0.28 | 6.2e-2 | 0.78 | 31 | 0.23 | 0.18 | 0.08 | 0.61 |
| random-init | TP1–TP4 image token | 0.03–0.08 | 2e-3–5e-3 | 0.97–0.98 | 19–20 | 0.28–0.36 | 0.04–0.07 | 0.15–0.21 | 0.61–0.62 |
| random-init | TP1–TP4 patch mean | 0.12–0.15 | 8e-3–1e-2 | 0.97–0.98 | 20–21 | 0.16–0.24 | 0.10–0.14 | 0.02–0.11 | 0.57–0.61 |
| random-init | SA4 | 0.15 | 7.7e-3 | 0.94 | 26 | 0.06 | 0.00 | 0.00 | 0.55 |

- **LOGO references** as in TODO 5: 50 cm (mean of the 5 other poses) and 18 cm (nearest other pose). Chance for the group probe is 0.17.
- **Non-geometric extractors are blind to the sweep.** In a6000_b64 and d8 the sweep shift inside the extractor is exactly 0, so every (B) probe there is at chance.
- **A nonlinear check on the output.** Distance from the clean training mean, used as a 1-D detector, gives AUC 0.57–0.64 at TP4 and the output. Its Spearman ρ with rotation magnitude is ≤ 0.37, and with translation magnitude ≈ 0. So a quadratic readout does not recover more than the linear probes.
- **Files.**
  - Scripts: `scripts/analysis/probe_features.py` (new: full-image extractor hooks, `--fp32`, legacy full-patch compat, K from `num_history`) and `scripts/analysis/fit_probes.py` (new extractor and `xdm` feature sets), via `probe_features.slurm` and `fit_probes.slurm <label> results/analysis/probes_fullpatch`. Figure and table: `scripts/analysis/plot_probes_fullpatch.py`.
  - Data: `results/analysis/probes_fullpatch/<model>.pt` and `fits/probe_results_<model>.json`.
  - Figure: [`figures/probe_fullpatch_extractor.png`](figures/probe_fullpatch_extractor.png). The grey band is the extractor. d8's 16 blocks are interleaved on the depth-4 axis.

**Reading:**
1. **Yes, in the rope3d extractors the miscalibration is linearly decodable inside the extractor, but only as rotation, and only in a vanishingly small subspace.**
   - Rotation magnitude and direction are best decodable in the patch tokens late in the stack: Refine TP3 R² 0.47 / 0.30, clean vs sweep 0.81; Direct TP4 0.60 / 0.41, 0.82.
   - **Translation is never decodable inside a trained extractor** (R² ≤ 0.05), even though the wrist cameras and the other external camera are calibrated, so a translated camera is inconsistent with them.
   - That fits RoPE being relative. A camera's own rigid translation is invisible within that camera. It shows up only through cross-camera consistency in the same-time blocks, which the trained extractors don't exploit; the random-init one does weakly (translation R² 0.10–0.14). A rotation about the world origin, by contrast, also rotates every within-camera displacement, so it is visible from one camera alone.
   - **All of this lives at the collapsed scale.** By TP2–TP3, spread and sweep shift are 1e-4–1e-3 of the token norm. The probes are scale-free after standardisation; the model is not. The head runs under bf16 (relative step ≈ 4e-3), and SA mixes these tokens with O(1) scene features.
2. **The rotation signal is lost at the readout to the per-image token, i.e. at the extractor output.**
   - In Refine, TP3–TP4 patch means carry R² 0.13–0.47 for rotation magnitude, but the TP4 image token handed to the head carries 0.09 (vector 0.05, clean vs sweep 0.60).
   - In Direct, the output token still carries 0.15 (vector 0.17). Then **delta_M is exactly constant** (spread 0 in fp32, sweep shift 0): the delta_M head maps that output to a single matrix, independent of both sample and miscalibration.
   - **Camera pose goes the same way.** Group accuracy falls from 0.95 at `enc` to 0.20–0.37 in the middle blocks of every trained extractor. It sometimes reappears in late patch means (Refine TP3 0.92, Direct TP4 0.95, again at 1e-4 scale), and ends at 0.27 (Refine) / 0.70 (Direct) at the output.
   - In a6000_b64 and d8 the extractor ends at chance (0.20–0.24), the same collapse as the pooled models in TODO 5.
3. **Not an architectural ceiling: the untrained rope3d extractor carries both signals at full scale.**
   - Random-init keeps pose at every block (0.97–0.98), with LOGO at the nearest-pose level.
   - Its image tokens carry the rotation error (R² 0.28–0.36; magnitude 0.15–0.21) and its patch means the translation (0.10–0.14). It passes them to the output with a spread of 0.08 rather than 1e-4.
   - So training actively removes both, with the same output collapse seen in the pooled models.
- **Downstream, nothing is gained from seeing geometry in the extractor.** The head's C-tokens re-acquire pose and some miscalibration by attending to the scene tokens, exactly as in TODO 5: SA4 rotation-vector R² 0.15–0.23, translation 0.12–0.19, clean vs sweep 0.59–0.62. This is the same in the geometric (Refine, Direct) and non-geometric (a6000_b64, d8) models, and the same as pooled exp 3 / control A. So the rope3d path's in-extractor signal does not reach the policy, and Direct's delta_M cannot be a geometric correction.
- **Caveats.**
  - The in-extractor probes read variations of 1e-4 of the token norm in fp32. They show the information is *present*, not *usable*, and the jagged depth curves for the trained models reflect that low signal-to-scale.
  - The same pose/task confound and linear-probe limits apply as in TODO 5.
  - The random-init floor shares the frozen SigLIP towers.

---

## Full-patch models: shared context for TODOs 12 and 13

All four use raw weights and their saved configs: K = 5, `video_deltam_full_image: true`, `patch_rope3d` and role as saved. Use the fixed loader, and apply legacy compat exactly where `load_model_for_evaluation` would. The a6000_b64 checkpoint needs `enable_legacy_checkpoint_compat` for its missing `image_readout`/`camera_readout`. Check that the `[load]` line and the compat path match online eval.

| Label | Checkpoint | Role | Sees patch xyz? |
|---|---|---|---|
| FP | `train_logs/PerAct2/peract2_orbital_vid_deltam_fullpatch_finetune_300k_a6000_b64/best.pth` | refine (head delta_M) | no |
| FP-d8 | `train_logs/PerAct2/peract2_orbital_vid_deltam_fullpatch_d8/best.pth` | refine, depth 8 | no |
| FP-R3D-refine | `train_logs/PerAct2/peract2_orbital_vid_deltam_refine_fullpatch_rope3d/best.pth` | refine | **yes** |
| FP-R3D-direct | `train_logs/PerAct2/peract2_orbital_vid_deltam_direct_fullpatch_rope3d/best.pth` | predict_delta_m (Direct) | **yes** |

The CPU check (TODO 11's Why) found every one of their extractors outputs a constant. Report each model's existing online IDLT numbers next to the results (`docs/research/results/README.md` and the IDLT campaign folders, method ids `video_fullpatch_best` etc.), so behaviour can be compared with robustness.

## TODO 12: full-patch models — C-tokens, delta_M and attention (TODOs 1, 3, 8 and 10 again)

**Status:** done

**Do**, on the four models above, with the same samples, seeds and conditions as before:
1. **C-tokens** (TODO 10):
   - **at head input:** magnitude, spread across samples, distance between all camera pairs, share explained by task and camera group;
   - position shift under heavy60, per camera;
   - layer-by-layer relative change clean → heavy, compared with scene tokens and trajectory tokens;
   - attention to and from them;
   - the masking and mean-replacement reliance checks.
2. **delta_M** (TODO 1), for FP, FP-d8 and FP-R3D-refine (head layer-wise) and FP-R3D-direct (one delta_M from the extractor):
   - ‖ΔM − I‖ per layer;
   - change between layers;
   - spread across samples and the share explained by task;
   - clean vs heavy;
   - camera 0 vs camera 1;
   - live predictor units.
3. **Attention** (TODO 3): external vs wrist share per layer, clean vs heavy; the FPS sampler's external share; per-token ratio; entropy.
4. One comparison table placing these four next to the pooled models (exp 3, 1b, VideoPooled) and the non-history models (DeltaM 140k, Base 160k) from TODOs 1, 3, 8 and 10.

**Results:**

Scripts and data:
- `scripts/analysis/fullpatch_suite.slurm` runs three parts:
  - `PART=attn`: `camera_attention.py --faithful`, job 3995049.
  - `PART=dm`: `delta_m_per_layer.py` and `delta_m_trunk_activity.py --faithful`, job 3995050.
  - `PART=ctok`: `ctoken_behaviour.py`, job 3995051.
- Summaries come from `plot_camera_attention.py`, `plot_delta_m_per_layer.py`, `plot_ctoken_behaviour.py` and `summarise_todo12.py` (the Direct delta_M). Data is in `results/analysis/todo12/{camera_attention,delta_m_per_layer,ctoken_behaviour}/`.
- Same 853 val samples, seeds and clean/train/heavy60 conditions as TODOs 1, 3, 8 and 10.

**Loading, matched to online eval.**
- `scripts/analysis/eval_faithful_load.py` now applies the same `enable_legacy_checkpoint_compat` rule as `load_model_for_evaluation`, using the same key test.
- All four run with raw weights, K=5 (the saved value, and what the IDLT plans pass), `full_image=True`, scene sampling `fps`, and consecutive history.
- `[load]` lines:
  - **FP:** legacy all-camera delta_M (`view_align_cameras=None`) **and** legacy readout. Its saved config is old-schema, so online eval applies both.
  - **FP-d8:** cameras [0,1], role refine.
  - **FP-R3D-refine:** `patch_rope3d=True`, role refine.
  - **FP-R3D-direct:** `patch_rope3d=True`, role `predict_delta_m`.
- The extractor gets per-patch tokens through the model's own `encode_inputs`. That includes `frame_pcd` from the miscalibrated point cloud for the rope3d variants. The miscalibration is drawn once per sample and shared by all 5 history frames, as on a fixed rig.
- Two script changes, both opt-in or new behaviour only:
  - `delta_m_per_layer.py` also records the Direct delta_M, which reaches the head as `history_view_align`.
  - `delta_m_trunk_activity.py` reads the extractor's `delta_m_head` when the head owns no predictor, and pools per-patch extractor inputs before the spread statistic.
- Recomputed attention matches the module output to ≤1.5e-6 relative (fp32).

**1. delta_M: one fixed matrix per layer in every full-patch model**

| | FP | FP-d8 | FP-R3D-refine | FP-R3D-direct |
|---|---|---|---|---|
| ‖ΔM − I‖, CA=SA0 / SA1 / SA2 / SA3 (cam 0 = cam 1) | 0.745 / 0.095 / 0.095 / 0.095 | 0.067 at every layer | 0.125 / **2.622** (clamp) / 0.125 / 0.125 | one delta_M: **0.115** |
| wrist cameras | 0.745 / 0.095 (legacy all-camera rule) | identity | identity | identity |
| spread across samples (RMS) | 0 at every layer | 0 | 0.004 at SA1 (task η² 0.26), else 0 | 0 |
| clean vs heavy60 ‖ΔM_c − ΔM_h‖ | 0 | 0 | 0.0055 at SA1, else 0 | 0 |
| cam 0 vs cam 1 | 0 | 0 | 0.005 (0.2°) at SA1 | 0 |
| across the 5 flow steps | 0 | 0 | 0.005 at SA1 | 0 (identical) |
| depth change, SA0→SA1 / SA1→SA2 | 0.71 (28°) / 0 | 0 / 0 | 3.05 (112°) / 3.05 (112°) | n/a |
| live predictor units of 120, CA / SA1 / SA2 / SA3 | 1 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 0 / 1 / 0 / 0 | 0 (extractor `delta_m_head`) |
| delta_M = exp(skew(bias)) where no unit is live | SA1–SA3 | every layer | CA, SA2, SA3 | yes |
| extractor current-frame output spread (input spread) | 0.0000 (0.021) | 0.00001 (0.024) | 0.00002 (0.028) | 0.00013 (0.022) |

**2. C-tokens**

| | FP | FP-d8 | FP-R3D-refine | FP-R3D-direct |
|---|---|---|---|---|
| what reaches the head | extractor output | extractor output | extractor output | **encoder C-tokens** (Direct leaves them untouched) |
| norm / spread across samples (ext) | 8.67 / 0 | 11.95 / 0 | 5.91 / ≤1e-4 | 1.3 / **0.051** |
| distance between cameras, 0–1 / 0–2 | 0 / 0 (4 identical copies) | 0 / 0 | 0.002 / 0.002 | 0.52 / 2.05 |
| η² task / camera group (ext) | n/a (no spread) | n/a | n/a | 0.72–0.79 / 0.17–0.18 |
| content clean vs heavy60 at head input | identical | identical | **max diff 0.07, relative 0.4–0.6%** (patch xyz) | identical |
| position shift, heavy60 (train) | 98 / 94 / 0 / 0 cm (14 / 13 cm) | same | same | same |
| relative change at sa3, C ext / C wr / FPS ext / FPS wr / traj | 0.087 / 0.014 / 0.118 / 0.020 / 0.024 | 0.080 / 0.018 / 0.137 / 0.033 / 0.032 | 0.086 / 0.017 / 0.123 / 0.028 / 0.032 | 0.059 / 0.025 / 0.122 / 0.033 / 0.035 |
| same at pos1 | 0.46 / 0.029 / 0.61 / 0.031 / 0.034 | 0.48 / 0.042 / 0.62 / 0.048 / 0.045 | 0.37 / 0.029 / 0.44 / 0.040 / 0.042 | 0.38 / 0.039 / 0.49 / 0.048 / 0.046 |
| traj → all C, SA / pos / rot | 2.1–8.5% / 0.8–3.1% / 1.9–2.8% | 0.7–6.4% / 1.0–3.0% / 1.9–3.0% | 2.8–8.1% / 1.1–3.7% / 2.1–2.8% | 2.1–2.4% / 1.0–1.4% / 1.9–2.0% |
| C rows → ext FPS (C0/1; C2/3) | 0.09–0.15; 0.09–0.15 | 0.10–0.17; 0.11–0.17 | 0.07–0.13; 0.07–0.13 | 0.09–0.12; 0.06–0.11 |
| heavy60 change in any C-token attention entry | ≤0.3 pp | ≤0.2 pp | ≤0.5 pp | ≤0.1 pp |
| pos_l2 (cm): clean → heavy60 | 1.25 → **1.26** | 1.25 → 1.95 | 1.26 → 1.89 | 1.19 → 1.83 |
| mask all C (clean / heavy60) | 1.27 / 1.27 | 1.28 / 1.97 | 1.27 / 1.90 | 1.19 / 1.83 |
| mask external C (clean / heavy60) | 1.26 / 1.27 | 1.26 / 1.95 | 1.26 / 1.89 | 1.19 / 1.83 |
| mean C (clean / heavy60); zero C (clean) | 1.25 / 1.26; 1.29 | 1.26 / 1.95; 1.28 | 1.26 / 1.89; 1.26 | 1.19 / 1.83; 1.19 |

**3. Attention.** External share = attention on cam 0+1 scene tokens / attention on all scene tokens. Values are clean → heavy60, Δ in pp; SE 0.02–0.09.

| Layer | FP | FP-d8 | FP-R3D-refine | FP-R3D-direct |
|---|---|---|---|---|
| ca0 | 50.5 → 51.1, +0.61 | 48.9 → 49.7, +0.74 | 45.0 → 45.7, +0.60 | 46.6 → 47.4, +0.84 |
| ca1 | 49.9 → 51.2, +1.30 | 50.8 → 50.7, −0.16 | 46.7 → 47.2, +0.50 | 47.6 → 48.7, +1.13 |
| sa0 | 14.6 → 15.2, +0.60 | 17.2 → 17.4, +0.25 | 14.0 → 14.2, +0.20 | 14.0 → 14.1, +0.11 |
| sa1 | 18.2 → 17.8, −0.45 | 19.0 → 18.8, −0.23 | 15.2 → 15.1, −0.06 | 15.0 → 14.9, −0.12 |
| sa2 | 14.9 → 13.5, −1.48 | 15.0 → 13.5, −1.55 | 14.4 → 13.3, −1.06 | 14.2 → 13.1, −1.05 |
| sa3 | 13.1 → 11.4, −1.62 | 14.3 → 12.6, −1.75 | 11.6 → 10.8, −0.88 | 11.1 → 10.2, −0.95 |
| pos0 | 9.0 → 7.1, −1.93 | 8.8 → 7.1, −1.78 | 7.5 → 6.4, −1.12 | 8.0 → 6.6, −1.33 |
| pos1 | 5.6 → 4.0, −1.60 | 6.2 → 4.2, −1.98 | 6.3 → 4.8, −1.54 | 6.1 → 4.7, −1.47 |
| rot0 / rot1 | −0.35 / −0.17 | −0.25 / −0.16 | −0.06 / +0.03 | −0.03 / −0.05 |

Other attention measurements:
- **FPS external share** (cam 0 / 1 / 2 / 3 tokens of 204):

  | Model | Tokens | External share |
  |---|---|---|
  | FP | 12.8 / 10.7 / 100.5 / 80.0 | 11.5% |
  | FP-d8 | 13.5 / 13.3 / 95.9 / 81.4 | 13.1% |
  | FP-R3D-refine | 14.1 / 8.4 / 99.9 / 81.7 | 11.0% |
  | FP-R3D-direct | 12.3 / 9.1 / 102.2 / 80.4 | 10.5% |

  Samples with no external FPS token: 0.6–1.5%.
- **Per-token ratio** (external share ÷ token share): 1.1–1.6 in SA; pos1 falls under heavy60 (FP 0.48 → 0.34, d8 0.47 → 0.32, R3D-refine 0.57 → 0.43, R3D-direct 0.58 → 0.44).
- **Entropy:** heavy60 lowers it by 0.02–0.06 nats in CA, SA and the position head, and by ≤0.006 in the rotation head.

**4. Comparison with the pooled and non-history models.** Offline numbers are this week's runs: TODOs 1 and 8 for delta_M, TODO 3 (K=5) and TODO 8 for attention, TODO 10 for C-tokens. IDLT is task-heldout, 13 tasks × 100 rollouts, for the same checkpoint. Calibrated and 60/60 cells come from `idlt_task_heldout_group_fixed_v03_video_k5`, `idlt_scaled_extreme_task_heldout_group_v0{1,2}` and `vdm_rope3d_sweep_task_heldout_group_v01`, aggregated with `plot_ablation_per_task.load`.

| Model | C-token spread at head input | ‖ΔM−I‖ CA / SA1 | delta_M spread (max over layers) | delta_M clean vs heavy | live trunk units (max) | FPS ext share | Δ ext share ca1 / pos1 (pp) | ext C-token rel. change pos1 | mask C, heavy60 Δ (cm) | offline pos_l2 clean → heavy60 (cm) | IDLT calibrated / 40-40 / 60-60 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| exp 3 200k (pooled, EMA) | ≤2e-4 | 0.70 / 2.66 | 0.06 | ≤0.010 | 2 | 18.8% | +0.58 / −1.73 | 0.25 | −0.05 | 1.03 → 1.40 | 77.5 / – / 71.6 |
| 1b 200k (pooled, EMA) | ≤1e-4 | (adaLN) | – | – | – | 10.0% | +0.11 / −1.14 | 0.31 | +0.03 | 0.99 → 1.53 | 80.8 / 71.5 / 70.4 |
| VideoPooled 84k (pooled) | 1e-5 | 0.68 / 0.08 | 0.024 | ≤0.0005 | 1.2 | – | – | – | – | 1.19 → 1.33 | 78.6 / 72.0 / 71.5 |
| DeltaM 140k (no history) | **0.109** | 0.55 / 0.55 | **0.40** | ≤0.020 | **7.6** | **30.0%** | +2.51 / +1.13 | 0.15 | +0.01 | 1.32 → 2.68 | – / 40.5 / 37.2 |
| Base 160k (no history) | **0.104** | (none) | – | – | – | **32.3%** | +2.33 / −3.21 | 0.12 | +0.02 | 1.49 → 3.33 | 70.8 / 49.8 / 27.1 |
| **FP** | 0 | 0.745 / 0.095 | 0 | 0 | 1 | 11.5% | +1.30 / −1.60 | 0.46 | +0.01 | 1.24 → **1.25** | 69.7 / 66.8 / 64.8 |
| **FP-d8** | 0 | 0.067 / 0.067 | 0 | 0 | 0 | 13.1% | −0.16 / −1.98 | 0.48 | +0.02 | 1.24 → 1.93 | no IDLT (SBRS best 79.2 at 140k) |
| **FP-R3D-refine** | ≤1e-4 | 0.125 / 2.62 | 0.004 | 0.0055 | 1 | 11.0% | +0.50 / −1.54 | 0.37 | +0.01 | 1.25 → 1.88 | 69.9 / 62.9 / 61.5 |
| **FP-R3D-direct** | 0.051 (encoder) | 0.115 (one) | 0 | 0 | 0 | 10.5% | +1.13 / −1.47 | 0.38 | 0.00 | 1.18 → 1.81 | 74.0 / 64.9 / 61.8 |

Figures (`figures/README.md`):
- `todo12_dm_layer_*.{png,svg}`: the six TODO 1 metrics for FP, FP-d8 and FP-R3D-refine.
- `todo12_cam_attn_external_share.{png,svg}` and `todo12_cam_attn_sa_mass_breakdown.{png,svg}`.
- `todo12_ctoken_rel_change.{png,svg}`.

**Reading:**
- **The full-patch models confirm TODO 11 from the inside.** Every one runs on constants. The three refine models hand the head a constant C-token, four identical copies. For FP-R3D-refine it is constant across samples but not quite across calibration: its patch xyz lets 60/60 move it by about 0.5%. Every head delta_M is one fixed matrix per layer, and the predictor trunks are fully dead (0–1 live units of 120). FP-d8's delta_M is exp(skew(bias)) at every layer. FP-R3D-refine reproduces exp 3's pattern exactly: a small constant everywhere except a clamp-limited 112° rotation at SA1. The Direct model's single extractor delta_M is one constant matrix (0.115 from I), identical for both cameras, every sample and both conditions. Seeing patch xyz (rope3d) does not make any of these delta_Ms respond to miscalibration.
- **Their attention behaves like the pooled history models, not the non-history ones.** The FPS sampler keeps 10.5–13% external tokens, against 30–32% for Base and DeltaM. The shift under 60/60 is the generic one: about +0.5 to +1.3 pp toward the external cameras in CA, and −1.5 to −2 pp in the position head. The C-tokens are low-attention pooling tokens with no reliance: masking or mean-replacing them changes offline error by ≤0.03 cm. FP-R3D-direct's C-tokens are sample-specific, like Base/DeltaM. Its sampler still keeps only 10.5% external tokens, though, so the low external share goes with these models' encoders and training, not with C-token collapse.
- **None of this separates the full-patch models from the pooled ones on robustness.** They sit below the pooled models on IDLT 60/60 (61.5–64.8 vs 70.4–71.6). They match them on every mechanism measured here: constant delta_M, constant or unused C-tokens, low external FPS share, the same attention shift. The robustness ordering is pooled > full-patch >> non-history. Only the last gap lines up with the attention and FPS-share difference, so whatever separates pooled from full-patch is elsewhere. Training length and recipe, and the scene tokens' own geometry (TODO 13's patching), are the candidates.
- **One offline anomaly to flag: FP shows essentially no offline damage at 60/60** (1.24 → 1.25 cm), while FP-d8, R3D-refine and R3D-direct lose about 0.65 cm. FP is the only one with the legacy all-camera delta_M: a large constant rotation (0.745) at CA/SA0 on all four cameras, wrist included. Its online drop is also the smallest of the four (69.7 → 64.8, against 8–12 points for the rope3d pair), so this may be real. TODO 4's kernel analysis applied to FP's CA/SA0 matrix would tell whether that rotation makes the scene-token RoPE insensitive to large shifts. I didn't test it here.
- **Caveats.**
  - These are raw `best.pth` checkpoints at different steps and from different recipes (FP-d8 is warm-started from VideoPooled; the rope3d pair has 160k/200k steps), so the pooled vs full-patch comparisons are correlations.
  - FP-d8 has no IDLT run.
  - `docs/research/results/README.md` §7 says the rope3d sweep has only two cells. The campaign directory now has all 13 tasks for calibrated through 60/60 for both rope3d models, so that note looks stale. The numbers above are from the complete cells.

---

## TODO 13: full-patch models — following and activation patching (TODOs 6, 7 and 9 again)

**Status:** done

**Do**, on the four models above, with the same samples and seeds:
1. **Following** (TODO 6): the following coefficient α for external-only, camera-0-only and camera-1-only errors, plus the all-four-cameras rigid control, at 10/10, 20/20, 40/40 and 60/60; share of predictions nearest the true target; orientation β.
2. **Activation patching** (TODOs 7 and 9), both directions:
   - C-token positions, FPS positions and dense positions;
   - **the extractor's patch xyz** (the `frame_pcd` it sees, for the rope3d variants);
   - proprio features;
   - delta_M replay (head per call, or the Direct delta_M);
   - combinations.

   Check that positions, extractor xyz and proprio together recover 100%.
3. One comparison table next to TODOs 7 and 9 (exp 3, 1b, A, Base, DeltaM).

**Results:**

**Scripts.** My TODO 6 / 7 / 9 scripts with a new `--eval-loader` option. It loads through `evaluation.checkpoints.load_model_for_evaluation` itself, then sets `visual_num_history` from the saved config, as online plans do.
- `scripts/analysis/follow_miscal.py` (job 3995125) and `scripts/analysis/activation_patching.py` (job 3995124), both via `scripts/analysis/todo13.slurm` with `PART=follow|patch`.
- Patching additions:
  - a `FramePcd` hook on `video_deltam.forward`, which records or overrides the extractor's `frame_pcd`;
  - an `ext_xyz` patch: re-encode with the extractor alone seeing the other condition's patch xyz, then copy the extractor outputs (C-tokens, register, Direct delta_M);
  - an `ext_out` patch: copy those outputs directly, as a cross-check;
  - a `direct_dm` patch: Direct delta_M (`history_view_align`).
- Plots: `plot_follow_miscal.py --fig-prefix todo13_` and `plot_activation_patching.py --fig-stem todo13_patch_fraction_recovered`.
- Data: `results/analysis/todo13/{follow_miscal,activation_patching}/` (`summary.json`, `summary_table.txt`).
- Figures: `figures/todo13_follow_alpha_pos.png`, `todo13_follow_beta_rot.png`, `todo13_follow_closest_share.png`, `todo13_patch_fraction_recovered.png`.

**Loading and inputs match online eval** (`[load]` and `[checkpoint compat]` lines):
- All four models: raw weights, K=5, `full_image=True`, `visual_history_sampling=consecutive`. Online uses the same causal window, left-padded with the first frame, which mirrors the dataset. Proprio is past→current with `proprio_num_history=3`, the eval default.
- **FP** (a6000_b64) takes both compat paths, exactly as `load_model_for_evaluation` prints them:
  - the legacy all-camera delta_M (`view_align_cameras=None`);
  - the pre-readout full-image forward (`legacy_readout=True`).
- FP-d8: `patch_rope3d=False`, refine.
- FP-R3D-refine: `patch_rope3d=True`, refine.
- FP-R3D-direct: `patch_rope3d=True`, `predict_delta_m`. The head builds no delta_M predictor, so "mechanism" is the Direct delta_M.
- Miscalibration is applied to every history frame's extrinsics, as online.
- Same 853 val samples and seeds as TODOs 6, 7 and 9.

**Checks.**
- Exactness passed for all four models: max |Δ| = 0.0 for patch-nothing = heavy and patch-everything = clean.
- For the rope3d models, `ext_xyz` and `ext_out` give identical predictions (max |Δ| = 0.0).
- **Positions + extractor xyz + proprio = 1.00** in both directions for FP-R3D-refine and FP-R3D-direct. Positions + proprio is 1.00 for FP and FP-d8, which have no xyz in the extractor. For FP-R3D-refine, positions + proprio *without* the extractor xyz is 0.99–1.00 (acc 1.03), because its extractor xyz moves the C-tokens slightly.

**Which encoded inputs differ, clean vs heavy60** (relative L2):
- dense xyz 0.76, FPS/C-token xyz 0.24–0.41;
- **proprio_feats: FP 0.047, FP-d8 0.029, FP-R3D-refine 0.072, FP-R3D-direct 0.056.** For comparison, 0.07–0.17 in exp 3, 1b, A, Base and DeltaM;
- FP-R3D-refine C-token features: 0.0014, through the extractor's RoPE on patch xyz;
- history register: 0.0002;
- FP-R3D-direct's Direct delta_M: **exactly 0**, because its extractor is constant (TODO 11);
- scene and language features: 0.

**Online robustness, for reference** (`docs/research/experiments/README.md`, `results/README.md`, weekly log 2026-09-22):
- FP: IDLT held-out calibrated 69.7, 60/60 **64.8**; scaled-extreme 50/60° 66.3 (task-seen) / 64.9 (held-out).
- Rope3D sweep, task-heldout G1–G6: direct 68.4, refine 66.9. G7 extreme rotation: direct 63–68, refine 60–64.
- FP-d8: SBRS best 79.2 at 140k. I found no IDLT 60/60 number for it.

**1. Following** (TODO 6 protocol). Following coefficient α with 95% CI where shown. Full sweep in `summary.json`.

| Model | both 60: α₀ / α₁ | only cam 0 60: α₀ | only cam 1 60: α₁ | both 10: α₀ | rigid, all 4 cams, 10 / 20 / 40 / 60 | closest to true p, both 60 |
|---|---|---|---|---|---|---|
| FP | 0.0014 [0.0004, 0.0023] / 0.0006 | 0.0014 | 0.0007 | 0.0055 | **0.71 / 0.62 / 0.50 / 0.41** | 99.9% |
| FP-d8 | 0.0010 / 0.0001 | 0.0011 | 0.0001 | 0.0058 | 0.71 / 0.62 / 0.50 / 0.41 | 99.8% |
| FP-R3D-refine | 0.0023 [0.0009, 0.0039] / 0.0014 | 0.0025 | 0.0014 | 0.0054 | 0.74 / 0.65 / 0.52 / 0.43 | 99.5% |
| FP-R3D-direct | 0.0031 [0.0016, 0.0048] / 0.0017 | 0.0030 | 0.0016 | 0.0056 | 0.73 / 0.64 / 0.52 / 0.43 | 99.6% |
| *exp 3 / 1b / A (TODO 6)* | *0.0028 / 0.0020 / 0.0036 (α₀)* | | | *0.006–0.007* | *0.63, 0.59, 0.68 at 10 → 0.39–0.41 at 60* | *99.7–99.9%* |

More on following:
- **Orientation.** For external-only miscalibration, |β| ≤ 0.005 at 60/60 and every CI includes 0. With the rigid all-camera error, β = 0.05–0.13: slightly more than the pooled models' 0.02–0.08, but still small.
- **Per task.** At both-60, α against the clean prediction is ≤ 0.012 for every task (largest: put_bottle_in_fridge, n=73, the rope3d models).
- **Error and shift at 60/60** (pos L2 cm clean → both / cam 0 only / cam 1 only; mean shift from the clean prediction):

  | Model | clean → both / cam 0 / cam 1 | shift (cm) |
  |---|---|---|
  | FP | 1.25 → 1.26 / 1.26 / 1.24 | 0.29 |
  | FP-d8 | 1.25 → 1.95 / 1.88 / 1.71 | 0.97 |
  | FP-R3D-refine | 1.26 → 1.89 / 1.80 / 1.70 | 0.97 |
  | FP-R3D-direct | 1.19 → 1.83 / 1.70 / 1.59 | 0.94 |

**2. Activation patching, share of the position-L2 degradation.** Forward recovered / reverse caused; 95% CI on forward in brackets. TODO 7 and TODO 9 columns are repeated for comparison.

| Patch | FP-d8 | FP-R3D-refine | FP-R3D-direct | exp 3 | 1b | A | DeltaM 140k | Base 160k |
|---|---|---|---|---|---|---|---|---|
| clean → heavy pos L2 (cm) | 1.25 → 1.95 | 1.26 → 1.89 | 1.19 → 1.83 | 1.04 → 1.41 | 1.00 → 1.54 | 1.20 → 2.05 | 1.32 → 2.69 | 1.49 → 3.30 |
| samples > 1 cm worse | 38 | 34 | 37 | 55 | 67 | 66 | 244 | 264 |
| C-token xyz | −0.00 / 0.01 | 0.00 / −0.00 | 0.01 / −0.00 | 0.19 / 0.13 | 0.01 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| FPS xyz | 0.05 / 0.04 | 0.07 / 0.04 | 0.15 / 0.09 | 0.28 / 0.20 | 0.12 / 0.11 | 0.06 / 0.03 | 0.26 / 0.32 | 0.20 / 0.19 |
| dense xyz | 0.03 / 0.01 | 0.04 / 0.01 | 0.06 / 0.01 | 0.16 / 0.11 | 0.09 / 0.06 | 0.10 / 0.05 | 0.27 / 0.34 | 0.17 / 0.22 |
| all head positions | 0.06 [0.02, 0.13] / 0.07 | 0.08 [0.03, 0.15] / 0.06 | 0.16 [0.08, 0.29] / 0.14 | 0.51 / 0.55 | 0.18 / 0.21 | 0.13 / 0.12 | 0.32 / 0.83 | 0.23 / 0.51 |
| extractor patch xyz | – | 0.00 / 0.00 | 0.00 / 0.00 | – | – | – | – | – |
| **proprio_feats** | **0.93** [0.88, 0.97] / 0.94 | **0.93** [0.86, 0.98] / 0.92 | **0.86** [0.74, 0.93] / 0.84 | 0.45 / 0.49 | 0.79 / 0.82 | 0.88 / 0.87 | 0.17 / 0.68 | 0.49 / 0.77 |
| delta_M replay (head) or Direct delta_M | 0.00 / 0.00 | 0.00 / −0.00 | 0.00 / 0.00 (Direct) | −0.00 / −0.01 | 0.00 / 0.01 (adaLN) | – | 0.02 / 0.01 | – |
| positions (+ extractor xyz) + proprio | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

- **Rotation L1:** proprio recovers 0.96 (FP-d8), 1.06 (FP-R3D-refine) and 1.00 (FP-R3D-direct). Positions recover ≤ 0.03.
- **Accuracy within 1 cm:** proprio recovers 0.60–0.64 and positions 0.25–0.50.
- **Damaged subsets** (more than 1 cm worse): heavy errors 16–17 cm. Proprio still recovers 0.86–0.95 there.

**FP (a6000_b64) is barely damaged offline,** so its shares are undefined. Heavy60 pos L2 is 1.26 cm against 1.25 clean, rotation L1 is 0.0472 against 0.0480, and only 9 samples get more than 1 cm worse. The fraction denominators are about zero (bootstrap CIs span ±20). On those 9 samples, heavy positions alone cause all the damage (reverse 0.99–1.00) and proprio none.

**Reading:**
- **The full-patch models don't follow miscalibrated external cameras either.** α is ≤ 0.003 at 60/60 and 0.005–0.006 at 10/10, the same as the pooled models; at least 99.5% of predictions stay nearest the true target. They do follow a rigid whole-scene shift, slightly more than the pooled models (α 0.71–0.74 at 10/10 against 0.59–0.68). Their orientation follows rigid rotation a little more too (β ≤ 0.13). So full-patch history makes the head no more trusting of inconsistent external geometry. If anything, it is a bit more geometric when the cameras agree.
- **Three of the four models break the same way as 1b and A.** In FP-d8, FP-R3D-refine and FP-R3D-direct, proprio_feats carries 86–93% of the position damage and about all the rotation damage. Head positions carry 6–16%, and C-token positions carry none.
- **The extractor carries none.** Feeding the rope3d extractor clean instead of miscalibrated patch xyz recovers exactly 0 (FP-R3D-direct) or less than 0.005 (refine). FP-R3D-direct's Direct delta_M is bit-identical clean vs heavy60, so the Direct mechanism cannot correct anything. This confirms, by intervention, TODO 11's finding that the extractors are constant.
- **FP is the exception: offline almost undamaged.** Its proprio features move least under miscalibration (relative change 0.047, against 0.07–0.17 elsewhere), and the error barely moves (+0.01 cm). Yet its IDLT 60/60 is 64.8, below exp 3's 71.6. **Offline keypose error does not rank online robustness across model families.** Among the history models, the IDLT ordering (exp 3 > FP > others) does not match the offline degradation ordering (FP ≈ 0 < exp 3 < FP-d8 / R3D). FP's online failures must come from something this offline test doesn't see: rollout drift, closed-loop history, or non-keypose steps.
- **Across all nine models, the per-model share through proprio** is one consistent axis: Base 0.49, exp 3 0.45, DeltaM 0.17 (forward), the rest 0.79–0.93. It does not track IDLT. The other axis, head-position sensitivity, is high only in the non-history models and exp 3. The `gripper_context_head` remains the most common single route for miscalibration into the action.
- **Caveats.**
  - The degradations are small (+0.6–0.7 cm), so the CIs are wide (for example FP-R3D-direct positions [0.08, 0.29]).
  - FP's fractions are ill-posed: there is essentially no gap to explain.
  - The rope3d extractors' xyz path would matter only if the extractor were alive.
  - Offline, keypose only, one heavy60 draw per sample.
  - The online numbers come from different protocols (IDLT, scaled-extreme, the Rope3D sweep) and are not strictly comparable across the four models.

---

## TODO 14: what does exp 4a's live extractor do? (pre-LN extractor, exp 3's recipe)

**Context:** exp 4a (`train_logs/PerAct2/peract2_orbital_vid_pooled_randhist_deltam_prenorm_scratch`, EMA, K = 5) uses a pre-LayerNorm history extractor (`video_deltam_prenorm: true`). Its extractor stays alive.

On real data at 120k (`results/analysis/exp4_delta_m_120k/`), against exp 3 at 120k:
- C-token spread into the head is 0.41 vs 0.0;
- delta_M spread across samples is 0.74 / 0.74 / 0.20 / 0.20 at CA / SA1 / SA2 / SA3, vs 0 / 0.19 / 0.04 / 0.0007;
- camera 0 and 1 differ, by 0.65–0.96 at CA and SA1;
- clean vs 60/60 is 0 at CA (appearance-only input) and 0.03–0.06 later.

Robustness ladder (13 × 20, EMA): 100k 71.9 / 58.5 and 120k 73.5 / 59.6 (calibrated / 60-60). Full 215-cell suites at 200k are queued on its training (arrays 3998707–13).

**Run each part on 4a at 120k now, and again at 200k** (`interm_step_200000.pth`, due about 9 h after 8 Oct 20:30), with exp 3 at the same steps as the reference. Use `:ema:ckpt` load specs, check the `[load]` line shows `visual_num_history=5`, and use the same samples and seeds as before.

### TODO 14a: attention and C-token behaviour

**Status:** done

TODO 3 and TODO 10 metrics:
- external vs wrist share per layer, clean vs heavy;
- FPS external share;
- C-token content spread, camera-pair distance and task η²;
- layer-by-layer relative change clean → heavy;
- attention to and from the C-tokens;
- masking and mean-replacement reliance.

**Question:** now that the C-tokens carry per-scene content, does the head attend to them or rely on them more than in exp 3?

**Results:**

**Setup.**
- Scripts: `scripts/analysis/todo14a.slurm`, run as `PART=attn` (`camera_attention.py --faithful`) and `PART=ctok` (`ctoken_behaviour.py`), with `STEP=120000`. Jobs 3998722 and 3998723.
- Data: `results/analysis/todo14a/{camera_attention,ctoken_behaviour}/`. Summaries come from `plot_camera_attention.py` and `plot_ctoken_behaviour.py`.
- Both models load EMA weights with `:ema:ckpt`. The `[load]` lines show `visual_num_history=5`, cameras [0,1], role refine. 4a's saved config has `video_deltam_prenorm: true`.
- Same 853 val samples, seeds and conditions as TODOs 3 and 10. Recomputed attention matches the module output to 1.1e-6.

**120k: exp 4a vs exp 3**

| | **exp 4a 120k** (pre-LN, live) | exp 3 120k (collapsed) |
|---|---|---|
| **C-token content at head input** | | |
| spread across samples (per-channel std), cam 0 / 1 / 2 / 3 | **0.66 / 0.59 / 0.62 / 0.72** | 0 / 0 / 0 / 0 |
| token norm | 11.2–11.9 | 11.1 |
| distance between cameras, 0–1 / 0–2 / 2–3 | **5.95 / 9.64 / 10.68** | 0 / 0 / 0 |
| η² task / camera group (per camera) | 0.47–0.53 / 0.05–0.07 | n/a (no spread) |
| content clean vs heavy60 at head input | identical (appearance-only input) | identical |
| **Attention** | | |
| FPS external token share | 19.0% (cam 0–3: 21.3 / 17.4 / 88.1 / 77.1) | 21.3% (22.9 / 20.6 / 86.1 / 74.4) |
| external share ca1, clean → heavy60 | 45.2 → 45.4 (+0.23) | 48.2 → 49.1 (+0.91) |
| external share sa3 | 20.7 → 18.2 (**−2.51**) | 28.1 → 27.7 (−0.39) |
| external share pos1 | 10.2 → 7.7 (**−2.53**) | 16.3 → 14.9 (−1.45) |
| external share rot0 / rot1 | −0.08 / −0.14 | 0.00 / 0.00 |
| largest entropy drop | −0.16 nats (sa2) | −0.065 (sa2) |
| traj → all C-tokens, sa0 / sa1 / sa2 / sa3 | 3.8 / **8.0 / 8.2** / 3.6% | 3.1 / 4.3 / 3.6 / 2.4% |
| traj → C-tokens, pos / rot | 2.4–4.0% / 2.4–3.0% | 1.5–3.0% / 2.1–2.2% |
| which C-tokens the traj queries read (sa2, pos1) | wrist: sa2 cams 2/3 3.5/3.1% vs ext 0.7/0.8%; pos1 2.4/1.3% vs 0.2/0.2% | all four alike (0.3–1.1%) |
| traj → C under heavy60 | sa2 8.2 → **11.2%**, all from the **external** C-tokens (cams 0/1: 0.7/0.8 → 2.7/2.7%; wrist 3.5/3.1 → 3.2/2.7%); elsewhere ≤0.3 pp | sa2 3.6 → 4.9%, also external (0.6/0.8 → 1.4/1.4%); elsewhere ≤0.1 pp |
| C rows → ext FPS / wrist FPS / other C | 0.17–0.23 / 0.62–0.80 / 1.5–9.8% (sa1–sa2 up to 9.8%) | 0.16–0.27 / 0.65–0.79 / 1.2–2.8% |
| **Relative change clean → heavy60** (C ext / C wr / FPS ext / FPS wr / traj) | | |
| sa1 | 0.077 / 0.022 / 0.079 / 0.025 / 0.032 | 0.117 / 0.029 / 0.108 / 0.048 / 0.037 |
| sa3 | **0.258** / 0.041 / 0.174 / 0.057 / 0.050 | 0.131 / 0.042 / 0.115 / 0.069 / 0.044 |
| pos0 | 0.375 / 0.047 / 0.308 / 0.060 / 0.045 | 0.229 / 0.040 / 0.144 / 0.059 / 0.037 |
| pos1 | 0.320 / 0.038 / 0.313 / 0.050 / 0.030 | 0.241 / 0.040 / 0.213 / 0.052 / 0.028 |
| **Reliance: offline pos_l2 (cm)** | | |
| clean / heavy60 | 1.14 / 1.57 | 1.14 / 1.81 |
| clean: mask all C / mask external C / mask wrist C | **1.76** / 1.14 / **1.45** | 1.18 / 1.14 / 1.16 |
| clean: mean C / zero C | **2.07 / 2.41** | 1.14 / 1.18 |
| heavy60: mask all C / mask external C / mask wrist C / mean C | 2.30 / 1.53 / 1.91 / 2.74 | 1.76 / 1.73 / 1.81 / 1.81 |

Figure: `figures/todo14a_ctoken_rel_change_120k.{png,svg}`, the item-3 per-layer relative change for both models.

**200k: exp 4a vs exp 3.** Jobs 4001660 and 4001661 were submitted by the watcher `results/analysis/todo14a/watch_200k.sh` once `interm_step_200000.pth` existed. Both `[load]` lines show `visual_num_history=5`.

| | **exp 4a 200k** | exp 3 200k |
|---|---|---|
| C-token spread at head input, cams 0–3 | **0.66 / 0.59 / 0.62 / 0.71** | ≤2e-4 |
| distance between cameras, 0–1 / 0–2 / 2–3 | 4.87 / 9.19 / 10.44 | ≤0.003 |
| η² task / camera group | 0.44–0.50 / 0.05–0.06 | n/a |
| FPS external token share | 17.6% | 18.8% |
| external share, clean → heavy60: ca1 / sa3 / pos1 | +0.09 / **−2.48** / **−2.39** pp | +0.58 / −0.47 / −1.73 pp |
| largest entropy drop | −0.16 nats (sa2) | −0.063 (sa2) |
| traj → all C, sa1 / sa2 / pos1 (clean) | **8.3 / 6.9** / 3.3% (pos1: wrist 1.8 + 1.2%, ext 0.1 + 0.1%) | 4.4 / 2.7 / 1.2% |
| traj → C at sa2, clean → heavy60, cams 0 / 1 / 2 / 3 | 0.6 / 0.8 / 2.9 / 2.6% → **2.5 / 2.6** / 2.6 / 2.3% | 0.5 / 0.6 / 0.8 / 0.8% → 1.2 / 1.2 / 0.8 / 0.8% |
| relative change under heavy60 at sa3, C ext / C wr / FPS ext / FPS wr / traj | **0.229** / 0.033 / 0.148 / 0.047 / 0.041 | 0.123 / 0.035 / 0.096 / 0.062 / 0.038 |
| same at pos1 | **0.378** / 0.039 / 0.337 / 0.049 / 0.035 | 0.248 / 0.036 / 0.257 / 0.066 / 0.034 |
| offline pos_l2, clean / heavy60 (cm) | 1.02 / **1.26** | 1.04 / 1.41 |
| clean: mask all / external / wrist C | **1.68** / 1.02 / **1.44** | 1.07 / 1.04 / 1.04 |
| clean: mean C / zero C | **1.87 / 2.32** | 1.04 / 1.08 |
| heavy60: mask all / external / wrist C / mean C | 1.99 / **1.19** / 1.66 / 2.37 | 1.36 / 1.35 / 1.42 / 1.41 |

Figure: `figures/todo14a_ctoken_rel_change.{png,svg}` covers both models at 120k and 200k. The 120k-only `todo14a_ctoken_rel_change_120k` is kept.

**Reading (120k, confirmed at 200k):**
- **Yes, the head now uses the C-tokens, and it relies on them, but mostly on the wrist ones.**
  - 4a's C-tokens carry real per-scene, per-camera content: spread 0.6–0.7, about six times Base's, with large camera-to-camera distances. About half of it is explained by task, and almost none by camera group.
  - The trajectory queries attend to them about twice as much as in exp 3 (8% at sa1–sa2).
  - Removing them as keys costs +0.62 cm clean (exp 3: +0.04). Replacing their content with the mean costs +0.93 cm, because it also changes the live delta_M.
  - Masking only the external C-tokens costs nothing (1.14 → 1.14). And the traj → C mass at sa2 and pos1 goes 4–10× more to cameras 2/3 than 0/1.
  - So the reliance is mainly on the **wrist-camera C-tokens**. A wrist-only mask (job 3998766, `results/analysis/todo14a_v2/`, which reproduces every other number exactly) costs +0.31 cm clean and +0.34 cm at heavy60.
  - The external C-tokens are redundant with the wrist ones rather than useless. Masking them alone costs nothing, but masking them on top of the wrist ones adds another +0.31 cm.
- **Under 60/60 the external C-tokens are the most-perturbed tokens again** (relative change 0.26 at sa3, twice exp 3's). Their content is unchanged; the change comes through their miscalibrated positions and the SA mixing. The head shifts scene attention away from the external cameras somewhat more than exp 3 does (−2.5 pp at sa3 and pos1, against −0.4 and −1.45). At sa2, though, the trajectory queries attend *more* to the displaced external C-tokens (+2 pp each). This corrects my first 120k note, which credited the sa2 increase to the wrist C-tokens.
- **The C-tokens still aren't a route for the damage.** Masking the external C-tokens recovers only 0.04 of 4a's 0.43 cm heavy60 damage; exp 3's figure is 0.08 of 0.67. 4a's offline damage is smaller (0.43 vs 0.67 cm). 4a has no lever over the external cameras through C-token content, since that content can't see miscalibration. Its gain more plausibly comes from its stronger use of calibrated wrist information (C-tokens and attention), which TODO 14c's patching can test.
- **200k holds the 120k picture.**
  - 4a's C-tokens stay live (spread 0.6–0.7).
  - The head still relies on them: masking all costs +0.66 cm clean, masking the wrist ones +0.42, masking the external ones 0. Exp 3 is unaffected by any mask (≤0.03 cm).
  - The extra sa2 attention under 60/60 still goes to the external C-tokens (0.6–0.8 → 2.5–2.6%).
  - 4a's offline damage is smaller: 0.24 cm against exp 3's 0.37.
  - One change from 120k: at 200k, masking the external C-tokens under 60/60 recovers a sizeable share of 4a's damage, 1.26 → 1.19 cm (0.07 of 0.24, ~30%); exp 3's is ~15% (0.06 of 0.37). Through their content the external C-tokens are redundant with the wrist ones, but under 60/60 their displaced positions start to carry some damage in 4a.
- **Answer to the question.** Yes: with per-scene content, the head attends to the C-tokens about twice as much at sa1–sa2 and relies on them (+0.6 cm when removed), where exp 3 ignores its constant C-tokens. The reliance is on the calibrated **wrist** C-tokens. The external ones are redundant through their content and a small damage route through their position, so this content still doesn't let 4a correct the external geometry. 4a's better 60/60 offline error goes with stronger use of wrist information, not with fixing the external cameras.

### TODO 14b: probes

**Status:** done

TODO 5 / TODO 11 probes on 4a: true camera pose (including leave-one-group-out) and the injected miscalibration (0–60° / 0–60 cm sweep), at every depth:
- encoder C-tokens;
- **inside the extractor**, after each block;
- extractor output;
- head SA layers;
- delta_M per layer.

**Question:** a live pooled extractor still sees appearance only. Does it at least keep camera pose, which exp 3 discards? Does any delta_M layer become decodable for pose or miscalibration?

**Results (120k, both EMA).**

*Setup.*
- **Load.** `[load]` shows `visual_num_history=5` for both models. `[arch]` shows pooled, refine, depth 4, random history, `prenorm=True` (4a) / `False` (exp 3).
- **Protocol: TODO 5's** — the same 3000 training / 853 val rows and seeds, and the clean / train-style / 0–60° · 0–60 cm sweep conditions on cameras [0, 1].
- **Features** are fp32. New: `--extractor_blocks` records the pooled extractor's current-frame token per camera after every same-time (ST) and temporal (TP) block. In 4a that is the residual stream, before `out_norm`.
- **delta_M per layer** is probed as flattened ΔM − I. `dm0` is predicted from the extractor output, and `dm1`–`dm3` from the C-tokens after SA1–SA3.
- **Columns.** "Spread" is across-sample std / mean norm. "Sweep shift" is ‖clean − sweep‖ / norm on the same sample. Both are on val, cameras 0+1.
- **References.** Chance for the group probe is 0.17, and a task one-hot gets 0.33. The leave-one-group-out (LOGO) mean-pose baseline is 50 cm and the nearest-other-pose baseline 18 cm.

| Stage | 4a spread | 4a group acc | 4a LOGO pos (cm) | exp 3 spread | exp 3 group acc | exp 3 LOGO pos (cm) |
|---|---|---|---|---|---|---|
| enc | 0.75 | 0.93 | 20 | 0.48 | 0.92 | 23 |
| ST1 / TP1 | 0.83 / 0.70 | 0.81 / 0.74 | 31 / 38 | 2e-2 / 3e-3 | 0.82 / 0.37 | 29 / 48 |
| TP2 / TP3 / TP4 | 0.84 / 1.0 / 1.1 | 0.62 / 0.59 / 0.57 | 45 / 47 / 47 | 6e-4 / 2e-4 / 5e-5 | 0.24 / 0.20 / 0.19 | 50 |
| extractor out | 0.86 | **0.57** | 47 | 5e-5 | 0.19 | 50 |
| SA1 / SA2 / SA3 / SA4 | 0.81 / 0.83 / 0.75 / 0.68 | 0.56 / 0.57 / 0.77 / 0.80 | 48 / 48 / 33 / 29 | 0.27–0.41 | 0.68–0.71 | 43 / 41 / 39 / 40 |
| delta_M (from hist, SA1, SA2, SA3) | 0.25 / 0.28 / 0.07 / 0.06 | 0.28 / 0.34 / 0.29 / 0.25 | 48–51 | 0 / 0.09 / 0.02 / 1.5e-4 | 0.20–0.22 | 49–51 |

*Injected miscalibration (sweep).* Values are 4a / exp 3.

| Stage | Sweep shift | Rotation-vector R² | Translation-vector R² | Rotation-magnitude R² | Clean vs sweep acc |
|---|---|---|---|---|---|
| enc → extractor out | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0.50 / 0.50 |
| SA2 | 0.06 / 0.07 | 0.07 / 0.20 | 0.03 / 0.13 | 0.00 / 0.06 | 0.54 / 0.60 |
| SA4 | 0.21 / 0.08 | **0.27** / 0.19 | **0.21** / 0.12 | 0.13 / 0.05 | 0.65 / 0.61 |
| delta_M, all layers | ≤ 0.014 / ≤ 0.005 | ≤ 0.004 / ≤ 0.01 | 0.00 / 0.00 | 0.00 / 0.00 | ≤ 0.52 / ≤ 0.51 |

- **Task decodability**, 13-way, camera 0. 4a: enc 1.00, extractor out 1.00, delta_M from hist 0.67, delta_M from SA1 0.77. exp 3: 1.00, 0.15, 0.16, 0.40.
- **Files.**
  - Data: `results/analysis/probes_exp4a/{exp4a,exp3}_{120k,200k}.pt` and `fits/probe_results_*.json`. Table: `summary_table_120k.md`.
  - Scripts: `scripts/analysis/probe_features.py --fp32 --extractor_blocks` and `scripts/analysis/plot_probes_exp4a.py --step 120k`.
  - Figure: [`figures/probe_exp4a_120k.png`](figures/probe_exp4a_120k.png). The grey band is the extractor; hollow squares are delta_M at its input depth.

**Reading (120k).**
- **4a's extractor is alive and keeps part of the camera identity that exp 3 throws away.**
  - Its spread stays at 0.7–1.1 through every block, against exp 3's fall to 5e-5.
  - The output still classifies the camera group at 0.57 (exp 3: 0.19, chance), and the task perfectly.
  - But identity decays steadily block by block (0.93 → 0.57), and LOGO position error at the output is 47 cm, near the 50 cm mean baseline. What survives is a group or scene label, not a continuous pose: the same per-viewpoint appearance code that TODO 5 found in the encoder, partly degraded.
- **4a's delta_M varies per sample, but with task, not camera pose.**
  - delta_M spread is 0.25–0.28 at hist/SA1, against 0 / 0.09 in exp 3. Task decodes at 0.67–0.77, but camera group only at 0.25–0.34 (task one-hot alone gives 0.33), and LOGO pose is at the mean baseline.
  - So a live input to delta_M yields a scene/task-conditioned RoPE mixing, not a pose estimate.
- **Miscalibration: as expected, invisible up to the extractor output in both models**, since the pooled extractor sees appearance only (zero sweep shift).
  - It appears only in the head SA C-tokens, through attention to scene tokens at miscalibrated positions. There 4a ends somewhat higher at SA4 (rotation vector 0.27, translation 0.21, clean vs sweep 0.65; exp 3: 0.19 / 0.12 / 0.61), but lower at SA1–SA2.
  - **No delta_M layer in either model decodes the miscalibration** (R² ≤ 0.01, accuracy ≤ 0.52). 4a's delta_M moves by only ≤ 1.4% under the sweep, and not in a decodable direction.
- **Ruled out at 120k:** a live extractor making delta_M a pose or miscalibration estimator. Its new per-sample content is task and scene.
- **Caveats.**
  - The same pose/task confound and linear-probe limits apply as in TODO 5.
  - The exp 3 numbers are at 120k. TODO 5's 200k fp16 features were not reused, because this protocol needs fp32 and the in-extractor blocks; exp 3 at 200k was re-extracted.

**Results (200k, both EMA).** Same protocol. `[load]` shows `visual_num_history=5`, and `[arch]` shows `prenorm=True` for 4a. exp 3 at 200k was re-extracted at fp32 with the extractor blocks; TODO 5's fp16 features have no in-extractor stages.

| Stage | 4a spread | 4a group acc | 4a LOGO pos (cm) | exp 3 spread | exp 3 group acc | exp 3 LOGO pos (cm) |
|---|---|---|---|---|---|---|
| enc | 0.73 | 0.92 | 21 | 0.45 | 0.93 | 25 |
| ST1 / TP1 | 1.0 / 0.84 | 0.71 / 0.62 | 41 / 44 | 2e-2 / 2e-3 | 0.82 / 0.36 | 30 / 47 |
| TP2 / TP3 / TP4 | 0.98 / 1.1 / 1.2 | 0.53 / 0.48 / 0.46 | 48 / 49 / 50 | 5e-4 / 1e-3 / 3e-4 | 0.20–0.23 | 50 |
| extractor out | 0.91 | **0.44** | 50 | 3e-4 | 0.20 | 50 |
| SA1 / SA2 / SA3 / SA4 | 0.84 / 0.87 / 0.75 / 0.64 | 0.48 / 0.49 / 0.75 / 0.79 | 49 / 48 / 34 / 31 | 0.27–0.42 | 0.68–0.69 | 44 / 44 / 41 / 41 |
| delta_M (from hist, SA1, SA2, SA3) | 0.29 / 0.32 / 0.07 / 0.07 | 0.27 / 0.33 / 0.24 / 0.22 | 49–50 | 0 / 0.01 / 0.03 / 0 | 0.20–0.24 | 50 |

*Injected miscalibration (sweep).* Values are 4a / exp 3.

| Stage | Sweep shift | Rotation-vector R² | Translation-vector R² | Rotation-magnitude R² | Clean vs sweep acc |
|---|---|---|---|---|---|
| enc → extractor out | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0.50 / 0.50 |
| SA2 | 0.06 / 0.08 | 0.06 / 0.16 | 0.02 / 0.11 | 0.00 / 0.03 | 0.53 / 0.59 |
| SA4 | 0.18 / 0.08 | **0.26** / 0.16 | **0.21** / 0.12 | 0.11 / 0.04 | 0.63 / 0.60 |
| delta_M, all layers | ≤ 0.015 / ≤ 0.003 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | ≤ 0.51 / 0.50 |

- **Task decodability**, 13-way, camera 0. 4a: enc 1.00, extractor out 1.00, delta_M from hist 0.69, delta_M from SA1 0.76. exp 3: 1.00, 0.16, 0.16, 0.47.
- **Files.** Table: `results/analysis/probes_exp4a/summary_table_200k.md`. Figure: [`figures/probe_exp4a_200k.png`](figures/probe_exp4a_200k.png).

**Reading (200k).**
- **Same picture as at 120k, and the pose 4a keeps is eroding with training.**
  - The extractor stays alive (spread 0.9–1.2) and perfectly task-decodable.
  - Camera-group accuracy at its output drops from 0.57 (120k) to **0.44** (200k), and LOGO position error is at the 50 cm mean baseline.
  - So the live extractor keeps scene/task content and a fading camera-identity code, never a continuous pose.
- **4a's delta_M is per-sample but task-driven.** Spread is 0.29–0.32 at hist/SA1. Task decodes at 0.69–0.76, camera group at ≤ 0.33 (about the task-only level), and miscalibration at R² 0.00 at every layer.
- **Miscalibration appears only in the head SA C-tokens**, through attention to displaced scene tokens. 4a is somewhat more decodable than exp 3 at SA3–SA4 (rotation-vector R² 0.26 vs 0.16, translation 0.21 vs 0.12, clean vs sweep 0.63 vs 0.60), but less at SA1–SA2. Neither model's delta_M reads it.
- **Answer to the question.** A live pooled extractor partly keeps camera identity, a group label that fades from 0.57 to 0.44 between 120k and 200k, where exp 3 discards it entirely. But **no delta_M layer becomes decodable for pose or miscalibration** at either checkpoint. Keeping the extractor alive changes what delta_M depends on (task) without making it a geometric correction.

### TODO 14c: following and activation patching

**Status:** in progress (intervention agent)

TODO 6 and TODO 7 on 4a:
- following coefficients, with single-camera and rigid controls and the magnitude sweep;
- patching: positions (C-token, FPS and dense), proprio, delta_M replay per call, combinations.

**Question:** does 4a's per-scene delta_M carry or reduce any of the 60/60 damage, where exp 3's carried 0%?

**Results:**

**Scripts.** `scripts/analysis/follow_miscal.py` and `scripts/analysis/activation_patching.py` with `--eval-loader`, via `scripts/analysis/todo14c.slurm` (`PART=follow|patch`, `STEP=120000|200000`).
- 120k jobs: 3998724 (patching) and 3998725 (following).
- Data: `results/analysis/todo14c/{follow_miscal,activation_patching}/`, with `summary.json` and `summary_table.txt` in each.
- Figures: `figures/todo14c_follow_alpha_pos.png`, `todo14c_follow_beta_rot.png`, `todo14c_follow_closest_share.png` and `todo14c_patch_fraction_recovered.png`. These are made from whatever checkpoints are present.

**Setup.**
- **Loading:** both models load through `load_model_for_evaluation`, EMA. The `[load]` lines show `visual_num_history=5`, `view_align_cameras=[0, 1]`, role refine and `history_sampling=random` for both.
- **Data and seeds:** the same 853 val samples, seeds and heavy60 draws as TODOs 6 and 7.
- **Exactness passes for both models.** Patching nothing reproduces heavy, and patching every input reproduces clean (max |Δ| = 0.0). Positions + proprio = 1.00 in both directions.
- **Inputs that differ, clean vs heavy60:** only the dense xyz, the FPS/C-token xyz and `proprio_feats` (relative change 0.15 for 4a, 0.17 for exp 3).
- **4a's extractor output does not change under miscalibration.** The C-tokens entering the head (`fps_scene_feats`) are bit-identical clean vs heavy60 in 4a too, because the live extractor still sees appearance only. So 4a's delta_M can differ between clean and heavy only through SA mixing in the head (TODO 14: 0.03–0.06).

#### 120k

**Following** (TODO 6 protocol). Following coefficient α with 95% CI. The last two columns are pos L2 at 60/60 in cm (clean → both / cam 0 / cam 1) and the mean shift from the clean prediction.

| Model | both 60: α₀ / α₁ | only cam 0 60 | only cam 1 60 | both 10: α₀ | rigid 10 / 20 / 40 / 60 | closest to true p (both 60) | pos L2 60/60 (cm) | shift (cm) |
|---|---|---|---|---|---|---|---|---|
| **exp 4a 120k** | 0.0025 [0.0012, 0.0036] / 0.0017 | 0.0023 | 0.0012 | 0.0078 | 0.68 / 0.59 / 0.47 / 0.39 | 100% | 1.14 → 1.57 / 1.38 / 1.36 | 0.93 |
| exp 3 120k | 0.0044 [0.0027, 0.0066] / 0.0032 | 0.0044 | 0.0034 | 0.0077 | 0.62 / 0.56 / 0.48 / 0.41 | 100% | 1.14 → 1.81 / 1.51 / 1.50 | 1.16 |

Orientation β: |β| ≤ 0.007 at 60/60 for external-only errors in both models; with the rigid all-camera error, β = 0.04–0.08.

**Activation patching, position L2.** Forward recovered / reverse caused; 95% CI on forward in brackets.

| Patch | exp 4a 120k | exp 3 120k |
|---|---|---|
| clean → heavy pos L2 (cm) | 1.14 → 1.57 | 1.14 → 1.81 |
| samples > 1 cm worse | 94 | 93 |
| **delta_M replay, per call** | **−0.00** [−0.02, 0.01] / 0.00 | −0.01 [−0.01, −0.00] / −0.00 |
| C-token xyz | 0.18 / 0.06 | 0.12 / 0.09 |
| FPS xyz | 0.29 / 0.22 | 0.35 / 0.29 |
| dense xyz | 0.15 / 0.08 | 0.14 / 0.09 |
| all positions | 0.54 [0.42, 0.66] / 0.48 | 0.52 [0.38, 0.69] / 0.55 |
| proprio_feats | 0.52 [0.37, 0.64] / 0.46 | 0.45 [0.25, 0.59] / 0.48 |
| C-token xyz + proprio | 0.62 / 0.64 | 0.57 / 0.56 |
| all positions + delta_M | 0.52 / – | 0.52 / – |
| all positions + proprio | 1.00 / 1.00 | 1.00 / 1.00 |

**Rotation L1** (forward / reverse):

| Patch | exp 4a 120k | exp 3 120k |
|---|---|---|
| proprio | 0.89 / 0.81 | 0.60 / 0.47 |
| all positions | 0.19 / 0.11 | 0.53 / 0.40 |
| C-token xyz | 0.14 | 0.35 |
| delta_M | 0.00 / −0.00 | −0.03 / −0.00 |

**Reading (120k):**
- **4a's live delta_M carries none of the 60/60 damage, and it doesn't reduce any either.**
  - Replaying the clean delta_M into the heavy run, call by call, recovers −0.00 [−0.02, 0.01] of the position damage and 0.00 of the rotation damage. Replaying the heavy delta_M into the clean run causes 0.00.
  - If delta_M were compensating the miscalibration, forcing the clean delta_M into the heavy run would *increase* the error (a negative share). It doesn't.
  - So per-scene variation in delta_M (TODO 14: spread 0.74 at CA and SA1, cameras differing by 0.65–0.96) is not a geometric-correction channel. Its clean→heavy changes come from SA mixing and are inert for the action, as exp 3's were.
- **4a is somewhat less damaged offline** (+0.43 cm against +0.67 cm), with the same split: about half through head positions and half through proprio.
  - 4a shifts rotation damage further onto proprio (0.89 against 0.60). In exp 3, the C-token positions carry 0.35 of the rotation damage; in 4a only 0.14.
  - 4a leans slightly less on the miscalibrated cameras: α 0.0025 against 0.0044 at 60/60, both negligible.
  - It follows a rigid whole-scene shift slightly more at 10/10 (0.68 against 0.62).
  - So the live extractor changes how much the head reacts to C-token positions, not whether delta_M matters.
- **Caveat.** The online ladder at 120k (13 × 20) has 4a at 59.6 for 60/60; I found no exp 3 120k IDLT number to compare. Offline keypose error does not rank online robustness across families (TODO 13), so the smaller offline damage is only suggestive.

#### 200k

Pending: the 4a checkpoint `interm_step_200000.pth` doesn't exist yet. A background poll will launch the same two jobs on 4a 200k and exp 3 200k.

---

## TODO 15: does exp 4b's input skip carry the task / camera-group fingerprint too?

**Status:** claimed (probing agent)

**Why:** TODO 14b found that 4a's live extractor output identifies the task (100%) and partly the camera group (0.44–0.57). At 200k, 4a is about 6 points below exp 3 at IDLT 60/60 and about 4 below calibrated. Hypothesis: the head uses this "task × camera setup" fingerprint as a shortcut that doesn't transfer to IDLT's unseen task × group pairings.

Exp 4b (`peract2_orbital_vid_pooled_randhist_deltam_inputskip_scratch`, EMA, K = 5, `video_deltam_input_skip: true`) has a collapsed stack plus a learned projection of the **raw** current-frame C-tokens (language-mixed encoder appearance) added onto its output. Its robustness ladder so far is the best of the exp 4 runs (160k: 80.4 calibrated / 65.8 at 60/60, 13 × 20).

**Do:** TODO 14b's probes (task 13-way, camera group 6-way, leave-one-group-out position, injected-miscalibration sweep) on:
- **4b at 160k**, now, and again at **200k** when `interm_step_200000.pth` exists (training due about 18:45 on 9 Oct);
- **4a at 160k** and **exp 3 at 160k**, for a same-step comparison (reuse TODO 14b's 200k numbers for those two at 200k).

Probe at:
- encoder C-tokens;
- the stack output (before the skip);
- the skip term alone (`input_skip_proj(raw current-frame tokens)`);
- the extractor output (stack + skip);
- head SA C-tokens;
- delta_M per layer.

Use EMA weights at K = 5, and check the `[load]` and `[arch]` lines (`input_skip=True` for 4b).

**Report:** one table, task and camera-group accuracy at each depth, for 4b / 4a / exp 3. Does 4b carry the fingerprint at its extractor output and into delta_M? If 4b carries it as strongly as 4a but robustness is better, the shortcut hypothesis is in trouble.

**Results:**

**Reading:**
