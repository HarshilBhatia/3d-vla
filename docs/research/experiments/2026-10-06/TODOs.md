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

**Status:** open

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
