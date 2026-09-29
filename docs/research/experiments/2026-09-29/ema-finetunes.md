# EMA fine-tunes

**Status:** done (all 6 runs finished 20k, 26 Sept)
**Week introduced:** [22–29 Sept](../../weekly_logs/2026-09-29.md)

## Question

Does exponential moving averaging of the weights lift the ablation checkpoints,
and does it damp the checkpoint-to-checkpoint swing that makes the SBRS ladder
hard to read?

## Setup

- Each of the six ablation checkpoints is fine-tuned for 20k steps with
  `use_ema: true` on 2 GPUs (global batch 128), saving every 5k.
- Evaluated at 5k/10k/15k/20k on **both** raw and EMA weights, via the new
  eval-owned key `eval_weights: raw|ema` (commit `64b5a70`).
- **EMA schedule:** warm-up from decay 0 at the start; about 0.9966 at 5k and
  0.9986 at 20k, so the effective averaging window is only about 300–700 steps.

| Arm | Starts from | Job |
|---|---|---|
| `ema20k_vidpooled` | VideoPooled `best.pth` (84k) | 3974952 |
| `ema20k_randhist` | random history 50k | 3974953 |
| `ema20k_noviewalign` | no-view-align 140k | 3974954 |
| `ema20k_regtest_a` | A 160k | 3974955 |
| `ema20k_regtest_c` | C 100k | 3974956 |
| `ema20k_base` | Base 160k | 3974957 |

## Results (26 Sept, 20:45)

SBRS mean (6 tasks × 20 rollouts), raw / EMA:

| Run | 5k | 10k | 15k | 20k | Mean over 4 checkpoints: raw → EMA |
|---|---|---|---|---|---|
| VideoPooled | 76 / 78 | 72 / 74 | 78 / 78 | 74 / 72 | 75.0 → 75.5 |
| Random history | 57 / 78 | 72 / 75 | 75 / 82 | 67 / 81 | 67.8 → **79.0** |
| No delta_M | 68 / 78 | 66 / 75 | 75 / 81 | 68 / 76 | 69.3 → **77.5** |
| A | 63 / 71 | 63 / 71 | 63 / 72 | 67 / 72 | 64.0 → 71.5 |
| C | 63 / 70 | 69 / 75 | 58 / 72 | 62 / 73 | 63.0 → 72.5 |
| Base | 72 / 63 | 67 / 67 | 68 / 72 | 62 / 73 | 67.3 → 68.8 |

## Reading

- **History models:** EMA is at or above raw at 19 of 20 checkpoints, and it
  narrows the checkpoint-to-checkpoint swing (random history: raw 57–75,
  EMA 75–82).
- **Base:** barely moves.
- **A and C** are still about 7 points below the warm-started models even with
  EMA.

## Against the starting checkpoint (27 Sept)

`figures/ema20k_ladders_per_task.png` (from `scripts/plot_ema_ladders.py`) adds step 0: the starting checkpoint's own SBRS score, before any fine-tuning.

| Run | Step 0 (start) | Best EMA, 5–20k | Best raw, 5–20k |
|---|---|---|---|
| Random history | 83 | 82 | 75 |
| No delta_M | 83 | 81 | 75 |
| A | 80 | 72 | 67 |
| C | 79 | 75 | 69 |
| Base | 74 | 73 | 72 |
| VideoPooled | not laddered at K=5 | 78 | 78 |

- **The fine-tune dropped performance** at its first checkpoint for every run, most for the raw weights: random history went 83 → 57.
- **Likely cause:** the warm start loads weights only, so Adam restarts at lr 3e-4 with empty moment estimates, which knocks the model off its optimum.
- **What EMA actually does here:** it *recovers* most of that drop, but no EMA checkpoint beats its starting checkpoint.
- **So the raw-vs-EMA gap measures robustness to the fine-tune's disruption,** not a gain over the original model.
- **A cleaner test** would be EMA from the start of a run, or a fine-tune that keeps the optimizer state (a true resume).

## Robustness evals of the best EMA checkpoints (submitted 27 Sept)

- **What:** one checkpoint per run, the best by EMA SBRS (later step on ties), on the same IDLT task-heldout and G7 extreme protocols as before. Plans: `instructions/eval_plans/*_ema20k.json`; 1,290 cells in 7 arrays (3977686–3977717).
- **Checkpoints:** VideoPooled 15k, random history 15k, no delta_M 15k, A 20k, C 10k, Base 20k.
- **An accidental cancel:** an interrupted cancel command cancelled about 1,050 pending cells for the five non-random-history models, not the 67 first reported; a grouped `sacct` view was misread.
  - The 67 G7 translation cells were resubmitted first (3977736).
  - On 27 Sept, 17:56 the remaining 985 missing cells were found by checking result files and resubmitted (3978720–3978725).
- **Scope trimmed 27 Sept, 23:20 (user):** for the 5 non-random-history models, the IDLT low-noise conditions are cut to calibrated and 10/10. The 195 pending bias, 3/1 and 5/5 cells in 3978725 were cancelled (verified: 130 pending remain in that array).
- **Random history EMA 15k is complete.** IDLT held-out, original 50k → EMA 15k: calibrated 75 → 78, 10/10 74 → 80, 20/20 74 → 80, 40/40 73 → 77, 60/60 72 → **77**.

## Robustness results: original checkpoint → best EMA checkpoint (complete 29 Sept)

Figures: [`figures/ema_vs_original.png`](figures/ema_vs_original.png) (task means) and one per model, [`figures/ema_per_model_<model>.png`](figures/README.md) (every task). All are indexed in [`figures/README.md`](figures/README.md).

**IDLT held-out** (13 tasks × 100 rollouts):

| Model | calib | 10/10 | 20/20 | 40/40 | 60/60 |
|---|---|---|---|---|---|
| VideoPooled | 79 → 81 | 78 → 80 | 78 → 79 | 72 → 77 | 71 → 74 |
| Random history | 75 → 78 | 74 → 80 | 74 → 80 | 73 → 77 | 72 → **77** |
| No delta_M | 77 → 78 | 78 → 77 | 74 → 77 | 72 → 73 | 71 → 71 |
| A | 75 → 75 | 74 → 76 | 67 → 70 | 65 → 66 | 63 → 64 |
| C | 74 → 78 | 75 → 78 | 72 → 74 | 66 → 71 | 62 → 69 |
| Base | 71 → 76 | 68 → 73 | 61 → 63 | 50 → 57 | 27 → **39** |

**G7 extremes** (6 tasks × 20 rollouts): no consistent change. Translation is mostly within noise. Large rotations got worse for several history models; for example no delta_M went 75 → 62 at 180°.

**Reading:**
- On IDLT, EMA helps almost every model. The gain is largest under severe miscalibration: Base +12 at 60/60, C +7, random history +5, VideoPooled +3.
- No delta_M and A barely move.
- **Random-history EMA 15k is the most robust model** in this study: 77 at 60/60.
- On G7 there is no benefit, so the IDLT gain doesn't transfer to extreme rotations.

## Operational notes

- **Resubmitted 26 Sept, 13:10** after the launcher's ARM-list brace bug (fixed
  in `069ef48`). Each run was queued three ways, and the first copy to claim
  `train_logs/PerAct2/peract2_orbital_ema20k_<run>` wins:
  - single-node 2 GPUs at 200 GB (3974961–3974966);
  - single-node 2 GPUs lean: 12 CPUs, 120 GB, 4 workers per GPU (3974973–3974978);
  - 2 nodes × 1 GPU on `all` (supervisors log to `logs/multinode/sup_ema20k_*.log`).
  - `scripts/train/race_many.sh` then cancels the run's other pending copies
    (log: `logs/train/race_ema20k_*.log`). Copies that start later exit through
    `claim_run.sh`.
- **Also found:** the old VideoPooled SBRS ladder was evaluated with
  `num_history: 1` (broken-evaluator period), so its "83.3 at 84k" is
  unreliable. A K=5 re-ladder of its 10k–100k checkpoints was submitted
  (jobs 3974886–3974895, method `video-pooled-k5`).
