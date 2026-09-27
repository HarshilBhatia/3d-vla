# EMA fine-tunes

**Status:** done (all 6 runs finished 20k, 26 Sept)
**Week introduced:** [22–29 Sept](../log/2026-09-29.md)

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

`docs/figures/ema20k_ladders_per_task.png` (from `scripts/plot_ema_ladders.py`) adds step 0: the starting checkpoint's own SBRS score, before any fine-tuning.

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
- **An accidental cancel:** an interrupted cancel command had already cancelled 67 pending G7 translation cells (3977686_[41-107]) before the user stopped it. They were resubmitted as 3977736.

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
