# No-wrist, 4 tasks, from scratch

**Status:** both runs done (100k steps, `best.pth` written); ladders complete
**Week introduced:** [22–29 Sept](../../weekly_logs/2026-09-29.md) (TODO 4)

## Question

How much do the wrist cameras carry? Train with external cameras only.

## Setup

**Tasks:** `handover_item`, `take_tray_out_of_oven`, `sweep_to_dustpan`,
`dual_push_buttons`. Chosen from the per-task gallery:

- Success rates: `docs/research/results/figures/task-gallery/task_sr.md`
- Rollout GIFs (5/task, Base 160k and VideoPooled 84k, calibrated, each task at
  its training camera group): `docs/research/results/figures/task-gallery/rollouts/`; eval plan
  `all_tasks_rollout_videos_v01`
- Stills: `docs/research/results/figures/task-gallery/<task>/{G1,G7}.png`

**Data:** the existing zarr, nothing regenerated. `OrbitalPeract2NoWrist` drops
the wrists at load time and the new `train_tasks` key filters to the 4 tasks,
leaving 2,917 of 7,607 train records.

**Code changes:**

- `train_tasks` reaches the train dataset, the validation dataset and the
  workspace normalizer; the filter reads only the kept records.
- Eval takes the camera set from the checkpoint's saved dataset and slices the
  model's cameras **after** miscalibration is applied.
- Existing checkpoints all saved `dataset=OrbitalPeract2`, so their evals are
  unchanged.

| | |
|---|---|
| Configs | `peract2_orbital_nowrist4_base.yaml`, `peract2_orbital_nowrist4_vid_pooled.yaml` |
| Training | from scratch, 100k steps, global batch 128 on 2 GPUs (64/GPU) |
| Ladder | `instructions/eval_plans/nowrist4_{base,video}_checkpoint_ladder_v01.json`, from 30k |
| Jobs | Base 3966637 (+ continuation 3966638), VideoPooled 3966639 (+ 3966640) |
| Memory | peak at 64/GPU on an A5000: 8.1 GB VideoPooled, 6.4 GB Base |

## Results

Plateau around **40–46%**: Base 42.5, VideoPooled 46.2.

Ladder, mean over the 4 tasks where all four finished:

| Step | Base | VideoPooled |
|---|---|---|
| 30k | 36.8 | — |
| 40k | 37.5 | 46.2 |
| 50k | 42.5 | — |
| 60k | 41.0 | — |
| 70k | — | 41.2 |
| 80k | — | 38.8 |
| 90k | 26.2 | 43.8 |

Per task: `sweep_to_dustpan` 85–100, `take_tray_out_of_oven` 45–75,
`handover_item` 0–25 (VideoPooled somewhat better), `dual_push_buttons` 0–15.

## Reading

Without wrist cameras, `handover_item` and `dual_push_buttons` **collapse** for
both models — the with-wrist 13-task models score about 75–80 on both. Adding
history does not recover them. `sweep_to_dustpan` does not need the wrists at
all. The camera slicing in the no-wrist eval path works.

## Caveats and open questions

- At 100k steps each record is seen about 2.6× as often as in the 13-task runs.
- There are no with-wrist 4-task controls, so the comparison is against the
  13-task models on these 4 tasks. **Open: are those controls needed?**

## Robustness without wrists (submitted 1 Oct)

IDLT task-heldout, calibrated and 10/10 → 60/60, 100 rollouts, on `sweep_to_dustpan` (held-out G1) and
`take_tray_out_of_oven` (held-out G4) only — the two tasks the no-wrist models solve. Checkpoints: Base 50k,
VideoPooled 40k (best on the ladder). Plans `instructions/eval_plans/*_nowrist4.json` from
`scripts/eval/create_nowrist_idlt_plans.py`; arrays 3984443–3984446 (32 cells). Compare against the 13-task
with-wrist models on the same two tasks.

Results (% of 100 rollouts; calibrated, 10, 15, 20, 30, 40, 50, 60):

| Model | sweep (held-out G1) | oven (held-out G4) |
|---|---|---|
| Base 160k, with wrists (13 tasks) | 99 98 98 98 99 98 98 98 | 42 23 42 22 18 15 3 0 |
| VideoPooled 84k, with wrists (13 tasks) | 69 76 71 81 78 69 64 67 | 78 81 74 75 76 76 75 79 |
| Base 50k, no wrists | 69 0 0 1 0 0 0 0 | 20 16 0 16 12 5 0 0 |
| VideoPooled 40k, no wrists | 0 70 0 0 0 0 0 0 | 19 19 11 3 1 0 0 0 |

Without wrists both models fail on held-out camera groups and collapse under miscalibration; the sweep cells are
all-or-nothing per calibration realization (configs verified in the logs). With clean wrists VideoPooled is flat.
So on these tasks the robustness depends on the clean wrist cameras — confounded by 4-task training and no
with-wrist 4-task control.
