# Experiments

One page per experiment that is running, recently finished, or still shaping
the plan. Each page answers: what it tests, exact config/ARM/jobs, results,
how to read them, and what is still open.

A page lives here while it is **interesting**. Once it is settled and the
numbers are final, the numbers move to [`../results/`](../results/) and the
page becomes history. Week-by-week narrative is in [`../log/`](../weekly_logs/).

New experiment → new `config/experiment/*.yaml`, a matching `ARM` in the
launcher, and a page here. Never a pile of CLI overrides.

## Active and recent

| Experiment | Question | Status | Headline |
|---|---|---|---|
| [camera-adaln](2026-10-06/camera-adaln.md) | Can C-token adaLN (1a per camera, 1b pooled, 1c static) replace delta_M and the register? | ready | — |
| [direct-randhist](2026-10-06/direct-randhist.md) | Is Direct delta_M from random history enough on its own? | ready | — |
| [pooled-no-view-align](2026-09-29/pooled-no-view-align.md) | Is view alignment needed if you have history? | stopped at 281k | **No.** Matches VideoPooled on every robustness suite |
| [random-history-no-time](2026-09-29/random-history-no-time.md) | Does history need order and recency? | active | **No.** Matches robustness while barely using history |
| [from-scratch-noalign-randhist](2026-09-29/from-scratch-noalign-randhist.md) | Does that survive random init — and is the history register the channel? | done | **Register not needed:** C ≈ A everywhere (IDLT 60/60 61.6 vs 62.8); from scratch is about 10 below warm start at 60/60 |
| [camframe-rope6d](2026-09-29/camframe-rope6d.md) | Can delta_M bridge camera frame → robot frame with no extrinsics? | active | SBRS 53.3 at 200k; fine tasks ~20 |
| [ema-finetunes](2026-09-29/ema-finetunes.md) | Does EMA lift the ablations and damp ladder noise? | done | Yes for history models: random history 67.8 → 79.0 |
| [history-probes](2026-09-29/history-probes.md) | Which input channel carries the robustness? | done | External cameras essential; history is a task-progress cue |
| [g7-viewpoint-sweep](2026-09-29/g7-viewpoint-sweep.md) | Is there a viewpoint memorisation cliff? | done | No cliff out to 3.5 m and the far side |
| [no-wrist-4task](2026-09-29/no-wrist-4task.md) | How much do the wrists carry? | done | Plateau 40–46%; handover and buttons collapse |
| [fullpatch-d8](2026-09-29/fullpatch-d8.md) | Does depth 8 beat depth 4? | stopped at 281k | No |

## Shared context

**G7 is inside the training camera grid.** All training cameras sit on one
sphere (r ≈ 1.31 m) around (0.25, 0, 0.95), at azimuth ±40/50° and elevation
10/35/60°. G7 is at left az 43° / right −47.5°, elevation ≈ 47.5°. So "unseen
G7" results measure **interpolation, not extrapolation**.

**Current best model:** VideoPooled-Refine at 84k
(`peract2_orbital_video_deltam_external_warmstart_k5v_k3p_a5000_resume/best.pth`).

| Model | ULRS G7 calibrated | IDLT held-out calibrated | IDLT held-out 60°/60 cm |
|---|---|---|---|
| Base 160k | 72.4 | 70.8 | 27.1 |
| DeltaM 140k | 75.5 | 72.7 | 37.2 |
| VideoPooled 84k | **78.5** | **78.6** | **71.5** |
| FullPatch | — | 69.7 | 64.8 |

Protocol definitions (SBRS, ULRS G7, IDLT) are in
[../reference/eval-protocols.md](../../handbook/eval/protocols.md); the standing
numbers are in [../results/README.md](../results/README.md).
