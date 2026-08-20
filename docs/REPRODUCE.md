# Reproducing the 3DFA August 2026 campaign

Everything needed to re-run, re-evaluate, or audit the results in
[`docs/status/experiments.md`](status/experiments.md). That file is the *findings*
log; this file is the *provenance* map — for every table there, which commit, which
sky YAML, which data, and which eval invocation produced it.

Companion artifact: the **egress bundle** (`3dfa_egress_<date>.tar.zst`, ~19 GB
uncompressed) holds checkpoints, the wandb export, the orbital dataset, and the eval
result JSONs. Where a section below says "bundle member", it means a path inside that
tarball. See [Egress bundle layout](#egress-bundle-layout).

---

## 1. Environment

Python is pinned to **3.11** (`requires-python = ">=3.11,<3.12"`); `torch==2.9.1` /
`torchvision==0.24.1` come from the PyTorch CUDA 12.8 index.

```bash
git clone git@github.com:HarshilBhatia/3d-vla.git && cd 3d-vla
uv sync --frozen --python 3.11
```

`--frozen` is deliberate: resolve against `uv.lock`, do not re-resolve. Every
training job in this campaign built its venv with exactly
`uv sync --frozen --no-install-project --python 3.11`.

### The `clip` extra

Only the `peract2_base_nhist3_clip_b200` run needs it (`backbone=clip`, OpenAI CLIP
RN50 + FPN). It is a git dependency and is lazily imported in
`modeling/encoder/vision/clip.py`, so SigLIP2 runs never touch it:

```bash
uv sync --frozen --extra clip
```

On the cluster this was a **separate venv**,
`/k8s-nfs/harsvbha/3dfa/venv_clip`, rather than the extra layered onto the shared
`/k8s-nfs/harsvbha/3dfa/venv`: the shared venv is the interpreter other running jobs
execute from, so it is not safe to mutate in place. Both venvs pin the same
`torch 2.9.1+cu128` / `torchvision 0.24.1+cu128`. The CLIP RN50 vision weights are
pre-staged at `/k8s-nfs/harsvbha/3dfa/clip-cache/RN50.pt`.

Separately, note that both arms of an experiment pair submit at the same instant and
share `$VENV`, so the setup blocks use a serialized (flock-guarded) build — a bare
`[ ! -x $VENV/bin/python ]` check is not atomic and lets one arm run against a
half-built venv.

### transformers 5.x note

`pyproject.toml` declares `transformers>=4.45` and `uv.lock` resolves it to a **5.x**
release. SigLIP2 loads fine there, but note two things if you ever re-resolve:

* The lockfile is the source of truth for what these results were produced with — a
  bare `uv sync` (no `--frozen`) can move `transformers` and silently change backbone
  behaviour. Always `--frozen` when reproducing.
* `>=4.45` is a floor, not a tested range. If you must upgrade, re-run the offline
  val gate before trusting any success rate.

### Online evaluation runs in a container, not this venv

PyRep / RLBench / CoppeliaSim are **deliberately absent** from `pyproject.toml`.
Online eval runs from an ECR image:

```
913524929094.dkr.ecr.us-east-1.amazonaws.com/rfm-h-eval-job:hb-3dfa-peract2-20260811
```

---

## 2. Data sources

### PerAct2 (13 bimanual tasks) — public

From the HuggingFace `katefgroup` release (public). Staged on cluster NFS as zarr:

```
/k8s-nfs/harsvbha/3dfa/data/peract2_zarr/Peract2_zarr/{train,val}.zarr
```

Download helper: `scripts/rlbench/download_peract2.py`. Instructions:
`instructions/peract2/instructions.json`. `dataset=Peract2_3dfront_3dwrist`.

Note this zarr has **no `demo_id` array**, which gates the visual-history path — so
`num_history=3` on PerAct2 is a **proprio-history-only** ablation (proprio is stored
`(N, 3, 2, 8)`; `RLBenchDataPreprocessor.process_proprio` slices `[:, :num_history]`).
Images stay single-frame at any `nhist`.

### Orbital PerAct2 — generated for this campaign

13 bimanual tasks x 6 camera groups (G1-G6) x 30 episodes = **2,340 episodes**,
collected 11-12 Aug 2026 on a local docker swarm. 4 cameras/episode (orbital pair +
2 over-shoulder). Final zarr: **1053 train / 117 val** episodes.

| where | path |
|---|---|
| bundle member | `datasets/orbital_peract2/{zarr,shards}/` |
| cluster NFS (train) | `/k8s-nfs/harsvbha/3dfa/data/orbital_peract2/` |
| local staging | `/local/home/harsvbha/3dfa_data/orbital_peract2/` |

The bundle carries **both** `zarr/` (the final train/val used by every orbital run)
and `shards/` (all **78** raw shards — a superset including camera groups no training
run consumed). `zarr/` alone is enough to reproduce; `shards/` is there so a
different task→group mapping can be built without recollecting.

Regenerate the zarr from shards with `scripts/../data/processing/convert_to_zarr/`
helpers; the collection driver is `data/generation/orbital/collection.py`.

Train/eval camera mapping: task *i* trains on groups `[i, i+1, i+2] mod 6`, eval
group is `i+3` — `instructions/peract2_orbital_task_group_mapping.json`. **G7 was
never collected** and remains a fully-OOD untested option.

### Eval test seeds

```
s3://far-research-internal/harsvbha/3dfa/eval/peract2_test
```

100 seeds per task. Standard PerAct2 evals use 25/variation; orbital evals use
`NUM_DEMOS_TOTAL=10` (a whole-task budget — `bimanual_dual_push_buttons` has 46
variations, so a per-variation cap of 10 would mean 460 rollouts).

---

## 3. PINNED FILES — never regenerate

> **`instructions/orbital_miscalibration_noise.json` and
> `instructions/orbital_miscalibration_noise_ood.json` MUST NOT be regenerated.**

These define the *fixed per-camera-group* miscalibration that R1a/R1b/R1c trained
under (`_noise.json`, seed 42) and the held-out base R2b evaluated against
(`_noise_ood.json`, seed 3187). They are **weights-bearing**: the checkpoints in this
bundle have absorbed the specific perturbation in `_noise.json`, and R2b's entire
finding — that ~20 pts of R1a's apparent robustness was fit to one specific base — is
only meaningful because `_noise_ood.json` is a *different direction at matched
magnitude* (82 deg mean axis separation, magnitudes within 1%).

Re-running `scripts/generate_orbital_miscal_noise.py` would silently produce
different rotations and invalidate every R2/R2b/R2c number, with no error raised.
The seed-3187 file in particular was chosen by searching **4000 candidates** for the
closest aggregate-magnitude match to the training file; that search is not cheaply
reproducible.

| file | seed | mean angle | mean \|t\| | role |
|---|---|---|---|---|
| `orbital_miscalibration_noise.json` | 42 | 5.95 deg | 5.26 cm | trained base (R1a/b/c) |
| `orbital_miscalibration_noise_ood.json` | 3187 | 5.90 deg | 5.26 cm | held-out base (R2b) |

Both list **three** cameras, so at `ncam=4` the fourth (`wrist_right`) is
**identity-padded** — a training quirk the eval harness reproduces deliberately.
`instructions/random_miscal_noise_bimanual.json` (the per-level random directions) is
pinned for the same reason: a level is *one fixed noise direction*, which is why
individual cells go non-monotonic and only level means are readable.

---

## 4. Per-experiment reproduction map

Training runs all used `workdir: .` — the submitting checkout is uploaded to
`~/sky_workdir` and re-uploaded byte-identical on preemption recovery, so a restart
resumes the same code. Each job log carries a provenance assertion.

### 4.1 PerAct2 training runs → `docs/status/experiments.md` "Training runs" table

| run | job | sky YAML | key config | ckpt (bundle member) | wandb |
|---|---|---|---|---|---|
| `peract2_base_b200` | 112602 | `scripts/sky/peract2_base_b200.yaml` | siglip2, nhist=1, bs256, lr3e-4, 350k, B200:4 | `checkpoints/peract2_base_b200_{best,last}.pth` | `7vjpod6m` |
| `peract2_base_nhist3_b200` | 117608 | `scripts/sky/peract2_base_nhist3_b200.yaml` | = above + `num_history=3`, B200:1 | `checkpoints/peract2_base_nhist3_b200_{best,last}.pth` | `bk6j5v66` |
| `peract2_base_nhist3_clip_b200` | 118960 | `scripts/sky/peract2_base_nhist3_clip_b200.yaml` | = above + `backbone=clip`, B200:1 | `checkpoints/peract2_base_nhist3_clip_b200_{best,last}.pth` | `iqa4wuqb` |
| `upstream_peract2_repro` | 117879 | `scripts/sky/upstream_peract2_repro.yaml` | upstream code @ `ab70932`, their recipe verbatim: CLIP RN50 frozen, bs64, lr1e-4, nhist=3, 350k | `checkpoints/upstream_peract2_repro_{best,last}.pth` | `rr1qjj1l` |

Shared: `bimanual=true`, `dataset=Peract2_3dfront_3dwrist`, `batch_size_val=64`,
`NUM_WORKERS=16`, `lr=backbone_lr`, `BASE_LOG_DIR=/k8s-nfs/harsvbha/3dfa/train_logs`.

Recipe rationale (batch-size probe, 11 Aug): the pipeline is **dataloader-bound**,
saturating ~950-980 samples/s/GPU from bs128 up, never OOMing through bs1024
(41/183 GB). `num_workers` is the dominant lever (8→16 ≈ 2x at bs128). bs256 global
at lr 3e-4 is a capped 3x linear scaling — a deliberate change from the paper's bs32.

```bash
sky jobs launch -y -d -n hb-3dfa-peract2-base-nhist3-b200 \
  --infra k8s/ll-sea --env PREEMPTIBLE=1 scripts/sky/peract2_base_nhist3_b200.yaml
```

### 4.2 PerAct2 online eval → "Online eval — PerAct2 13 tasks" table

**Read the sampler bug note before trusting any pre-14-Aug number.** All pre-fix
fork evals ran a train/eval sampler mismatch: `image_space_sampling: true` in config
was never forwarded by the trainer (models trained with density FPS) but *was*
overlaid from the checkpoint config at eval (rollouts ran uniform sampling). Offline
val could not see it — it uses the trainer's model. Cost ~9 pts (siglip2) to ~31 pts
(clip). Fix chain:

| commit | what |
|---|---|
| `eb06b4c` | trainer forwards `image_space_sampling` / `skip_fps` into the model |
| `39efdae` | eval takes an explicit `ISS` env — no silent inheritance |
| `b933802` | config default flipped to `false` (density FPS = what was trained) |
| `5777b27` | single shared model-kwargs construction path + step-0 guardrail (this class of bug now crashes rather than degrading) |

The `_issfix` columns are the corrected numbers. Evaluate any checkpoint from this
campaign with **`ISS=false`**.

YAML: `scripts/sky/peract2_online_eval.yaml` (`scripts/sky/peract2_upstream_eval.yaml`
for the upstream code path + patches in `scripts/sky/patches/`).

```bash
sky jobs launch -y -d -n hb-3dfa-eval-<run>-<task> \
  --infra k8s/sky-us-east-1 --env PREEMPTIBLE=1 \
  --env TASK=bimanual_push_box \
  --env RUN_NAME=peract2_base_nhist3_b200 \
  --env CKPT_S3=s3://far-research-internal/harsvbha/3dfa/eval/ckpt/peract2_base_nhist3_b200.pth \
  --env OUT_S3=s3://far-research-internal/harsvbha/3dfa/eval/results/peract2_base_nhist3_b200_issfix \
  --env NUM_DEMOS=25 --env ISS=false --env SAVE_VIDEO=true \
  scripts/sky/peract2_online_eval.yaml
```

Campaigns: base 11 Aug (116805-116826), released ckpt 11-12 Aug (117323-117442),
nhist3 12 Aug (118915-118931), clip + repro 13 Aug (120514-120549), issfix
falsification + re-eval 13-14 Aug (120580-120619).

### 4.3 Orbital training + camera generalization → "Orbital PerAct2 — training + camera-generalization eval"

| run | job | sky YAML | ckpt (bundle member) | wandb |
|---|---|---|---|---|
| `peract2_orbital_nhist3_b200` | 120769 | `scripts/sky/peract2_orbital_b200.yaml` (`dataset=OrbitalPeract2`, `num_history=3`, 100k) | `checkpoints/peract2_orbital_nhist3_b200_{best,last}.pth` | `pnvpafcg` |

B200:4, ~5h20m, 2 recoveries, full 100k verified in the checkpoint. Unlike standard
PerAct2, the orbital zarr **has `demo_id`**, so `nhist=3` here enables the full
visual-history path (3 stacked rgb/depth frames), not just proprio.

Eval needed a new harness — no existing code path did bimanual + orbital spawning
(`e042de3`, `online_evaluation_rlbench/utils_with_orbital_bimanual_rlbench.py`; also
introduced `num_demos_total` as a whole-task budget).

```bash
sky jobs launch -y -d -n hb-3dfa-orb-eval-<task> \
  --infra k8s/sky-us-east-1 --env PREEMPTIBLE=1 \
  --env TASK=bimanual_push_box --env SPAWN_GROUP=<group> \
  --env CKPT_S3=s3://far-research-internal/harsvbha/3dfa/eval/ckpt/peract2_orbital_nhist3_b200.pth \
  --env OUT_S3=s3://far-research-internal/harsvbha/3dfa/eval/results/peract2_orbital_nhist3_b200/indomain \
  --env NUM_DEMOS_TOTAL=10 --env ISS=false \
  scripts/sky/peract2_orbital_online_eval.yaml
```

in-domain = first train group, OOD = held-out `eval_group`, both from
`instructions/peract2_orbital_task_group_mapping.json`.

### 4.4 Clean-trained noise sweep → "camera-miscalibration noise sweep, OOD camera"

Same checkpoint (`peract2_orbital_nhist3_b200` @ iter 100000,
`predict_extrinsics=false`), same OOD condition as 4.3, plus paired rot+trans noise
at eval time. `d586f71` extended the sweep machinery to the 4-camera bimanual harness
(the grogu-era version was single-arm only). Noise perturbs **only the extrinsics fed
to depth→PCD** — RGB and depth untouched — so the model sees a corrupted 3D scene.

The `0` column **is** the 4.3 OOD column, not a rerun: with no levels set the harness
leaves extrinsics untouched, confirmed by a 5deg+5cm smoke test.

```bash
sky jobs launch -y -d -n hb-3dfa-orbnoise-5deg5cm-<task> \
  --infra k8s/sky-us-east-1 --env PREEMPTIBLE=1 \
  --env TASK=<task> --env SPAWN_GROUP=<eval_group> \
  --env MISCAL_ROT=5deg --env MISCAL_TRANS=5cm \
  --env OUT_S3=.../peract2_orbital_nhist3_b200/noise_5deg5cm \
  scripts/sky/peract2_orbital_online_eval.yaml
```

52 jobs, L40S:1. **Prefer `sky-us-east-1` for L40S waves** — east-2 had no L40S
capacity for the duration.

### 4.5 R1a / R1b / R1c — the deltaM ladder → sections "R2", "R2b", "R2c"

Three checkpoints trained on the orbital zarr with **persistently miscalibrated**
extrinsics (`miscal=orbital_fixed_medium_randnoise`: fixed per-group medium base from
the pinned `orbital_miscalibration_noise.json`, plus `<=3deg`/`<=1cm` random top-up
resampled per batch):

```
T_applied = T_random @ T_base[camera_group]
```

| arm | job | sky YAML | the delta | ckpt (bundle member) | wandb |
|---|---|---|---|---|---|
| **R1a** baseline | 126269 | `scripts/sky/peract2_orbital_miscal_base_h200.yaml` | `experiment=default` (`predict_extrinsics=false`) | `checkpoints/orbital_nhist3_miscal_base_{best,last}.pth` | `aq54hwdi` |
| **R1b** deltaM | 126270 | `scripts/sky/peract2_orbital_miscal_deltam_h200.yaml` | `experiment=camtoken_deltaM` (`predict_extrinsics=true`, `extrinsics_prediction_mode=delta_m`, `dynamic_rope_from_camtoken=true`) | `checkpoints/orbital_nhist3_miscal_deltaM_{best,last}.pth` | `2ks5zjmt` |
| **R1c** deltaM + EE-aux | 127191 | `scripts/sky/peract2_orbital_miscal_deltam_eeaux_h200.yaml` | = R1b + `predict_ee_aux=true`, `lambda_aux=1.0`, `ee_aux_cam_ids=[0,1]` | `checkpoints/orbital_nhist3_miscal_deltaM_eeaux_{best,last}.pth` | `9w9w8xwy` |

H200:4 on `k8s/ll-lax`, `BATCH_SIZE=256`, `LR=3e-4`, `NUM_WORKERS=16`,
`TRAIN_ITERS=100000`, `NUM_HISTORY=3`, `bimanual=true`, `dataset=OrbitalPeract2`,
`image_space_sampling=false`, `backbone=siglip2`. R1a and R1b YAMLs are
**byte-identical except one line** — that is the point of the pair. All three verified
at `iter=100000` before staging, and the flags were confirmed to *arrive in each
rollout process* by reading the `loaded_from_ckpt` dump, not assumed.

**Eval-side code the ladder required** (`b9e05b0`): the bimanual orbital harness
originally supported only the *random* half of the recipe, so the training condition
could not be reproduced at eval. `orbital_miscal_noise_level` is now threaded through
the bimanual branch of `evaluate_policy.py`, composed in training's order. It is in
`_EVAL_RUNTIME_KEYS`, so a checkpoint's saved `medium` is **never silently
inherited** — `ORBITAL_MISCAL_LEVEL` must name it. `17a440e` adds
`orbital_miscal_noise_file` (env `MISCAL_FILE`), also runtime-keyed.

> **`level 0` means fixed base only, NOT clean.** The `clean0` condition is the
> no-base condition. These are different columns and conflating them misreads every
> R2 table.

```bash
# a trained-miscal noise cell (R2)
--env ORBITAL_MISCAL_LEVEL=medium --env MISCAL_ROT=5deg --env MISCAL_TRANS=5cm
# level 0 — fixed base, no random noise
--env ORBITAL_MISCAL_LEVEL=medium
# clean0 — no base, no random noise
(omit all three)
# held-out fixed base (R2b)
--env ORBITAL_MISCAL_LEVEL=medium \
--env MISCAL_FILE=instructions/orbital_miscalibration_noise_ood.json
```

all on `scripts/sky/peract2_orbital_online_eval.yaml`, L40S:1, `sky-us-east-1`,
10 rollouts/task, per-task `eval_group`.

Job counts: R2 **156** (`hb-3dfa-r2-{base,dm}-<level>-<task>`), R2b **26**
(`hb-3dfa-oodmiscal-{base,dm}-<task>`), R2c **91** (`hb-3dfa-r1c-<cond>-<task>`).

**Fan-out and collection: `scripts/eval/reconcile_r1c.sh` (idempotent — diffs the
grid against the sky queue and S3, resubmits only missing cells) and
`scripts/eval/collect_r1c.py`.** Use these rather than a bare submit loop:
**parallel submission does not work.** Three concurrent `sky jobs launch` loops
overwhelmed the `RestfulAdminPolicy` sidecar (`admin-policy:80` connection refused /
read timeout) and **silently dropped 66 of 91 submissions**. Submit serially with
retry, then reconcile against S3. R2 lost 10 cells the same way, detected by diffing
S3 against the task list.

Each wave was gated on a smoke test before fan-out (R2 base/`push_box`/level-0 →
0.70; R2b → 0.70; R2c → 1.00), with the log checked for the right level, group, and
miscal file. A near-zero smoke would have meant the base was applied wrong.

### 4.6 grogu-era single-arm results (May 2026)

The tables above the "August 2026 Campaign" heading in `experiments.md` predate the
FAR migration (repo moved 11 Aug 2026). They ran on grogu/delta/babel with slurm
scripts under `scripts/eval/*.slurm` and data at `/grogu/user/harshilb/low_dim_demos`
and `/grogu/datasets/hbhatia/full_rollouts`. **Those checkpoints and datasets are not
in this bundle** and the slurm infrastructure no longer exists. They are retained as
context for the bimanual results (e.g. grogu's `deltaM_EEF` holding 0.77 at
15deg+15cm on `turn_tap` is the bar R2 was written against), not as reproducible
runs.

---

## 5. Checkpoint inventory

All 8 runs, `best.pth` (lowest offline val) + `last.pth` (final iter). `best` came
from cluster NFS; `last` from the S3-staged copies used by eval.

| run | bundle member | size | S3 path (`last`) |
|---|---|---|---|
| `peract2_base_b200` | `checkpoints/peract2_base_b200_best.pth` | 33 MB | — (NFS only) |
| | `checkpoints/peract2_base_b200_last.pth` | 44 MB | `s3://far-research-internal/harsvbha/3dfa/eval/ckpt/peract2_base_b200_last.pth` |
| `peract2_base_nhist3_b200` | `checkpoints/peract2_base_nhist3_b200_best.pth` | 34 MB | — |
| | `checkpoints/peract2_base_nhist3_b200_last.pth` | 45 MB | `.../ckpt/peract2_base_nhist3_b200.pth` |
| `peract2_base_nhist3_clip_b200` | `checkpoints/peract2_base_nhist3_clip_b200_best.pth` | 42 MB | — |
| | `checkpoints/peract2_base_nhist3_clip_b200_last.pth` | 58 MB | `.../ckpt/peract2_base_nhist3_clip_b200.pth` |
| `upstream_peract2_repro` | `checkpoints/upstream_peract2_repro_best.pth` | 422 MB | — |
| | `checkpoints/upstream_peract2_repro_last.pth` | 402 MB | `.../ckpt/upstream_peract2_repro.pth` |
| `peract2_orbital_nhist3_b200` | `checkpoints/peract2_orbital_nhist3_b200_best.pth` | 34 MB | — |
| | `checkpoints/peract2_orbital_nhist3_b200_last.pth` | 45 MB | `.../ckpt/peract2_orbital_nhist3_b200.pth` |
| `orbital_nhist3_miscal_base` (R1a) | `checkpoints/orbital_nhist3_miscal_base_best.pth` | 34 MB | — |
| | `checkpoints/orbital_nhist3_miscal_base_last.pth` | 45 MB | `.../ckpt/orbital_miscal_base.pth` |
| `orbital_nhist3_miscal_deltaM` (R1b) | `checkpoints/orbital_nhist3_miscal_deltaM_best.pth` | 34 MB | — |
| | `checkpoints/orbital_nhist3_miscal_deltaM_last.pth` | 46 MB | `.../ckpt/orbital_miscal_deltam.pth` |
| `orbital_nhist3_miscal_deltaM_eeaux` (R1c) | `checkpoints/orbital_nhist3_miscal_deltaM_eeaux_best.pth` | 34 MB | — |
| | `checkpoints/orbital_nhist3_miscal_deltaM_eeaux_last.pth` | 46 MB | `.../ckpt/orbital_miscal_deltam_eeaux.pth` |

Notes:

* **`upstream_peract2_repro_last.pth` is the optimizer-stripped 402 MB S3 copy**, not
  the 448 MB NFS `last.pth` (which carries optimizer state). Weights are identical;
  the bundle takes the smaller one. Both upstream files are ~10x the others because
  the upstream recipe saves the frozen CLIP RN50 backbone.
* `interm_step_*.pth` (every 10k, 22-23 MB each) are **excluded** — they remain on
  cluster NFS at `{ll-sea,ll-lax}:/k8s-nfs/harsvbha/3dfa/train_logs/exp/<run>/`.
* NFS source: `peract2_*` and `upstream_*` on **ll-sea**;
  `orbital_nhist3_miscal_*` on **ll-lax**.
* `peract2_orbital_b200` (job 117772) was cancelled at ~13k iters pending the nhist
  decision and is not included.

### The released-checkpoint column

The `released ckpt` column in the eval table is the upstream authors' public
checkpoint, staged at
`s3://far-research-internal/harsvbha/3dfa/eval/ckpt/upstream_3dfa_peract2.pth`
(402 MB). **Not in the bundle** — it is not ours to redistribute; fetch it from the
upstream release.

---

## 6. wandb

Project **`far-wandb/3dfa`** at `https://far.wandb.io`.

| archive name | run ID | history rows | link |
|---|---|---|---|
| `peract2_base_b200` | `7vjpod6m` | 7000 | [run](https://far.wandb.io/far-wandb/3dfa/runs/7vjpod6m) |
| `peract2_base_nhist3_b200` | `bk6j5v66` | 7000 | [run](https://far.wandb.io/far-wandb/3dfa/runs/bk6j5v66) |
| `peract2_base_nhist3_clip_b200` | `iqa4wuqb` | 7000 | [run](https://far.wandb.io/far-wandb/3dfa/runs/iqa4wuqb) |
| `upstream_peract2_repro` | `rr1qjj1l` | 7000 | [run](https://far.wandb.io/far-wandb/3dfa/runs/rr1qjj1l) |
| `peract2_orbital_nhist3_b200` | `pnvpafcg` | 268 | [run](https://far.wandb.io/far-wandb/3dfa/runs/pnvpafcg) |
| `orbital_miscal_base` (R1a) | `aq54hwdi` | 2000 | [run](https://far.wandb.io/far-wandb/3dfa/runs/aq54hwdi) |
| `orbital_miscal_deltaM` (R1b) | `2ks5zjmt` | 2000 | [run](https://far.wandb.io/far-wandb/3dfa/runs/2ks5zjmt) |
| `orbital_miscal_deltaM_eeaux` (R1c) | `9w9w8xwy` | 2000 | [run](https://far.wandb.io/far-wandb/3dfa/runs/9w9w8xwy) |

Offline export (in the bundle at `wandb/<name>__<id>/`, produced by
`scripts/export_wandb_runs.py`):

```bash
WANDB_BASE_URL=https://far.wandb.io uv run python scripts/export_wandb_runs.py --out ./wandb_export
```

Per run: `config.json`, `summary.json`, `meta.json`, `history.csv` (+
`history.parquet` when `pandas`/`pyarrow` are present — not in the locked env, so the
committed export is CSV). History uses `scan_history()`, which streams **every**
logged step; `run.history()` downsamples and must not be used for archival. Auth
comes from `WANDB_API_KEY` or `~/.netrc`.

R1c's `train/ee_aux_loss` trace is the pre-registered diagnostic for R2c and is in
`orbital_miscal_deltaM_eeaux__9w9w8xwy/history.csv`: 0.0582 (step 49) → 0.0027
(100k), min 0.00235 @ 96549. It fell ~10x past the early 0.028 plateau, so R2c's null
result is a verdict on the mechanism, not a failed optimization.

---

## 7. Eval results

Bundle member `results/` — **887 JSON + log files, 14 MB**, synced from:

```
s3://far-research-internal/harsvbha/3dfa/eval/results/
```

Layout: `<run_name>/[<condition>/]bimanual_<task>.json` plus `logs/`. Conditions:
`indomain`, `ood`, `clean0`, `noise_0`, `level0`, `noise_{2deg2cm,5deg5cm,10deg10cm,15deg15cm}`,
`ood_miscal`.

### Deliberately left on S3: rollout videos

**Videos are NOT in the bundle** — they are ~10.4 GB of the 10.4 GB S3 results tree
(the JSONs are 9.7 MB of it) and are regenerable with `SAVE_VIDEO=true`. They live in
`videos/` subdirectories under each results prefix:

```
s3://far-research-internal/harsvbha/3dfa/eval/results/peract2_orbital_nhist3_b200/{indomain,ood}/videos/
s3://far-research-internal/harsvbha/3dfa/eval/results/peract2_orbital_nhist3_b200/noise_{2deg2cm,5deg5cm,10deg10cm,15deg15cm}/videos/
s3://far-research-internal/harsvbha/3dfa/eval/results/{orbital_miscal_base,orbital_miscal_deltam}/{clean0,noise_0,noise_2deg2cm,noise_5deg5cm,noise_10deg10cm,noise_15deg15cm,ood_miscal}/videos/
s3://far-research-internal/harsvbha/3dfa/eval/results/orbital_miscal_deltam_eeaux/{level0,noise_2deg2cm,noise_5deg5cm,noise_10deg10cm,15deg15cm,ood_miscal,clean0}/videos/
s3://far-research-internal/harsvbha/3dfa/eval/results/peract2_base_*/videos/
```

They are worth keeping: a rollout video is the only way to separate a *policy*
failure from a *harness* failure (wrong camera, wrong scene) when a success rate
comes back at zero. Fetch with:

```bash
AWS_PROFILE=far-compute aws s3 sync \
  s3://far-research-internal/harsvbha/3dfa/eval/results/<run>/<condition>/videos/ ./videos/
```

---

## 8. Egress bundle layout

```
3dfa_egress_<date>/
├── MANIFEST.txt                    # members, sizes, sha256 per part, git sha, S3 pointers
├── checkpoints/                    # 16 files (8 runs x best+last), 1.4 GB
├── datasets/orbital_peract2/
│   ├── zarr/                       # 6.0 GB — final 1053 train / 117 val
│   └── shards/                     # 12 GB — all 78 raw shards (superset)
├── instructions/                   # pinned miscal JSONs, camera + task-group mappings
├── results/                        # 887 eval JSONs + logs, 14 MB (no videos)
└── wandb/                          # 8 runs x {config,summary,meta,history.csv} + INDEX.json
```

Split into `<=8 GB` parts for transfer resilience; reassemble with

```bash
cat 3dfa_egress_<date>.tar.zst.part-* | zstd -d | tar -xf -
```

Verify part checksums against `MANIFEST.txt` (`sha256sum -c`) before extracting.

---

## 9. Known-live issues

* **`RTExtrinsicsPredictor` signature bug** — `modeling/policy/base_denoise_actor.py`
  around lines 903-905 passes `fps_*` kwargs the class cannot accept. Still a 1-line
  fix. The `rt` extrinsics mode is unused (cut with `delta_m_full`), so this is
  latent, not hit by any run above.
* **Noise floor.** Orbital evals are 10 rollouts/task, so per-cell resolution is
  about `+/-0.15`. **Read 13-task level means, not individual cells.** Several
  non-monotonic cells are genuine runs verified from their logs (R1c
  `sweep_to_dustpan` 0.0 at level 0 but 1.0 at 5deg is the loudest) — a level is one
  fixed noise *direction*, so a task can tolerate one axis and not another.
* **A single fixed miscal base is partly memorized** (R2b: R1a lost 20.8 pts to a
  matched-magnitude held-out base). Future orbital training should use **resampled**
  fixed bases, not one pinned base. That is the cheapest outstanding improvement.
