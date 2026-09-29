# Where deltaM Actually Wins — Mining R1a/R1b/R1c (Aug 2026)

Question posed: find the regimes, conditions, and metrics where the deltaM arms
are **much** better than the no-deltaM baseline, so the narrative and the next
launches can aim at them. Three matched checkpoints, step 100k, identical
except the deltaM flags:

| arm | flags |
|---|---|
| R1a | `predict_extrinsics=false` |
| R1b | `+delta_m`, `dynamic_rope_from_camtoken` |
| R1c | R1b `+predict_ee_aux=true, lambda_aux=1.0, ee_aux_cam_ids=[0,1]` |

All three trained on orbital PerAct2 under persistent miscalibration
(`orbital_fixed_medium_randnoise`: fixed per-group base from
`instructions/orbital_miscalibration_noise.json`, plus <=3deg / <=1cm random
top-up per batch).

Two independent bodies of evidence:

- **Tier 1** — the 273 existing closed-loop eval cells (13 tasks x 7 conditions
  x 3 arms, 10 rollouts/cell). A single cell resolves ~+-0.15, so nothing here
  is read cell-by-cell; every claim is a task-level paired statistic (n=13
  tasks as the cluster-robust unit, exact sign-flip permutation p, bootstrap CI
  over tasks).
- **Tier 2** — open-loop keypose error on the orbital val zarr (853 keypose
  samples over 117 episodes) under a controlled 7-point extrinsics-corruption
  grid, plus a direct readout of the predicted delta_M. Script:
  `scripts/eval/offline_deltam_analysis.py`.
- **Tier 3** — the asymmetric per-camera miscalibration test (section 6), run
  after the above to check the mechanism Tier 2 proposed. Scripts:
  `scripts/eval/offline_asym_miscal_analysis.py`,
  `scripts/eval/analyze_asym_miscal.py`.

**The headline is a split verdict, and the split is the finding.** deltaM alone
(R1b) does not survive scrutiny: its one strong Tier-1 signal does not
reproduce in the higher-powered offline measurement. deltaM **+ the EE-aux
loss** (R1c) does reproduce, on both bodies of evidence, at p<0.01. And the
delta_M matrix itself turns out to be very nearly a learned constant, which
reframes the mechanism entirely.

---

## 1. Bottom line

1. **R1c is the only arm with a real, replicated advantage.** High-corruption
   keypose accuracy (`pos_acc_001` at n10+n15): **+0.053 CI[+0.030,+0.079],
   p=0.0005, better on 12/13 tasks.** Under the held-out OOD fixed base:
   **+0.099 CI[+0.053,+0.146], p=0.0029.** Its error also grows more slowly
   with corruption: **+0.63 mm/deg flatter than R1a, CI[+0.28,+0.98],
   p=0.0068, 10/13 tasks.**
2. **R1b's Tier-1 slope win does not replicate offline.** Closed-loop SR slope
   said R1b was flatter (+0.0134 SR/deg, p=0.0034, 11/13). The 853-sample
   offline position-error slope says the opposite and nulls out: **-0.089
   mm/deg, CI[-0.60,+0.41], p=0.74, 7/13.** With ~65x the samples, this is the
   measurement to believe. Treat the closed-loop R1b slope as an artifact of
   binary thresholding on 10 rollouts.
3. **The invariance effect is real but is not accuracy.** Level-0 -> OOD-base SR
   drop: R1a -0.208, R1b -0.062, R1c -0.054; diff-in-diff vs R1a R1b +0.146
   CI[+0.023,+0.277], R1c +0.154 CI[+0.031,+0.285]. Offline confirms the
   *ordering* on retained fraction of own-clean accuracy (0.816 / 0.870 /
   0.905) while showing all three take a large absolute hit. deltaM buys
   insensitivity to *which* calibration error, not accuracy under calibration
   error — the prior R2b reading, now with a proper paired test behind it.
4. **The smoking gun fires — but only for R1c, and only on the aux cameras.**
   `||delta_M - I||_F` is essentially a constant. Sweeping injected corruption
   from 0 to ~10deg moves it by **0.03%** of its own sample-level SD in R1b
   (0.652822 -> 0.652635, and *downward*). In R1c it moves the right way and
   ~10x more, and the movement is confined to **cameras 0 and 1 — exactly
   `ee_aux_cam_ids=[0,1]`.** Cameras 2 and 3 stay flat. The aux loss is what
   makes the extrinsics token carry calibration information at all; without it
   the token is a learned reparameterization, not a sensor.
   ~~The aux loss makes the token sense *its own* camera's miscalibration.~~
   **Retracted by section 6.** A per-camera targeted probe shows the response is
   *column*-structured, not diagonal: cams 0/1's tokens move when **any** camera
   is corrupted — including cam3, and by more than when their own camera is the
   corrupted one — and the largest response anywhere is 0.75% of the token's own
   SD. The aux loss changes *which tokens are live*, not what they sense. The
   "delta_M is nearly a learned constant" half of this point stands and is
   reinforced.
5. **The cost is real too, and it is at low corruption.** Closed-loop SR at n2:
   R1b -0.154 CI[-0.238,-0.062] p=0.0137; R1c -0.223 CI[-0.408,-0.054]
   p=0.0381. Integrated over 2-15deg the three arms are close (SR AUC 0.337 /
   0.329 / 0.379). The "trade, not Pareto win" framing survives — what is new
   is that both the win at high corruption and the loss at low corruption clear
   the noise floor.

**Narrative implication.** The story is not "learned extrinsics correct
miscalibration." It is: *a learned RoPE-basis transform plus an EE-supervised
auxiliary target buys robustness to unseen calibration error, and the auxiliary
supervision is a necessary ingredient — it is what turns the extrinsics token
from a constant into something that responds to the perturbation.* R1b alone is
not a result. Every Tier-3 proposal below is therefore an R1c-vs-R1a test with
R1b carried only as the ablation.

**Narrative implication, revised after section 6.** The "responds to the
perturbation" clause is too strong — the response exists but is global, tiny, and
sign-inconsistent. The claim that survives all three tiers is narrower and
mechanism-agnostic: *the EE-aux loss, not the deltaM parameterization, is what
produces robustness to unseen calibration error, and it appears to do so by
reducing reliance on any single camera rather than by estimating calibration.*
R1c wins in a fourth independent condition; R1b still wins nowhere.

---

## 2. Tier 1 — mining the 273 closed-loop cells

Data: `s3://far-research-internal/harsvbha/3dfa/eval/results/orbital_miscal_{base,deltam,deltam_eeaux}/`.
Note the condition-dir naming differs across arms (base/deltam use `noise_0`,
eeaux uses `level0`). Unit of analysis is the task (n=13); p-values are exact
sign-flip permutation over the 13 paired task differences; CIs are 20k-resample
bootstraps over tasks.

### 2a. Degradation slope (SR vs corruption deg, levels 0/2/5/10/15)

| arm | aggregate slope (SR/deg) | 50%-of-own-L0 retention crossing |
|---|---|---|
| R1a | -0.0400 | 8.7 deg |
| R1b | -0.0266 | 10.3 deg |
| R1c | -0.0149 | never within 15 deg |

Paired flatness vs R1a: R1b **+0.0134** CI[+0.0066,+0.0203] p=0.0034 (11/13
flatter); R1c **+0.0251** CI[+0.0107,+0.0449] p=0.0015 (11/13 flatter).

This was the strongest Tier-1 effect. **Tier 2 sustains it for R1c and refutes
it for R1b** (section 3c) — see the honesty note in section 5.

### 2b. Invariance to *which* calibration error (L0 -> OOD base)

Magnitude held constant to within 1%; only the rotation directions change
(seed 3187 vs 42, axes ~82deg apart). SR drop: R1a **-0.208**, R1b **-0.062**,
R1c **-0.054**. Diff-in-diff vs R1a: R1b **+0.146** CI[+0.023,+0.277]; R1c
**+0.154** CI[+0.031,+0.285]. Both CIs exclude zero. This is the cleanest
Tier-1 effect because it holds corruption magnitude fixed and varies only its
direction — nothing about task difficulty or accuracy tax can produce it.

### 2c. High-corruption absolute wins (R1c only)

| comparison | R1c - R1a | CI | p |
|---|---|---|---|
| n15 | **+0.185** | [+0.054,+0.354] | 0.0234 |
| n10+n15 pooled | **+0.119** | [+0.027,+0.215] | 0.0449 |

Cell-level sign test over the high-corruption cells: 18 wins / 7 losses,
p=0.043. Pooled binomial over n10+n15 (260 rollouts each): R1a 43/260=0.165,
R1b 52/260=0.200, R1c **74/260=0.285**.

### 2d. Where it concentrates (task families)

Coordination-gross family (`push_box`, `lift_ball`, `lift_tray`,
`sweep_to_dustpan`, `straighten_rope`), high-corruption SR gain vs R1a:
R1b **+0.100** CI[+0.040,+0.173]; R1c **+0.233** CI[+0.100,+0.353]. The
articulated family (drawer / fridge / oven) is **negative** for both deltaM
arms.

Tier 2 partly disagrees on the family split: on offline `pos_acc_001` at
n10+n15, R1c gains on *all three* families (coord +0.052, precision +0.043,
articulated +0.070). With n=3-5 tasks per family in both analyses, the family
decomposition is the weakest claim in this document. Do not build a story on
it.

### 2e. The counterweight — the losses are also real

R1b-R1a @ n2 **-0.154** CI[-0.238,-0.062] p=0.0137; R1c-R1a @ n2 **-0.223**
CI[-0.408,-0.054] p=0.0381. SR AUC over 2-15deg: R1a 0.337, R1b 0.329, R1c
0.379.

---

## 3. Tier 2 — offline geometric error on the val zarr

`scripts/eval/offline_deltam_analysis.py`, 853 samples per cell, 7 conditions x
3 checkpoints, ~20s/condition on one H200. Open-loop keypose prediction on
ground-truth observations (the `run_inference=True` path the trainer's
validation uses) — **not** closed-loop, so this measures the geometric quality
of the prediction, isolated from rollout compounding.

Conditions: `clean` (none), `base` (trained fixed per-group base, no top-up),
`ood_base` (held-out seed-3187 base, magnitude-matched), `n2/n5/n10/n15`
(trained base + random top-up of that magnitude, composed `T_rand @ T_base` as
in training). Every condition is reseeded identically so all three arms see the
same noise draws.

Note: `orbital_miscal_noise_file` was honored only by the online bimanual
harness; `utils/data_preprocessors/rlbench.py` now threads it through, which is
what makes the offline `ood_base` cell reachable.

### 3a. Aggregate (ALL tasks)

`pos_l2_mean` (m, lower better):

| condition | R1a | R1b | R1c |
|---|---|---|---|
| clean | 0.0145 | 0.0160 | **0.0139** |
| base | 0.0142 | 0.0148 | **0.0138** |
| ood_base | **0.0199** | 0.0220 | 0.0209 |
| n2 | **0.0145** | 0.0153 | 0.0143 |
| n5 | **0.0160** | 0.0179 | 0.0161 |
| n10 | 0.0245 | 0.0270 | **0.0227** |
| n15 | 0.0364 | 0.0377 | **0.0333** |

`pos_acc_001` (fraction of keyposes within 1 cm — the discriminating metric):

| condition | R1a | R1b | R1c |
|---|---|---|---|
| clean | 0.620 | 0.590 | **0.666** |
| base | 0.641 | 0.621 | **0.676** |
| ood_base | 0.505 | 0.513 | **0.603** |
| n2 | 0.627 | 0.610 | **0.665** |
| n5 | 0.568 | 0.555 | **0.622** |
| n10 | 0.432 | 0.407 | **0.487** |
| n15 | 0.340 | 0.332 | **0.385** |

`pos_acc_005` (5 cm) is near-saturated at low corruption and only separates at
n10/n15 (R1a 0.913/0.817, R1b 0.901/0.805, R1c **0.922/0.842**). Gripper
accuracy is ~0.97-0.99 everywhere and never separates the arms — drop it as a
diagnostic.

**Answering the posed question (i) — does deltaM reduce position error where SR
does not move?** For R1c, yes, and it does more: it reduces error even in cells
where closed-loop SR *lost* (clean, base, n2 offline errors are all lowest for
R1c, while closed-loop SR at n2 was significantly *worse*). That divergence is
itself informative: R1c's open-loop keyposes are more accurate at low corruption
but its closed-loop rollouts are worse, which points at a rollout-stability or
compounding problem rather than a per-keypose accuracy problem. For R1b the
answer is no — its offline error is worse than R1a's in 6 of 7 conditions.

### 3b. Per-task paired tests (n=13, same machinery as Tier 1)

`pos_acc_001`, positive = deltaM better:

| conditions | R1b - R1a | R1c - R1a |
|---|---|---|
| n10+n15 | -0.015 CI[-0.036,+0.005] p=0.18 (5/13) | **+0.053 CI[+0.030,+0.079] p=0.0005 (12/13)** |
| n15 | -0.003 CI[-0.029,+0.023] p=0.82 (6/13) | **+0.045 CI[+0.020,+0.075] p=0.0022 (11/13)** |
| ood_base | +0.010 CI[-0.023,+0.041] p=0.58 (6/13) | **+0.099 CI[+0.053,+0.146] p=0.0029 (10/13)** |
| clean | -0.026 CI[-0.056,+0.002] p=0.12 (5/13) | +0.031 CI[-0.017,+0.079] p=0.25 (9/13) |
| base | -0.024 CI[-0.071,+0.026] p=0.38 (4/13) | +0.010 CI[-0.043,+0.064] p=0.74 (8/13) |

`pos_l2_mean` tells the same story: R1c n10+n15 **+0.0028 m CI[+0.0011,+0.0044]
p=0.0107 (10/13)**; R1b nowhere significant.

R1c's advantage is **absent at clean/base and present at n10/n15/ood_base**.
That is the correct shape for a robustness mechanism, and it is the single most
convincing pattern in this analysis: the effect appears precisely where the
hypothesis says it should and vanishes where it says it should.

### 3c. Error-growth slope — question (ii)

`pos_l2_mean` regressed on injected angle over n2..n15:

| arm | slope | n15/clean ratio |
|---|---|---|
| R1a | +3.442 mm/deg | 2.50 |
| R1b | +3.521 mm/deg | 2.35 |
| R1c | **+2.943 mm/deg** | 2.39 |

Paired flatness vs R1a: R1b **-0.089 mm/deg CI[-0.601,+0.406] p=0.74 (7/13)** —
null, and the point estimate has the wrong sign. R1c **+0.633 mm/deg
CI[+0.280,+0.979] p=0.0068 (10/13)** — real.

Per-task slopes (mm/deg), R1a / R1b / R1c: `lift_tray` 6.57/8.20/**5.66**,
`pick_laptop` 5.63/4.92/**4.97**, `pick_plate` 5.04/3.81/**3.62**,
`put_bottle_in_fridge` 4.75/6.12/**4.02**, `sweep_to_dustpan`
4.46/**3.10**/3.50, `dual_push_buttons` 3.64/3.16/**1.94**, `push_box`
3.31/2.81/**2.14**, `take_tray_out_of_oven` 2.04/1.76/**1.70**.

### 3d. Per-arm asymmetry — question (iii)

`|pos_l2_arm0 - pos_l2_arm1|` at n15: R1a 0.0064, R1b 0.0087, R1c **0.0039**.
The hypothesis (the bimanual midpoint aux target should balance the arms) is
**supported**: R1c is the most symmetric arm under heavy corruption and R1b the
least, i.e. adding deltaM without the aux loss *worsens* arm balance and the
aux loss more than recovers it. At clean/base/n2/n5 all three are <=0.002 and
the metric does not discriminate — the asymmetry only opens up under corruption.
Caveat: these are ALL-task aggregates and the effect is ~4 mm; it is
directionally clean but not independently tested per task.

### 3e. Does delta_M sense miscalibration? — question (iv), the smoking gun

The head is initialized at ~I, so `||delta_M - I||_F` reads out how much
correction it thinks it needs. Recorded at the birth site
(`_predict_from_cam_feat`), because `dynamic_rope_from_camtoken` re-predicts
after every cross- and self-attn block and the head runs once per denoising
step, so the `_last_predicted_cam_params` stash retains only the final
prediction. (The trainer's own logging silently drops delta_M behind a
`dim()==2 and shape[-1]==6` guard — it has never been looked at.)

Mean `||delta_M - I||_F` across the whole corruption sweep:

| arm | clean | n15 | change | as % of sample-level SD |
|---|---|---|---|---|
| R1b | 0.652822 | 0.652635 | **-0.000188** | **0.22%**, wrong direction |
| R1c | 0.681375 | 0.683580 | **+0.002205** | **1.50%**, right direction |

Per-camera regression of deviation on that camera's *actual* injected rotation
magnitude:

| cam | R1b slope (/deg) | R1b range as % of SD | R1c slope (/deg) | R1c range as % of SD |
|---|---|---|---|---|
| 0 | -0.000051 | 0.65% | **+0.000349** | **2.76%** |
| 1 | -0.000032 | 0.53% | **+0.000335** | **3.34%** |
| 2 | +0.000022 | 0.57% | -0.000029 | 0.27% |
| 3 | +0.000011 | 0.32% | +0.000042 | 0.43% |

Two conclusions, and the second is the more interesting:

- **delta_M is, to three decimal places, a learned constant.** Between-condition
  movement is 0.2-3% of within-condition sample-to-sample SD, and that SD is
  itself mostly cross-camera and cross-scene variation, not calibration
  response. Whatever robustness deltaM buys, it is **not** delivered by sensing
  the perturbation and correcting it. The mechanism is better described as a
  learned reparameterization of the RoPE basis that happens to be flatter in the
  corruption direction — closer to a regularizer or an implicit basis change
  than to a calibration estimator. The per-sample correlations
  (`corr_dev_vs_injected`: R1b -0.10 to -0.21, R1c +0.03 to +0.12) confirm this:
  R1b's is consistently *negative*.
- **The only cameras that respond are cameras 0 and 1 — exactly
  `ee_aux_cam_ids=[0,1]`.** R1c's slope is 10x R1b's on cams 0/1 and
  indistinguishable from R1b on cams 2/3. The EE-aux loss is what makes the
  extrinsics token calibration-responsive, and it does so **only on the cameras
  it supervises.** This is the mechanistic explanation for why R1c is the only
  arm with a replicated advantage, and it is a directly actionable design
  finding: the aux supervision, not the deltaM parameterization, is the
  load-bearing part.

Corroborating quirk: camera 3 (`wrist_right`) is **identity-padded** — the noise
JSON lists 3 cameras but `ncam=4`, so cam 3 is never perturbed in training or
eval. In R1b it nevertheless carries the *largest* deviation from identity
(0.723 in the earlier 128-sample probe; 0.666 at full scale), i.e. the head
assigns a large "correction" to a camera that is by construction perfectly
calibrated. That is only coherent if the matrix is a learned constant.

---

## 4. Where the advantage is largest — direct answer

Ranked by strength of evidence, all for **R1c**:

1. **Held-out calibration direction at fixed magnitude** (`ood_base`):
   `pos_acc_001` +0.099 (p=0.0029), closed-loop SR diff-in-diff +0.154. Largest
   effect size of any condition. This is the regime to build the narrative on.
2. **Heavy corruption, 10-15deg** (`n10`,`n15`): `pos_acc_001` +0.053
   (p=0.0005, 12/13 tasks); closed-loop SR +0.119 to +0.185; pooled rollout SR
   0.285 vs 0.165.
3. **Rate of degradation** rather than level: 0.63 mm/deg flatter (p=0.0068),
   and closed-loop retention never crosses 50% within 15deg where R1a crosses at
   8.7deg.
4. **Bimanual arm balance under corruption**: n15 asymmetry 0.0039 vs 0.0064
   (R1a) and 0.0087 (R1b).

Regimes where there is **no** advantage, and should not be claimed: clean, the
trained base, and low corruption (n2/n5) — where R1c's closed-loop SR is
significantly *worse*. And for R1b: nowhere that replicates.

---

## 5. Honesty ledger

- **The R1b slope claim is retracted.** Tier 1 gave +0.0134 SR/deg p=0.0034,
  11/13; Tier 2 with ~65x the samples gives -0.089 mm/deg p=0.74, 7/13. A
  task-level permutation test on 13 tasks whose per-task inputs are each 10
  binary rollouts is not immune to the noise floor — it controls the
  across-task inference, not the within-cell variance that feeds it. Believe
  the offline number.
- **Two Tier-1 findings survive Tier 2** (R1c's high-corruption win and the
  invariance-to-direction effect) and one is contradicted. That is the expected
  attrition rate for 10-rollout cells and is the reason Tier 2 was worth
  running.
- **Family decomposition is unreliable.** Tier 1 says the gain concentrates in
  coordination-gross and is negative for articulated; Tier 2 says R1c gains on
  all three. n=3-5 tasks per family. Do not use.
- **`pos_acc_005` and gripper accuracy are near-saturated** and separate the
  arms only at n10/n15. `pos_acc_001` is the metric with power.
- **Open-loop != closed-loop.** R1c is *better* offline at clean/base/n2 while
  being *worse* in closed-loop SR there. Per-keypose accuracy and rollout
  success are dissociated in this direction, which is unexplained and worth its
  own look.
- **`ood_base` is confounded with viewpoint**, not just calibration: it is a
  different fixed base, and all three arms lose 0.10-0.14 of `pos_acc_001`
  absolutely. The paired diff-in-diff is what is interpretable, not the level.
- **Camera 3 is never perturbed** in any condition here (identity padding, see
  3e). Every "corruption magnitude" quoted is really over 3 of 4 cameras.
- **One seed per arm.** Nothing here separates a deltaM effect from a
  seed effect. This is the single largest threat to all of the above and no
  amount of eval-side analysis can fix it.
- **Tier 1(e) (video inspection of R1c-wins/R1a-loses cells) was not done.** The
  offline error analysis subsumed its purpose with far more power, so it was
  dropped rather than run.
- **The 3e per-camera mechanism claim is retracted** (added after section 6).
  3e read a per-camera regression of `||delta_M - I||_F` on that camera's injected
  angle and found cams 0/1 responsive. That regression cannot distinguish
  "responds to its own camera" from "responds to overall corruption" because in
  the Tier-2 grid **all cameras were perturbed together**, so every camera's
  injected angle is collinear with total corruption. The T3-1 design breaks that
  collinearity by perturbing one camera at a time, and the response turns out to
  be global. This is a confounded-regressor error in 3e, not a data problem —
  the numbers in 3e are correct, the causal reading of them was not.
- **Section 6's effects are millimetre-scale by construction.** Single-camera
  corruption moves `pos_l2` by 1-5 mm on a 16 mm base, ~10x less than
  all-camera corruption at the same per-camera magnitude. Three clean views are
  nearly as good as four. Do not quote section 6 effect sizes as evidence of
  practical robustness; they are mechanism diagnostics.

---

## 6. Asymmetric miscalibration — T3-1, run

`scripts/eval/offline_asym_miscal_analysis.py` +
`scripts/eval/analyze_asym_miscal.py`. Corrupt **exactly one** camera and leave
the other three clean; sweep which camera. Two magnitudes (5deg+5cm,
10deg+10cm), **3 independent random directions per cell**, all three arms
evaluated against the identical direction set. 853 val samples x 75 passes
(3 arms x (4 cams x 2 magnitudes x 3 directions + 1 clean)), ~25 min on one
H200. Bootstrap CIs resample **episodes** (n=117, `demo_id`) and are paired
across arms.

Sanity check: the clean pooled `pos_l2` reproduces section 3a exactly (R1a
0.0145, R1b 0.0160, R1c 0.0139), so this harness and the Tier-2 one agree on the
shared cell. The tables below quote *per-episode* means, which run ~0.0017 m
higher than sample-pooled means because long episodes are downweighted; only
differences are interpreted, and both weightings give the same signs.

### 6a. The three predictions, up front

| prediction | verdict |
|---|---|
| (a) R1c degrades less than R1a/R1b when the bad camera is **0 or 1** (aux-supervised) | **HELD** at 10deg: R1c-R1a `pos_l2` **-0.0008 CI[-0.0014,-0.0002]** (cam0), **-0.0017 CI[-0.0023,-0.0011]** (cam1); `pos_acc_001` **+0.046** and **+0.066**, both CIs excluding zero. |
| (b) all arms degrade similarly when the bad camera is **2** (wrist_left, unsupervised) | **FAILED, and inverted.** R1c degrades *more*: +0.0038 CI[+0.0022,+0.0054] worse than R1a at 10deg; R1b +0.0023 CI[+0.0011,+0.0036] worse. Both deltaM arms are *hurt* by wrist_left corruption relative to the baseline, 0/3 directions won. |
| (c) cam3 (`wrist_right`, identity-padded in training) is the OOD case | **R1c generalizes, R1b does not.** R1c **-0.0025 CI[-0.0035,-0.0016]** better than R1a at 10deg (3/3 directions); R1b **+0.0023 CI[+0.0014,+0.0032]** *worse* (1/3). |

So the camera x arm interaction is real, but it is **not the diagonal the
mechanism story predicted**. R1c is more robust on cams 0, 1 **and 3**, and less
robust on cam 2. Two of three predictions did not come out as stated.

### 6b. Degradation vs own clean (per-episode `pos_l2`, m)

10deg+10cm on one camera, rest clean:

| corrupted cam | R1a | R1b | R1c |
|---|---|---|---|
| 0 `orbital_left` | +0.0027 | **+0.0011** | +0.0019 |
| 1 `orbital_right` | +0.0030 | +0.0019 | **+0.0013** |
| 2 `wrist_left` | **+0.0021** | +0.0043 | +0.0058 |
| 3 `wrist_right` | +0.0052 | +0.0075 | **+0.0026** |

5deg+5cm:

| corrupted cam | R1a | R1b | R1c |
|---|---|---|---|
| 0 | +0.0003 | **-0.0001** | +0.0004 |
| 1 | +0.0011 | +0.0011 | **+0.0009** |
| 2 | **+0.0013** | +0.0028 | +0.0030 |
| 3 | +0.0014 | +0.0020 | **+0.0007** |

The absolute magnitudes are the first thing to notice: **single-camera
corruption is cheap.** The worst cell here (+0.0052 m for R1a at cam3) is a
32% error increase, where corrupting *all four* cameras at the same 10deg costs
R1a **+0.0210 m** — a 169% increase. That all-camera figure is a separate
positive control on a 256-sample subset (R1a clean 0.0124 -> allcam-10deg
0.0334), run to confirm the harness can move the metric at all before trusting
the small single-camera numbers. On the same subset a **20deg** single-camera
corruption of cam0 moved R1c's error by **-0.00001 m** (0.01200 -> 0.01199) —
indistinguishable from zero at 4x the magnitude of the main sweep's larger cell.
Three clean cameras out of four localize the scene nearly as well as four. That
caps how much any per-camera correction mechanism could buy here, and it is the
main reason every effect in this section is millimetre-scale.

`pos_acc_001` is more sensitive and tells the same story with the same signs
(R1c-R1a at 10deg: cam0 +0.046, cam1 +0.066, cam2 +0.000, cam3 +0.034).

### 6c. Direction-paired replication

Direction spread is large — the per-cell SD across the 3 directions is often
comparable to the mean degradation itself (e.g. R1b cam3 m10: +0.0068, +0.0141,
-0.0000). That vindicates sampling multiple directions and means single-direction
readings are not trustworthy. But every arm saw the *same* directions, so the
paired contrast is far tighter than the marginals. Sign consistency of
`R1x - R1a` in `pos_l2` degradation, out of 3 directions (lower = better):

| cam | mag | R1b better | R1c better |
|---|---|---|---|
| 0 | 10deg | 3/3 | 3/3 |
| 1 | 10deg | 2/3 | **3/3** |
| 2 | 10deg | 0/3 | 0/3 |
| 3 | 10deg | 1/3 | **3/3** |
| 3 | 5deg | 0/3 | **3/3** |

R1c's cam0/cam1/cam3 wins and its cam2 loss are 3/3 and 0/3 respectively —
consistent across directions, not a single-axis artifact. R1b is only consistent
at cam0.

### 6d. delta_M response matrix — the prediction that failed hardest

Change in `||delta_M - I||_F` vs clean, row = corrupted camera, column = the
camera whose token is read out, expressed as % of that token's clean sample SD
(the section-3e scale, where 0.2-3% was the verdict "nearly a learned constant").

**R1c, 10deg+10cm:**

| corrupted cam | tok0 | tok1 | tok2 | tok3 |
|---|---|---|---|---|
| 0 `orbital_left` | -0.02% | +0.09% | -0.12% | -0.08% |
| 1 `orbital_right` | **+0.69%** | **+0.69%** | +0.17% | +0.26% |
| 2 `wrist_left` | -0.73% | -0.75% | -0.03% | -0.49% |
| 3 `wrist_right` | **+0.64%** | **+0.53%** | +0.11% | -0.21% |

**R1b, 10deg+10cm:** every entry is within ±0.5% of SD and the largest is on a
row/column pair with no mechanistic meaning (cam3 corrupted -> tok1, +0.46%).

Three things follow, and they undercut the sensing story rather than confirming
it:

- **The response is not diagonal — it is column-structured.** When *any* camera
  is corrupted, the response shows up in **tok0 and tok1** regardless of which
  camera was actually perturbed. Corrupting cam3 moves tok0 by +0.64% and tok1
  by +0.53%, while moving tok3 (the actually-corrupted camera's own token) by
  **-0.21%**. Corrupting cam1 moves tok0 as much as tok1. The aux-supervised
  tokens are the only ones that move at all, but they move in response to
  corruption *anywhere*, not corruption of themselves. That is a **global
  corruption detector living in the supervised tokens**, not per-camera
  calibration sensing.
- **The magnitudes remain tiny.** The largest response in the whole matrix is
  0.75% of the token's own clean sample SD. Section 3e's "delta_M is
  approximately a learned constant" verdict **stands and is reinforced** by a
  measurement designed specifically to break it. A per-camera targeted probe was
  the best available chance to find per-camera structure, and it found ~0.7%.
- **The sign is not even consistent.** cam1 and cam3 corruption push tok0/tok1
  *up*; cam2 corruption pushes them *down* by a similar amount. A correction
  magnitude should not reverse sign with which camera is broken.

### 6e. The wrist asymmetry — a train-distribution effect, not a deltaM effect

cam2 (`wrist_left`, perturbed during training) vs cam3 (`wrist_right`,
identity-padded and never perturbed), difference in `pos_l2` degradation:

| arm | 5deg | 10deg | cam3 - cam2 @ 10deg [95% CI] |
|---|---|---|---|
| R1a | +0.0013 / +0.0014 | +0.0021 / +0.0052 | **+0.0031 [+0.0021,+0.0042]** |
| R1b | +0.0028 / +0.0020 | +0.0043 / +0.0075 | **+0.0032 [+0.0010,+0.0053]** |
| R1c | +0.0030 / +0.0007 | +0.0058 / +0.0026 | **-0.0032 [-0.0051,-0.0014]** |

For R1a and R1b, corrupting the never-corrupted wrist hurts **~2.5x more** than
corrupting the wrist they were trained to see corrupted, at matched magnitude
(+0.0052 vs +0.0021 for R1a). That is a clean train-distribution generalization
gap and it is **independent of deltaM** — R1a has no extrinsics head and shows
the effect just as strongly. It is the cleanest result in this section and the
one least dependent on any mechanism claim.

R1c **reverses** it: it is the only arm for which the OOD wrist is *easier* than
the in-distribution one. Combined with 6d, the most consistent reading is that
the aux loss makes R1c less dependent on any individual wrist view — which shows
up as robustness to the untrained one and as *fragility* to the trained one,
where R1a apparently learned a specific compensation R1c did not.

### 6f. Does this strengthen "aux supervision is the engine"?

**It complicates it more than it strengthens it.** The honest summary:

- **What survives.** R1c is the most robust arm on 3 of 4 cameras and the effect
  replicates across corruption directions. R1b is not consistently better than
  R1a anywhere. So "R1c is the only arm worth pursuing" — the section-1 headline
  — holds under a new and independent stressor. This is a fourth condition where
  R1c wins and R1b does not.
- **What does not survive.** The specific mechanism proposed in 3e — that the
  aux loss makes each supervised camera's token sense *that camera's*
  miscalibration and correct it — is **refuted by its own best test.** The
  response is column-structured, not diagonal; it fires on tok0/tok1 whichever
  camera is broken; it reverses sign; and it is <1% of SD. R1c is also more
  robust on cam3, which no camera-specific-correction story predicts, and *less*
  robust on cam2, which it predicts should be neutral.
- **What replaces it.** The aux loss appears to act as a **representational
  regularizer** that reduces reliance on any single camera, not as a calibration
  estimator. Everything observed is consistent with that: robustness spread over
  cams 0/1/3 rather than concentrated on 0/1; a global rather than per-camera
  delta_M response; better OOD-wrist behavior; and the section-2e cost at low
  corruption, which is what a regularizer that discards view-specific
  information would charge.
- **Does it motivate aux-on-all-cameras?** **Weakened, and it should not be the
  next launch.** The original argument was "the token only senses on supervised
  cameras, so supervise all of them." That premise is now gone — the tokens do
  not sense per-camera at all. R1c already gets its largest gain on cam3, which
  is *unsupervised*, so extending supervision to cams 2/3 is not obviously where
  the gain lives. It also has a specific predicted failure: R1c's one loss is on
  cam2, and the natural reading of 6e is that aux supervision trades away the
  view-specific compensation R1a uses there. Supervising cam2 might deepen that
  loss rather than fix it.
- **What to run instead.** (i) **A second seed** — one seed per arm remains the
  binding limitation, and the effects in this section are 1-4 mm, well within
  plausible seed variation. (ii) If the regularizer reading is right, the cheap
  test is an **aux target that is not extrinsics-shaped at all** (e.g. the same
  EE-midpoint target from a non-extrinsics head, or plain feature dropout across
  cameras), which would separate "aux supervision regularizes" from "aux
  supervision teaches calibration." That is a sharper and cheaper experiment than
  aux-on-all-cameras.

### 6g. Caveats specific to this section

- **Ceiling effect.** Single-camera corruption barely moves the metric (see 6b),
  so every contrast here is millimetre-scale on a base error of ~16 mm. The
  arms are being separated inside a narrow band. Contrast with the all-camera
  positive control, where the same 10deg costs 10x more.
- **Open-loop only.** Section 5's warning applies: R1c's open-loop and
  closed-loop rankings dissociated before. Nothing here says these differences
  survive rollout.
- **3 directions per cell**, and the direction SD is comparable to the effect
  (6c). The paired contrast is what carries the inference; the marginal
  degradation numbers individually are noisy.
- **One seed per arm.** Unchanged and still the dominant threat.
- **The delta_M readout is a scalar norm.** `||delta_M - I||_F` could stay
  constant while the matrix *rotates* to track the perturbation. This section
  rules out a magnitude response, not a directional one. A per-camera readout of
  the matrix's action on the RoPE basis would be the stronger probe and was not
  done.

Reproduce:

```bash
CK=/k8s-nfs/harsvbha/3dfa/train_logs/exp
ulimit -n 65536   # 75 passes; the default 1024 exhausts FDs
HF_HOME=/k8s-nfs/harsvbha/3dfa/hf-cache HF_HUB_CACHE=/k8s-nfs/harsvbha/3dfa/hf-cache \
CUDA_VISIBLE_DEVICES=0 /k8s-nfs/harsvbha/3dfa/venv/bin/python \
  scripts/eval/offline_asym_miscal_analysis.py \
  checkpoints=$CK/orbital_nhist3_miscal_base/interm_step_100000.pth,$CK/orbital_nhist3_miscal_deltaM/interm_step_100000.pth,$CK/orbital_nhist3_miscal_deltaM_eeaux/interm_step_100000.pth \
  arm_names=R1a,R1b,R1c \
  data_path=/k8s-nfs/harsvbha/3dfa/data/orbital_peract2/val.zarr \
  samples_npz=results/asym_miscal/samples.npz n_directions=3 num_batches=100 \
  data=orbital_peract2_nfs bimanual=true dataset=OrbitalPeract2 \
  num_history=3 batch_size_val=64 num_workers=8

python scripts/eval/analyze_asym_miscal.py \
  samples_npz=results/asym_miscal/samples.npz \
  out_md=results/asym_miscal/tables.md n_boot=20000
```

Raw per-sample records, the full table dump, and the run log are staged at
`s3://far-research-internal/harsvbha/3dfa/eval/results/asym_miscal/`
(`samples.npz`, `tables.md`, `run.log`). The npz holds every keypose so the
aggregation can be redone without re-running inference.

---

## 7. Tier 3 — proposed next evals (T3-1 now run, see section 6)

All framed as **R1c vs R1a**, with R1b as the ablation that isolates whether
the aux loss is necessary — which section 3e says it is. Ranked by expected
information per GPU-hour.

### T3-1. Asymmetric per-camera miscalibration — **RUN, see section 6**

One camera badly wrong (15-20deg), the rest clean. Sweep which camera.

**Outcome:** the camera x arm interaction is real but not the predicted one. R1c
is more robust on cams 0, 1 **and 3** and *less* robust on cam 2; the delta_M
response is column-structured (fires on tok0/tok1 whichever camera is broken)
rather than diagonal, so the per-camera-correction mechanism below is refuted.
The proposal's "risk it shows nothing" branch is roughly what happened for the
mechanism claim — and it was informative, as anticipated. The design rationale
is left below as written for the record.

- **Motivation.** The strongest mechanistic result (3e) is that only the
  aux-supervised cameras 0/1 respond to perturbation. A per-camera-targeted
  corruption tests that directly: R1c should degrade much less when the
  corrupted camera is 0 or 1 than when it is 2, and R1a should be indifferent
  to which camera it is.
- **Demonstrates superiority if:** R1c-R1a gain is large for cam0/cam1
  corruption and ~0 for cam2, i.e. a camera x arm interaction. That would turn
  "deltaM helps" into "aux-supervised deltaM corrects the cameras it
  supervises" — a mechanism claim, not a benchmark claim, and by far the most
  publishable outcome available.
- **Cost.** 3 arms x 3 corrupted-camera choices x 13 tasks. Run offline first
  (~1 GPU-hour total, the harness already supports it with a per-camera mask)
  and only promote to closed-loop if the interaction appears.
- **Risk it shows nothing.** Moderate. If the constant-matrix reading is
  complete, no camera-specific interaction exists and the result is flat. But
  flat here is *also* informative: it would kill the correction story outright
  and settle the mechanism question.

### T3-2. Larger fixed miscalibration base (10-15deg), per-episode resampled

Replace the fixed per-group base with a per-episode-resampled base at 10-15deg.

- **Motivation.** Both surviving effects live at high magnitude and at *novel*
  direction. A per-episode-resampled base is the purest form of that: tolerance
  cannot average over a base it sees once. And 10-15deg is where R1a's tolerance
  is exhausted (its 50% retention crossing is 8.7deg) so the ceiling is out of
  the way.
- **Demonstrates superiority if:** R1c-R1a gap widens beyond the current +0.119
  SR / +0.053 acc001 — this is the condition designed to maximize it.
- **Cost.** Highest of the three: needs 3 retrains (per-episode resampling is a
  training-time change), then eval. Only worth it after T3-1 says the mechanism
  is real.
- **Risk it shows nothing.** Low-moderate for the comparison, but high for
  *usefulness*: at 15deg all arms may collapse to near-zero SR, compressing the
  gap. Mitigate by including 10deg.

### T3-3. In-domain camera + corruption sweep

Same corruption grid, but on the in-domain camera configuration rather than the
orbital/OOD viewpoint.

- **Motivation.** Section 5 flags that `ood_base` mixes viewpoint novelty with
  calibration error, and that all arms lose 0.10-0.14 absolute there. Removing
  the viewpoint confound stops it eating everyone's slack and should expose the
  calibration effect more cleanly.
- **Demonstrates superiority if:** the R1c-R1a gap at n10/n15 persists or grows
  with the viewpoint confound removed. If it *shrinks*, the "advantage" was
  really viewpoint robustness, which is a materially different (and weaker)
  claim.
- **Cost.** Lowest — eval only, no retrain, the offline harness runs it as-is.
  ~1 GPU-hour.
- **Risk it shows nothing.** Low. This one is diagnostic either way and should
  be run alongside T3-1 simply because it is cheap.

### Not recommended

A fourth deltaM training sweep. Section 3e says the deltaM parameterization is
not the load-bearing part; the aux supervision is. If anything is retrained, the
right axis is **aux-loss design** (which cameras, what target, `lambda_aux`),
not more deltaM variants. And before any of it: **a second seed per arm**, since
one seed per arm is the binding limitation on every number in this document.

Section 6f narrows this further: **aux-on-all-cameras is no longer the indicated
aux-loss variant**, because the per-camera premise it rested on did not hold. The
indicated experiments are a second seed, and an aux target that is not
extrinsics-shaped, to test the regularizer reading directly.

---

## 8. Reproducing

Tier 2, on a 1-GPU devbox with the repo at `/root/3dfa/3d_flowmatch_actor`:

```bash
HF_HOME=/k8s-nfs/harsvbha/3dfa/hf-cache HF_HUB_CACHE=/k8s-nfs/harsvbha/3dfa/hf-cache \
CUDA_VISIBLE_DEVICES=0 /k8s-nfs/harsvbha/3dfa/venv/bin/python \
  scripts/eval/offline_deltam_analysis.py \
  checkpoints=$CK/orbital_nhist3_miscal_base/interm_step_100000.pth,$CK/orbital_nhist3_miscal_deltaM/interm_step_100000.pth,$CK/orbital_nhist3_miscal_deltaM_eeaux/interm_step_100000.pth \
  data_path=/k8s-nfs/harsvbha/3dfa/data/orbital_peract2/val.zarr \
  output_csv=errors.csv deltam_csv=deltam.csv \
  num_batches=100 data=orbital_peract2_nfs bimanual=true dataset=OrbitalPeract2 \
  num_history=3 batch_size_val=64 num_workers=8
```

Runs in ~8 min for the full 3 x 7 grid. Architecture and dataset type are
restored from each checkpoint's saved `config`. Checkpoint stems collide (all
`interm_step_100000`), so rows are disambiguated by write order — pass
distinctly-named copies if that matters downstream.
