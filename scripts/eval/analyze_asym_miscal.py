"""Aggregate the asymmetric per-camera miscalibration sweep.

Consumes the per-sample ``.npz`` written by
:mod:`scripts.eval.offline_asym_miscal_analysis` and produces the three tables the
Tier-3 T3-1 verdict needs:

1. **Degradation table** — per (arm, corrupted camera, magnitude), the increase in
   keypose position error over that arm's own clean pass, pooled and per arm-of-the-
   robot (left/right EE). Averaged over the independent corruption directions.
2. **Paired arm contrasts** — R1c-vs-R1a and R1b-vs-R1a differences in that
   degradation, with a bootstrap CI. Resampling is over **episodes** (``demo_id``),
   not keyposes, because keyposes within an episode share a scene; and it is
   **paired**, since every arm saw the same episodes under the same corruptions.
3. **delta_M response matrix** — ``(corrupted camera) x (token camera)`` change in
   ``||delta_M - I||_F`` relative to clean. The prediction from the Tier-2 result is
   a diagonal-ish response confined to rows 0/1 of R1c and nothing anywhere in R1b.

Usage::

    python scripts/eval/analyze_asym_miscal.py \\
        samples_npz=results/asym_miscal/samples.npz \\
        out_md=results/asym_miscal/tables.md \\
        n_boot=10000
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

CAM_NAMES = {0: "orbital_left", 1: "orbital_right", 2: "wrist_left", 3: "wrist_right"}
MAG_LABELS = ["m5", "m10"]
MAG_TITLES = {"m5": "5deg+5cm", "m10": "10deg+10cm"}


def load_records(npz_path):
    """Group the flat ``arm|condition|field`` npz keys into a nested dict."""
    d = np.load(npz_path, allow_pickle=True)
    recs = defaultdict(dict)
    for key in d.files:
        arm, cond, field = key.split("|")
        recs[(arm, cond)][field] = d[key]
    out = {}
    for (arm, cond), fields in recs.items():
        meta = fields.pop("meta")
        fields["cam"] = int(meta[2])
        fields["mag"] = str(meta[3])
        fields["direction"] = int(meta[4])
        out[(arm, cond)] = fields
    return out


def episode_means(rec, field):
    """``(episode_ids, per-episode mean of field)`` — the bootstrap resampling unit."""
    demo = rec["demo_id"]
    eps = np.unique(demo)
    return eps, np.array([rec[field][demo == e].mean() for e in eps])


def build_episode_matrix(recs, arms, field):
    """``{(arm, cond): (n_eps,) per-episode means}`` on a shared episode index.

    Asserts every (arm, condition) pass covers the same episode set, which is what
    makes the downstream contrast paired.
    """
    ref_eps = None
    out = {}
    for (arm, cond), rec in recs.items():
        if arm not in arms:
            continue
        eps, vals = episode_means(rec, field)
        if ref_eps is None:
            ref_eps = eps
        elif not np.array_equal(eps, ref_eps):
            raise ValueError(f"{arm}/{cond} covers a different episode set")
        out[(arm, cond)] = vals
    return ref_eps, out


def conds_for(recs, arm, cam, mag):
    """All direction-replicate condition names for one (arm, camera, magnitude) cell."""
    return sorted(
        cond for (a, cond), r in recs.items()
        if a == arm and r["cam"] == cam and r["mag"] == mag
    )


def degradation(mat, recs, arm, cam, mag):
    """Per-episode error increase vs clean, averaged over corruption directions."""
    clean = mat[(arm, "clean")]
    conds = conds_for(recs, arm, cam, mag)
    if not conds:
        return None
    return np.mean([mat[(arm, c)] - clean for c in conds], axis=0)


def boot_ci(x, n_boot, rng, alpha=0.05):
    """Percentile bootstrap CI of the mean, resampling episodes with replacement."""
    n = len(x)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = x[idx].mean(1)
    return float(np.mean(x)), float(np.quantile(means, alpha / 2)), float(
        np.quantile(means, 1 - alpha / 2)
    )


def paired_ci(a, b, n_boot, rng, alpha=0.05):
    """CI of ``mean(a - b)`` with the *same* episode resample applied to both."""
    n = len(a)
    idx = rng.integers(0, n, size=(n_boot, n))
    diffs = (a - b)[idx].mean(1)
    d = float(np.mean(a - b))
    return d, float(np.quantile(diffs, alpha / 2)), float(np.quantile(diffs, 1 - alpha / 2))


def deltam_response(recs, arm):
    """``(4, ncam)`` matrix of mean ``||delta_M - I||_F`` change vs clean, per magnitude.

    Row = corrupted camera, column = the camera whose token is being read out.
    Returns ``{mag: matrix}``, or ``{}`` when the arm has no extrinsics predictor.
    """
    clean = recs[(arm, "clean")]["dev"]
    if clean.size == 0:
        return {}
    ncam = clean.shape[1]
    clean_mean = clean.mean(0)
    out = {}
    for mag in MAG_LABELS:
        M = np.full((4, ncam), np.nan)
        for cam in range(4):
            conds = conds_for(recs, arm, cam, mag)
            if not conds:
                continue
            M[cam] = np.mean([recs[(arm, c)]["dev"].mean(0) for c in conds], 0) - clean_mean
        out[mag] = M
    return out


def fmt(v, digits=4):
    return "n/a" if v is None or v != v else f"{v:+.{digits}f}"


def main():
    custom = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    npz_path = custom.get("samples_npz", "results/asym_miscal/samples.npz")
    out_md = Path(custom.get("out_md", "results/asym_miscal/tables.md"))
    n_boot = int(custom.get("n_boot", 10000))
    rng = np.random.default_rng(int(custom.get("seed", 0)))

    recs = load_records(npz_path)
    arms = sorted({a for a, _ in recs})
    n_dirs = len(conds_for(recs, arms[0], 0, "m5"))
    print(f"arms={arms}  passes={len(recs)}  directions/cell={n_dirs}")

    lines = []
    w = lines.append

    # Position error is the metric with power (pos_acc_005 and gripper accuracy are
    # near-saturated on this val set); arm0/arm1 are the robot's left/right EE.
    fields = {
        "pos": ("pos_l2 (m)", ["pos_arm0", "pos_arm1"]),
        "rot": ("rot err (deg)", ["rot_arm0", "rot_arm1"]),
        "acc001": ("pos_acc_001", ["acc001_arm0", "acc001_arm1"]),
        "grip": ("gripper acc", ["grip_arm0", "grip_arm1"]),
    }
    mats = {}
    for key, (_, subfields) in fields.items():
        for sf in subfields:
            _, mats[sf] = build_episode_matrix(recs, arms, sf)
        # Pooled over the two EEs.
        for arm_cond in list(mats[subfields[0]]):
            mats.setdefault(key, {})[arm_cond] = 0.5 * (
                mats[subfields[0]][arm_cond] + mats[subfields[1]][arm_cond]
            )

    # --- clean reference -----------------------------------------------------
    w("### Clean reference (no corruption)\n")
    w("| arm | pos_l2 (m) | pos_l2 left EE | pos_l2 right EE | rot (deg) | pos_acc_001 | gripper acc |")
    w("|---|---|---|---|---|---|---|")
    for arm in arms:
        w(
            f"| {arm} | {mats['pos'][(arm, 'clean')].mean():.4f} | "
            f"{mats['pos_arm0'][(arm, 'clean')].mean():.4f} | "
            f"{mats['pos_arm1'][(arm, 'clean')].mean():.4f} | "
            f"{mats['rot'][(arm, 'clean')].mean():.2f} | "
            f"{mats['acc001'][(arm, 'clean')].mean():.4f} | "
            f"{mats['grip'][(arm, 'clean')].mean():.4f} |"
        )
    w("")

    # --- degradation --------------------------------------------------------
    for mag in MAG_LABELS:
        w(f"\n### Degradation vs own clean — {MAG_TITLES[mag]} on ONE camera\n")
        w("| corrupted cam | arm | d pos_l2 (m) [95% CI] | d pos_l2 left EE | d pos_l2 right EE | d pos_acc_001 |")
        w("|---|---|---|---|---|---|")
        for cam in range(4):
            for arm in arms:
                dp = degradation(mats["pos"], recs, arm, cam, mag)
                if dp is None:
                    continue
                m, lo, hi = boot_ci(dp, n_boot, rng)
                d0 = degradation(mats["pos_arm0"], recs, arm, cam, mag).mean()
                d1 = degradation(mats["pos_arm1"], recs, arm, cam, mag).mean()
                da = degradation(mats["acc001"], recs, arm, cam, mag).mean()
                w(
                    f"| {cam} ({CAM_NAMES[cam]}) | {arm} | {fmt(m)} [{fmt(lo)},{fmt(hi)}] | "
                    f"{fmt(d0)} | {fmt(d1)} | {fmt(da)} |"
                )
        w("")

    # --- paired contrasts ---------------------------------------------------
    base = arms[0]
    for mag in MAG_LABELS:
        w(f"\n### Paired contrast in degradation vs {base} — {MAG_TITLES[mag]}\n")
        w(f"Negative = the arm degrades LESS than {base}, i.e. more robust.\n")
        w("| corrupted cam | contrast | d(d pos_l2) [95% CI] | d(d pos_acc_001) [95% CI] |")
        w("|---|---|---|---|")
        for cam in range(4):
            for arm in arms[1:]:
                dp_a = degradation(mats["pos"], recs, arm, cam, mag)
                dp_b = degradation(mats["pos"], recs, base, cam, mag)
                if dp_a is None or dp_b is None:
                    continue
                m, lo, hi = paired_ci(dp_a, dp_b, n_boot, rng)
                aa = degradation(mats["acc001"], recs, arm, cam, mag)
                ab = degradation(mats["acc001"], recs, base, cam, mag)
                am, alo, ahi = paired_ci(aa, ab, n_boot, rng)
                w(
                    f"| {cam} ({CAM_NAMES[cam]}) | {arm} - {base} | "
                    f"{fmt(m)} [{fmt(lo)},{fmt(hi)}] | "
                    f"{fmt(am)} [{fmt(alo)},{fmt(ahi)}] |"
                )
        w("")

    # --- delta_M response matrix -------------------------------------------
    w("\n### delta_M response matrix — change in ||delta_M - I||_F vs clean\n")
    w("Row = corrupted camera, column = camera whose token is read out.\n")
    for arm in arms:
        resp = deltam_response(recs, arm)
        if not resp:
            w(f"**{arm}**: no extrinsics predictor.\n")
            continue
        sd = recs[(arm, "clean")]["dev"].std(0)
        w(f"**{arm}** (clean per-cam sample SD: {np.round(sd, 4).tolist()})\n")
        for mag in MAG_LABELS:
            M = resp[mag]
            w(f"{MAG_TITLES[mag]}:\n")
            w("| corrupted cam | tok0 | tok1 | tok2 | tok3 | diagonal / off-diag mean |")
            w("|---|---|---|---|---|---|")
            for cam in range(M.shape[0]):
                row = M[cam]
                off = np.delete(row, cam)
                ratio = f"{fmt(row[cam], 5)} / {fmt(off.mean(), 5)}"
                w(
                    f"| {cam} ({CAM_NAMES[cam]}) | "
                    + " | ".join(fmt(v, 5) for v in row)
                    + f" | {ratio} |"
                )
            w("")

            # Same numbers as a fraction of the clean sample SD — the Tier-2 doc's
            # scale, where 0.2-3% was the verdict "nearly a learned constant".
            w("as % of that token's clean sample SD:\n")
            w("| corrupted cam | tok0 | tok1 | tok2 | tok3 |")
            w("|---|---|---|---|---|")
            for cam in range(M.shape[0]):
                pct = 100.0 * M[cam] / sd
                w(
                    f"| {cam} ({CAM_NAMES[cam]}) | "
                    + " | ".join(f"{v:+.2f}%" for v in pct)
                    + " |"
                )
            w("")

    # --- wrist_left vs wrist_right (train-distribution effect) ---------------
    w("\n### Wrist asymmetry — cam2 (perturbed in training) vs cam3 (identity-padded)\n")
    w("| arm | magnitude | d pos_l2 cam2 | d pos_l2 cam3 | cam3 - cam2 [95% CI] |")
    w("|---|---|---|---|---|")
    for arm in arms:
        for mag in MAG_LABELS:
            d2 = degradation(mats["pos"], recs, arm, 2, mag)
            d3 = degradation(mats["pos"], recs, arm, 3, mag)
            if d2 is None or d3 is None:
                continue
            m, lo, hi = paired_ci(d3, d2, n_boot, rng)
            w(
                f"| {arm} | {MAG_TITLES[mag]} | {fmt(d2.mean())} | {fmt(d3.mean())} | "
                f"{fmt(m)} [{fmt(lo)},{fmt(hi)}] |"
            )
    w("")

    text = "\n".join(lines)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(text)
    print(text)
    print(f"\n-> {out_md}")


if __name__ == "__main__":
    main()
