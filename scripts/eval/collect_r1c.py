#!/usr/bin/env python3
"""Collect the R1c eval grid from S3 into the comparison tables for R2c.

Reads every s3://.../orbital_miscal_deltam_eeaux/<cond>/<task>.json written by
scripts/sky/peract2_orbital_online_eval.yaml and prints (a) coverage, (b) the
per-task R1c table, and (c) the condition | R1a | R1b | R1c comparison, with the
R1a/R1b reference numbers from docs/status/experiments.md (R2, R2b).
"""

import json
import subprocess
import sys

BUCKET = "far-research-internal"
PREFIX = "harsvbha/3dfa/eval/results/orbital_miscal_deltam_eeaux"

CONDS = [
    "level0",
    "noise_2deg2cm",
    "noise_5deg5cm",
    "noise_10deg10cm",
    "noise_15deg15cm",
    "ood_miscal",
    "clean0",
]
TASKS = [
    "bimanual_push_box",
    "bimanual_lift_ball",
    "bimanual_dual_push_buttons",
    "bimanual_pick_plate",
    "bimanual_put_item_in_drawer",
    "bimanual_put_bottle_in_fridge",
    "bimanual_handover_item",
    "bimanual_pick_laptop",
    "bimanual_straighten_rope",
    "bimanual_sweep_to_dustpan",
    "bimanual_lift_tray",
    "bimanual_handover_item_easy",
    "bimanual_take_tray_out_of_oven",
]

# docs/status/experiments.md — R2 three-curve table and R2b OOD table.
REF = {
    "level0": (0.623, 0.508),
    "noise_2deg2cm": (0.654, 0.500),
    "noise_5deg5cm": (0.485, 0.469),
    "noise_10deg10cm": (0.254, 0.262),
    "noise_15deg15cm": (0.077, 0.138),
    "ood_miscal": (0.415, 0.446),
    "clean0": (0.692, 0.569),
}
LABEL = {
    "level0": "trained fixed miscal (level 0)",
    "noise_2deg2cm": "+ random 2deg+2cm",
    "noise_5deg5cm": "+ random 5deg+5cm",
    "noise_10deg10cm": "+ random 10deg+10cm",
    "noise_15deg15cm": "+ random 15deg+15cm",
    "ood_miscal": "held-out fixed miscal",
    "clean0": "clean extrinsics",
}


def fetch(cond: str, task: str):
    """Return the task's mean SR, or None when the cell has not landed yet."""
    uri = f"s3://{BUCKET}/{PREFIX}/{cond}/{task}.json"
    p = subprocess.run(
        ["aws", "s3", "cp", uri, "-"],
        capture_output=True,
        text=True,
        env={"AWS_PROFILE": "far-compute", "PATH": "/usr/bin:/bin:/usr/local/bin"},
    )
    if p.returncode != 0:
        return None
    return json.loads(p.stdout)[task]["mean"]


def main() -> int:
    # 91 serial `aws s3 cp` calls take minutes; the fetches are independent.
    from concurrent.futures import ThreadPoolExecutor

    cells = [(c, t) for c in CONDS for t in TASKS]
    with ThreadPoolExecutor(max_workers=16) as pool:
        vals = list(pool.map(lambda ct: fetch(*ct), cells))
    grid = {c: {} for c in CONDS}
    for (c, t), v in zip(cells, vals):
        grid[c][t] = v

    missing = [(c, t) for c in CONDS for t in TASKS if grid[c][t] is None]
    have = sum(1 for c in CONDS for t in TASKS if grid[c][t] is not None)
    print(f"coverage: {have}/{len(CONDS) * len(TASKS)} cells")
    if missing:
        print(f"MISSING ({len(missing)}):")
        for c, t in missing:
            print(f"  {c}/{t}")

    def mean(cond):
        vals = [v for v in grid[cond].values() if v is not None]
        return sum(vals) / len(vals) if vals else float("nan")

    print("\n### Per-task: R1c (miscal-trained + deltaM + EE-aux)\n")
    hdr = ["Task", "0", "2deg+2cm", "5deg+5cm", "10deg+10cm", "15deg+15cm", "ood", "clean0"]
    print("| " + " | ".join(hdr) + " |")
    print("|---|" + ":---:|" * (len(hdr) - 1))
    for t in TASKS:
        cells = [
            "—" if grid[c][t] is None else f"{grid[c][t]:.1f}"
            for c in CONDS
        ]
        print(f"| {t.replace('bimanual_', '')} | " + " | ".join(cells) + " |")
    means = [f"**{mean(c):.3f}**" for c in CONDS]
    print("| **MEAN** | " + " | ".join(means) + " |")

    print("\n### condition | R1a | R1b | R1c — mean SR over 13 tasks, OOD camera\n")
    print("| condition | R1a (no deltaM) | R1b (deltaM) | R1c (deltaM + EE-aux) | R1c-R1b | R1c-R1a |")
    print("|---|:---:|:---:|:---:|:---:|:---:|")
    for c in CONDS:
        a, b = REF[c]
        m = mean(c)
        print(
            f"| {LABEL[c]} | {a:.3f} | {b:.3f} | **{m:.3f}** "
            f"| {m - b:+.3f} | {m - a:+.3f} |"
        )
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
