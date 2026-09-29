#!/usr/bin/env python3
"""Plot task-heldout success vs external-camera miscalibration for every model arm."""
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/grogu/datasets/hbhatia/3dfa_online_eval_100rollouts")
OUT = Path(__file__).resolve().parents[1] / "docs/research/results/figures/heldout_miscal_curves.png"
EXPECTED_TASKS = 13
# Checkpoint directory -> (column label, categorical slot, marker, complete sampling).
ARMS = {
    "peract2_orbital_video_deltam_external_warmstart_k5v_k3p_a5000_resume": ("VideoPooled", "#2a78d6", "o", True),
    "peract2_orbital_vid_deltam_fullpatch_finetune_300k_a6000_b64": ("FullPatch Video", "#eb6834", "s", True),
    "peract2_orbital_vid_deltam_direct_fullpatch_rope3d": ("Rope3D direct", "#1baf7a", "^", False),
    "peract2_orbital_vid_deltam_refine_fullpatch_rope3d": ("Rope3D refine", "#eda100", "D", False),
    "peract2_orbital_new_external_only_miscal_deltam_external_200k": ("DeltaM", "#e87ba4", "v", True),
    "peract2_orbital_new_external_only_miscal_base_200k": ("Base", "#008300", "P", True),
}
CONDITIONS = [
    ("calibrated", "clean"), ("supported-e3deg-t1cm", "3°/1cm"),
    ("unknown-e5deg-t5cm", "5/5"), ("unknown-e10deg-t10cm", "10/10"),
    ("unknown-e15deg-t15cm", "15/15"), ("unknown-e20deg-t20cm", "20/20"),
    ("unknown-e30deg-t30cm", "30/30"), ("unknown-e40deg-t40cm", "40/40"),
    ("unknown-e50deg-t50cm", "50/50"), ("unknown-e60deg-t60cm", "60/60"),
]


def collect():
    per_task = defaultdict(dict)
    for manifest in ROOT.glob("**/*.manifest.json"):
        campaign = manifest.relative_to(ROOT).parts[0]
        if "heldout" not in campaign:
            continue
        meta = json.loads(manifest.read_text())
        arm = ARMS.get(Path(meta.get("checkpoint", "")).parent.name)
        if arm is None:
            continue
        calibration = meta.get("calibration_id", "")
        condition = "calibrated" if calibration == "calibrated" else re.sub(
            r"^seen-g\d-medium-", "", calibration).removesuffix("-v01")
        result_file = manifest.with_name(manifest.name.replace(".manifest", ""))
        # A cell that died mid-run leaves only a .progress.json behind.
        if not result_file.exists():
            continue
        results = json.loads(result_file.read_text())
        for task, cell in results.items():
            if isinstance(cell, dict) and "mean" in cell:
                per_task[(arm[0], condition)][task] = cell["mean"]
    return per_task


def main():
    per_task = collect()
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    xs = range(len(CONDITIONS))
    for label, color, marker, complete in ARMS.values():
        points = [(x, per_task[(label, key)]) for x, (key, _) in zip(xs, CONDITIONS) if per_task[(label, key)]]
        means = [(x, 100 * sum(tasks.values()) / len(tasks), len(tasks)) for x, tasks in points]
        ax.plot(
            [x for x, _, _ in means], [m for _, m, _ in means],
            color=color, marker=marker, markersize=6, linewidth=2,
            linestyle="-" if complete else "--", label=label,
            markeredgecolor="#fcfcfb", markeredgewidth=1.2, clip_on=False, zorder=3,
        )
        # Ring a cell that is short of the full task set so the mean is not read as final.
        for x, mean, ntask in means:
            if ntask != EXPECTED_TASKS:
                ax.plot(x, mean, marker="o", markersize=13, markerfacecolor="none",
                        markeredgecolor=color, markeredgewidth=1.0, zorder=2)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([label for _, label in CONDITIONS])
    ax.set_xlabel("External-camera miscalibration (rotation° / translation cm)", color="#52514e")
    ax.set_ylabel("Mean task success (%)", color="#52514e")
    ax.set_title("Task-heldout robustness to camera miscalibration\n13 tasks × 100 rollouts, G1–G6 heldout groups",
                 color="#0b0b0b", fontsize=13, loc="left")
    ax.set_ylim(20, 85)
    ax.grid(axis="y", color="#e6e5e1", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c9c8c3")
    ax.tick_params(colors="#52514e", length=0)
    ax.legend(frameon=False, ncol=3, loc="lower left", labelcolor="#52514e", fontsize=9)
    # Dashed arms were only ever run at the conditions their markers show.
    fig.text(0.125, 0.005, "Dashed = Rope3D arms, sampled at 6 of 10 conditions.  Hollow ring = 12/13 tasks.",
             color="#84837d", fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, facecolor=fig.get_facecolor())
    print(OUT)


if __name__ == "__main__":
    main()
