"""pick_laptop external-camera renders at every G7 sweep pose, annotated with each model's success there."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CAPS = Path("docs/research/experiments/2026-09-29/figures/viewpoint-sweep/captures/bimanual_pick_laptop")
RES = Path("/grogu/datasets/hbhatia/3dfa_online_eval_20rollouts/g7_viewpoint_sweep_v01")
OUT = Path("docs/research/experiments/2026-09-29/figures/g7_sweep_pick_laptop.png")
GROUPS = [("G7", "g7", "anchor · 43/48/1.33"), ("G7_el72", "g7_el72", "45/72/1.31"), ("G7_el85", "g7_el85", "45/85/1.31"),
          ("G7_az25", "g7_az25", "25/47.5/1.31"), ("G7_az65", "g7_az65", "65/47.5/1.31"), ("G7_az90", "g7_az90", "90/47.5/1.31"),
          ("G7_r180", "g7_r180", "45/47.5/1.80"), ("G7_slight_az56", "g7_slight_az56", "slight · 56/47.5/1.31"),
          ("G7_slight_el66_r145", "g7_slight_el66_r145", "slight · 45/66/1.45"),
          ("G7_extreme_r350", "g7_extreme_r350", "extreme · 45/35/3.50"),
          ("G7_extreme_az150_r250", "g7_extreme_az150_r250", "extreme · 150/30/2.50")]
ARMS = [("Base", "base_s160k"), ("VP", "video_pooled_s084k"), ("FP", "video_fullpatch_best")]
TEXT, MUTED, BAD = "#0b0b0b", "#52514e", "#c0392b"


def success(arm, group):
    f = RES / arm / group / "calibrated" / "results_bimanual_pick_laptop.json"
    return 100 * json.load(open(f))["bimanual_pick_laptop"]["mean"] if f.exists() else float("nan")


def main():
    fig, axes = plt.subplots(3, 4, figsize=(16, 8.6), facecolor="#fcfcfb")
    for ax, (cap, res, pose) in zip(axes.flat, GROUPS):
        rgb = np.load(CAPS / f"{cap}.npz")["rgb"]
        ax.imshow(np.concatenate([rgb[0], np.full((256, 6, 3), 252, np.uint8), rgb[1]], axis=1))
        ax.set_xticks([]), ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        scores = [(name, success(arm, res)) for name, arm in ARMS]
        ax.set_title(f"{res.replace('g7_', '').replace('g7', 'G7')}  ({pose})", fontsize=9.5, color=TEXT, pad=16)
        x = 0.0
        for name, v in scores:
            ax.text(x, 1.015, f"{name} {v:.0f}", transform=ax.transAxes, fontsize=9, ha="left", va="bottom",
                    color=BAD if v < 20 else MUTED, fontweight="bold" if v < 20 else "normal")
            x += 0.24
    last = axes.flat[-1]
    last.axis("off")
    last.text(0.02, 0.55, "Each tile: orbital_left | orbital_right render.\nTitle: az° / el° / radius m.\n"
              "Numbers: pick_laptop success (%), 20 rollouts,\ncalibrated extrinsics. Red = below 20%.\n\n"
              "Base 160k · VP = VideoPooled 84k · FP = FullPatch best", fontsize=9.5, color=MUTED, va="center")
    fig.suptitle("G7 viewpoint sweep — pick_laptop: what each external camera sees, and who succeeds", fontsize=12, color=TEXT)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
