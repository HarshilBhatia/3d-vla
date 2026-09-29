"""Rerun view of the G7 viewpoint-sweep cameras against the G1-G6 training cameras and the captured scene."""
import argparse
import json
from pathlib import Path

import numpy as np
import rerun as rr

TRAIN_COLOUR, G7_COLOUR = (150, 150, 150), (255, 255, 255)
PALETTE = [(255, 190, 60), (255, 105, 70), (210, 70, 200), (90, 150, 255), (80, 220, 170),
           (240, 230, 90), (255, 140, 180), (120, 200, 255), (170, 120, 255), (120, 255, 120)]


def frustum(p, R, scale=0.2):
    c = np.array([[0, 0, 0], [.4, .3, 1], [-.4, .3, 1], [-.4, -.3, 1], [.4, -.3, 1]]) * scale
    w = (R @ c.T).T + p
    return [[w[0], w[i]] for i in (1, 2, 3, 4)] + [[w[1], w[2]], [w[2], w[3]], [w[3], w[4]], [w[4], w[1]]]


def log_pair(path, group, colour):
    for side in ("left", "right"):
        p, R = np.array(group[side]["pos"]), np.array(group[side]["R"])
        rr.log(f"{path}/{side}/frustum", rr.LineStrips3D(frustum(p, R), colors=[colour]), static=True)
        rr.log(f"{path}/{side}/centre", rr.Points3D([p], colors=[colour], radii=0.03,
                                                    labels=[group["group"]]), static=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-cameras", default="instructions/orbital_cameras_grouped.json")
    ap.add_argument("--sweep-cameras", default="instructions/orbital_cameras_g7_sweep_v01.json")
    ap.add_argument("--scene", default="docs/handbook/eval/figures/g7_pick_laptop_scene_pointcloud.npz")
    ap.add_argument("--captures", default="docs/research/experiments/2026-09-29/figures/viewpoint-sweep/captures/bimanual_pick_laptop",
                    help="per-group npz from capture_g7_sweep_views.py; logged when present")
    ap.add_argument("--out", default="docs/research/experiments/2026-09-29/figures/viewpoint-sweep/g7_viewpoint_sweep.rrd")
    ap.add_argument("--no-sweep", action="store_true", help="G1-G7 and the scene only")
    args = ap.parse_args()

    rr.init("g7_viewpoint_sweep", spawn=False)
    rr.save(args.out)
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    scene = np.load(args.scene)["points"].reshape(-1, 3)
    lo, hi = np.percentile(scene, [1, 99], axis=0)
    scene = scene[(scene >= lo).all(1) & (scene <= hi).all(1)]
    rr.log("world/scene_g7", rr.Points3D(scene, colors=[(200, 200, 200)], radii=0.004), static=True)

    for g in json.loads(Path(args.train_cameras).read_text()):
        colour = G7_COLOUR if g["group"] == "G7" else TRAIN_COLOUR
        log_pair(f"world/{'g7' if g['group'] == 'G7' else 'train'}/{g['group']}", g, colour)

    sweep = [] if args.no_sweep else [g for g in json.loads(Path(args.sweep_cameras).read_text()) if g["group"] != "G7"]
    for i, g in enumerate(sweep):
        colour = PALETTE[i % len(PALETTE)]
        log_pair(f"world/sweep/{g['group']}", g, colour)
        cap = Path(args.captures) / f"{g['group']}.npz"
        if not cap.exists():
            continue
        d = np.load(cap)
        for ci, side in enumerate(("left", "right")):
            rr.log(f"world/sweep/{g['group']}/{side}/cloud",
                   rr.Points3D(d["points"][ci], colors=[colour], radii=0.004), static=True)
            rr.log(f"images/{g['group']}/{side}", rr.Image(d["rgb"][ci]), static=True)
        for ci, cam in ((2, "wrist_left"), (3, "wrist_right")):
            if len(d["rgb"]) > ci:
                rr.log(f"images/{g['group']}/{cam}", rr.Image(d["rgb"][ci]), static=True)
        print("logged capture", cap)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
