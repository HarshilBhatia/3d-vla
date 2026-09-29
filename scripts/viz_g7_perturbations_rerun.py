"""Rerun view of the G7 scene point cloud under extreme external-camera perturbations."""
import argparse, json
import numpy as np
import rerun as rr

CAM_ORDER = ["orbital_left", "orbital_right", "wrist_left", "wrist_right"]
# One hue per perturbation magnitude; clean is white so the true geometry reads as reference.
PALETTE = [(235,235,235),(255,190,60),(255,105,70),(210,70,200),(90,150,255),(80,220,170)]


def load_registry(path):
    reg = json.loads(open(path).read())
    return reg["camera_order"], reg["realizations"]


def frustum(T, scale=0.25):
    """Camera-frame corner rays -> world, for a simple wireframe frustum."""
    c = np.array([[0,0,0],[ .4, .3,1],[-.4, .3,1],[-.4,-.3,1],[ .4,-.3,1]]) * scale
    w = (T[:3,:3] @ c.T + T[:3,3:4]).T
    return [[w[0],w[i]] for i in (1,2,3,4)] + [[w[1],w[2]],[w[2],w[3]],[w[3],w[4]],[w[4],w[1]]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="docs/handbook/eval/figures/g7_pick_laptop_scene_pointcloud.npz")
    ap.add_argument("--registry", default="instructions/eval_calibrations_g7_extreme_opposite_v01.json")
    ap.add_argument("--out", default="docs/handbook/eval/figures/g7_perturbations.rrd")
    ap.add_argument("--cameras", nargs="*", default=["orbital_left","orbital_right"],
                    help="which cameras to displace; wrists are never miscalibrated in this setup")
    args = ap.parse_args()

    d = np.load(args.scene)
    pts, extr = d["points"].astype(np.float64), d["extrinsics"].astype(np.float64)
    order, reals = load_registry(args.registry)

    rr.init("g7_camera_perturbations", spawn=False)
    rr.save(args.out)
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    for li, (level, spec) in enumerate(reals.items()):
        colour = PALETTE[li % len(PALETTE)]
        rr.set_time("perturbation", sequence=li)
        for ci, cam in enumerate(order):
            T = np.array(spec["transforms"][cam], dtype=np.float64)
            displaced = cam in args.cameras
            p = (T[:3,:3] @ pts[ci].T + T[:3,3:4]).T if displaced else pts[ci]
            rr.log(f"world/{level}/{cam}/points",
                   rr.Points3D(p, colors=np.tile(colour,(len(p),1)), radii=0.006))
            Tw = T @ extr[ci]
            rr.log(f"world/{level}/{cam}/pose", rr.Points3D([Tw[:3,3]], colors=[colour], radii=0.05))
            rr.log(f"world/{level}/{cam}/frustum",
                   rr.LineStrips3D(frustum(Tw), colors=[colour]))
        shift = {c: float(np.linalg.norm(np.array(spec["transforms"][c])[:3,3])) for c in args.cameras}
        print(f"{level:26} " + "  ".join(f"{c}={v*100:6.1f}cm" for c,v in shift.items()))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
