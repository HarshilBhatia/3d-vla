"""Plot the EEF trajectory from an orbital rollout episode."""
import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.processing.rlbench_utils import CustomUnpickler, keypoint_discovery


def load_demo(ep_path):
    path = os.path.join(ep_path, "low_dim_obs.pkl")
    with open(path, "rb") as f:
        return CustomUnpickler(f).load()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ep_path", help="Path to episode directory")
    p.add_argument("--out", default=None, help="Save plot to file instead of showing")
    p.add_argument("--all-frames", action="store_true", help="Plot every frame, not just keyframes")
    args = p.parse_args()

    demo = load_demo(args.ep_path)
    key_frames = keypoint_discovery(demo, bimanual=False)
    key_frames.insert(0, 0)

    # Full trajectory (every frame)
    all_pos = np.array([demo[i].gripper_pose[:3] for i in range(len(demo))], dtype=np.float32)

    # Keyframe positions
    kf_pos = np.array([demo[k].gripper_pose[:3] for k in key_frames], dtype=np.float32)
    kf_open = np.array([demo[k].gripper_open for k in key_frames], dtype=np.float32)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    if args.all_frames:
        ax.plot(all_pos[:, 0], all_pos[:, 1], all_pos[:, 2],
                color="lightsteelblue", linewidth=0.8, alpha=0.5, label="all frames")

    # Keyframe path
    ax.plot(kf_pos[:, 0], kf_pos[:, 1], kf_pos[:, 2],
            color="steelblue", linewidth=1.5, zorder=2)

    # Keyframe scatter colored by gripper open
    closed = kf_open < 0.5
    ax.scatter(kf_pos[closed, 0], kf_pos[closed, 1], kf_pos[closed, 2],
               c="crimson", s=60, zorder=3, label="gripper closed")
    ax.scatter(kf_pos[~closed, 0], kf_pos[~closed, 1], kf_pos[~closed, 2],
               c="limegreen", s=60, zorder=3, label="gripper open")

    # Label keyframe indices
    for i, (k, pos) in enumerate(zip(key_frames, kf_pos)):
        ax.text(pos[0], pos[1], pos[2], f" {i}", fontsize=8, color="navy")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(os.path.basename(args.ep_path))
    ax.legend(fontsize=8)

    plt.tight_layout()
    if args.out:
        plt.savefig(args.out, dpi=150)
        print(f"Saved to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
