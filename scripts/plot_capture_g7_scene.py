"""Capture one real G7 RLBench scene and plot its point cloud with camera shifts.

The capture uses the same fixed orbital-bimanual environment and depth-to-world
projection as evaluation.  It intentionally writes PNG/NPZ only (no PDF).
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

def capture(args):
    from evaluation.online.rlbench.backends.orbital_bimanual import RLBenchEnv
    from evaluation.online.rlbench.backends.bimanual import task_file_to_task_class
    from evaluation.online.rlbench.stored_demos import get_stored_demos
    env = RLBenchEnv(
        data_path=args.data_path,
        task_str=args.task,
        image_size=(args.image_size, args.image_size),
        headless=True,
        cameras_file=args.cameras_file,
        spawn_camera_group=args.camera_group,
        calibration_registry=None,
    )
    try:
        env._launch_env()
        task = env.env.get_task(task_file_to_task_class(args.task))
        task.set_variation(args.variation)
        demos = get_stored_demos(
            amount=-1, dataset_root=args.data_path, variation_number=args.variation,
            task_name=args.task, random_selection=False, from_episode_number=0,
        )
        if not demos:
            raise RuntimeError("No stored demos found for requested task/variation")
        _, obs = task.reset_to_demo(demos[args.demo])
        _, pcd, _ = env.get_rgb_pcd_gripper_from_obs(obs)
        pcd = pcd[0].detach().float().cpu().numpy()  # (4, 3, H, W), world frame
        exts = np.stack([
            env._orbital_extrinsics["left_extrinsics"],
            env._orbital_extrinsics["right_extrinsics"],
            obs.misc["wrist_left_camera_extrinsics"],
            obs.misc["wrist_right_camera_extrinsics"],
        ])
        rng = np.random.default_rng(0)
        clouds = []
        for cam in range(pcd.shape[0]):
            xyz = pcd[cam].transpose(1, 2, 0).reshape(-1, 3)
            valid = np.isfinite(xyz).all(1) & (np.linalg.norm(xyz, axis=1) > 1e-4)
            xyz = xyz[valid]
            if len(xyz) > args.max_points_per_camera:
                xyz = xyz[rng.choice(len(xyz), args.max_points_per_camera, replace=False)]
            clouds.append(xyz)
        out = Path(args.output_npz)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, points=np.array(clouds, dtype=np.float32), extrinsics=exts.astype(np.float32))
        print(f"saved {out} with {sum(map(len, clouds))} points")
    finally:
        env.env.shutdown()


def plot(args):
    data = np.load(args.output_npz)
    points, exts = data["points"], data["extrinsics"]
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    colors = ["#377eb8", "#e41a1c", "#4daf4a", "#984ea3"]
    labels = ["orbital left", "orbital right", "wrist left", "wrist right"]
    # Depth sensors occasionally return a few far-plane pixels.  Keep the
    # measured cloud, but suppress only the outer 1% per coordinate so the
    # camera translations remain visible at scene scale.
    merged = points.reshape(-1, 3)
    lo, hi = np.percentile(merged, [1, 99], axis=0)
    for i, xyz in enumerate(points):
        xyz = xyz[(xyz >= lo).all(1) & (xyz <= hi).all(1)]
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=0.8, alpha=0.25,
                   color=colors[i], label=labels[i])
    centers = exts[:, :3, 3]
    ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], marker="^", s=65,
               c=colors, edgecolor="black", linewidth=0.5)
    d = np.array([0.03489949670250096, 0.9993908270190957, 0.0])
    for level in (0.10, 0.20, 0.40, 0.60):
        shifted = centers.copy()
        shifted[0] += level * d
        shifted[1] -= level * d
        ax.plot([centers[0, 0], shifted[0, 0]], [centers[0, 1], shifted[0, 1]],
                [centers[0, 2], shifted[0, 2]], "--", color="#e41a1c", alpha=0.7)
        ax.plot([centers[1, 0], shifted[1, 0]], [centers[1, 1], shifted[1, 1]],
                [centers[1, 2], shifted[1, 2]], "--", color="#377eb8", alpha=0.7,
                label=f"opposing shift {int(level*100)} cm" if level == 0.10 else None)
        ax.scatter(shifted[:2, 0], shifted[:2, 1], shifted[:2, 2], marker="x",
                   c=[colors[0], colors[1]], s=28)
    ax.set_xlabel("world X (m)"); ax.set_ylabel("world Y (m)"); ax.set_zlabel("world Z (m)")
    ax.set_title("Real G7 pick-laptop scene point cloud + opposing camera translations")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_box_aspect((1, 1, 0.75))
    pad = 0.12
    ax.set_xlim(lo[0] - pad, hi[0] + pad)
    ax.set_ylim(lo[1] - pad, hi[1] + pad)
    ax.set_zlim(lo[2] - pad, hi[2] + pad)
    fig.tight_layout()
    Path(args.output_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, dpi=220)
    print(f"saved {args.output_png}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--capture", action="store_true")
    p.add_argument("--data-path", default="/grogu/datasets/hbhatia/peract2_test/peract2_test")
    p.add_argument("--task", default="bimanual_pick_laptop")
    p.add_argument("--variation", type=int, default=0)
    p.add_argument("--demo", type=int, default=0)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--max-points-per-camera", type=int, default=6000)
    p.add_argument("--camera-group", default="G7")
    p.add_argument("--cameras-file", default="instructions/orbital_cameras_grouped.json")
    p.add_argument("--output-npz", default="docs/figures/g7_pick_laptop_scene_pointcloud.npz")
    p.add_argument("--output-png", default="docs/figures/g7_pick_laptop_scene_pointcloud.png")
    a = p.parse_args()
    if a.capture:
        capture(a)
    plot(a)
