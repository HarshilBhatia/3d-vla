"""Render one stored test scene from every G7 viewpoint-sweep group and save orbital RGB plus world point clouds."""
import argparse
import json
from pathlib import Path

import numpy as np


def capture_group(args, group, out):
    from evaluation.online.rlbench.backends.orbital_bimanual import RLBenchEnv
    from evaluation.online.rlbench.backends.bimanual import task_file_to_task_class
    from evaluation.online.rlbench.stored_demos import get_stored_demos
    env = RLBenchEnv(data_path=args.data_path, task_str=args.task, image_size=(args.image_size, args.image_size),
                     headless=True, cameras_file=args.cameras_file, spawn_camera_group=group)
    try:
        env._launch_env()
        task = env.env.get_task(task_file_to_task_class(args.task))
        task.set_variation(args.variation)
        demos = get_stored_demos(amount=-1, dataset_root=args.data_path, variation_number=args.variation,
                                 task_name=args.task, random_selection=False, from_episode_number=0)
        _, obs = task.reset_to_demo(demos[args.demo])
        rgb, pcd, _ = env.get_rgb_pcd_gripper_from_obs(obs)
        rgb = (rgb[0].permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
        pcd = pcd[0, :2].float().cpu().numpy()
        rng = np.random.default_rng(0)
        clouds = []
        for cam in range(2):
            xyz = pcd[cam].transpose(1, 2, 0).reshape(-1, 3)
            xyz = xyz[np.isfinite(xyz).all(1) & (np.linalg.norm(xyz, axis=1) > 1e-4)]
            clouds.append(xyz[rng.choice(len(xyz), min(len(xyz), args.max_points), replace=False)])
        np.savez_compressed(out, rgb=rgb, points=np.array(clouds, dtype=np.float32))
        if args.png:
            from PIL import Image
            Image.fromarray(np.concatenate(list(rgb), axis=1)).save(out.with_suffix(".png"))
        print(f"saved {out}")
    finally:
        env.env.shutdown()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", default="/grogu/datasets/hbhatia/peract2_test/peract2_test")
    p.add_argument("--task", default="bimanual_pick_laptop")
    p.add_argument("--variation", type=int, default=0)
    p.add_argument("--demo", type=int, default=0)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--max-points", type=int, default=8000)
    p.add_argument("--cameras-file", default="instructions/orbital_cameras_g7_sweep_v01.json")
    p.add_argument("--out-dir", default="docs/figures/g7_sweep_captures")
    p.add_argument("--tasks", nargs="*", default=None, help="overrides --task; one capture per task")
    p.add_argument("--groups", nargs="*", default=None, help="subset of groups in --cameras-file")
    p.add_argument("--png", action="store_true", help="also write a left|right|wrist_l|wrist_r RGB strip")
    args = p.parse_args()
    groups = [g["group"] for g in json.loads(Path(args.cameras_file).read_text())]
    groups = [g for g in groups if args.groups is None or g in args.groups]
    for task in args.tasks or [args.task]:
        args.task = task
        out_dir = Path(args.out_dir) / task
        out_dir.mkdir(parents=True, exist_ok=True)
        for g in groups:
            try:
                capture_group(args, g, out_dir / f"{g}.npz")
            except Exception as e:
                print(f"FAILED {task} {g}: {e!r}")


if __name__ == "__main__":
    main()
