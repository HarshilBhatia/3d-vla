"""
Collect RLBench PerAct rollouts using orbital camera groups.

Each episode records 3 cameras: [orbital_left, orbital_right, wrist]

Normal mode  → saves raw episodes to --save-path/{task}/{group}/episode_{N}/
Video mode   → saves a single episode as MP4 and a single-episode zarr
               for pipeline validation.

Example (headless):
    xvfb-run -a python scripts/rlbench/collect_orbital_rollouts.py \\
        --task close_jar --groups G1 G2 G3 --n-episodes 30 \\
        --save-path data/orbital_rollouts \\
        --cameras-file orbital_cameras_grouped.json

    # Debug / video mode:
    xvfb-run -a python scripts/rlbench/collect_orbital_rollouts.py \\
        --task close_jar --groups G1 --video-only \\
        --cameras-file orbital_cameras_grouped.json
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from data_generation.orbital.collection import (
    collect_one_episode,
    load_group_cameras,
    make_obs_config,
    save_debug_video,
    save_debug_zarr,
    save_orbital_episode,
)
from data_generation.orbital.scene import OrbitalEnvironment


def parse_args():
    p = argparse.ArgumentParser(description="Collect RLBench orbital rollouts.")
    p.add_argument("--task",         required=True,
                   help="RLBench task name (e.g. close_jar)")
    p.add_argument("--groups",       nargs="+", required=True,
                   help="One or more camera groups (e.g. G1 G2 G3). "
                        "All groups are collected in a single CoppeliaSim session.")
    p.add_argument("--n-episodes",   type=int, default=30,
                   help="Number of episodes per group (default: 30)")
    p.add_argument("--save-path",    default="data/orbital_rollouts",
                   help="Root directory for raw episode output")
    p.add_argument("--image-size",   type=int, default=256)
    p.add_argument("--fov-deg",      type=float, default=60.0,
                   help="FOV for orbital cameras in degrees (default: 60.0)")
    p.add_argument("--cameras-file", default="orbital_cameras_grouped.json",
                   help="Path to orbital_cameras_grouped.json")
    p.add_argument("--video-only",   action="store_true",
                   help="Debug: collect 1 episode per group, save MP4 + zarr")
    p.add_argument("--video-dir",    default="debug_videos",
                   help="Output directory for debug MP4s (--video-only)")
    p.add_argument("--variation",    type=int, default=0,
                   help="Task variation index (default: 0)")
    return p.parse_args()


def main():
    args = parse_args()

    try:
        from rlbench.action_modes.action_mode import MoveArmThenGripper
        from rlbench.action_modes.arm_action_modes import JointVelocity
        from rlbench.action_modes.gripper_action_modes import Discrete
        from rlbench.backend.utils import task_file_to_task_class
    except ImportError as e:
        sys.exit("[ERROR] RLBench import failed: {}\n"
                 "Set COPPELIASIM_ROOT etc. first.".format(e))

    env = OrbitalEnvironment(
        action_mode=MoveArmThenGripper(JointVelocity(), Discrete()),
        obs_config=make_obs_config(args.image_size),
        headless=True,
        robot_setup="panda",
    )
    env.launch()

    task_class = task_file_to_task_class(args.task)
    task_env   = env.get_task(task_class)
    task_env.set_variation(args.variation)
    scene      = env._scene

    print("[INFO] task={} groups={} mode={}".format(
        args.task, args.groups, "video" if args.video_only else "collect"))

    for group in args.groups:
        cam_left, cam_right = load_group_cameras(args.cameras_file, group)

        if args.video_only:
            video_out = os.path.join(args.video_dir, "{}_{}.mp4".format(args.task, group))
            zarr_path = video_out + ".zarr"
            if os.path.exists(zarr_path):
                print("[SKIP] {}/{} already exists.".format(args.task, group))
                continue

            demo, orbital_extrinsics, timing = collect_one_episode(
                task_env, scene, cam_left, cam_right, args.image_size, args.fov_deg)
            if demo is None:
                continue

            print("[TIME] reset={:.2f}s sensors={:.2f}s demos={:.2f}s "
                  "cleanup={:.2f}s steps={}".format(
                timing["reset"], timing["sensors"], timing["demos"],
                timing["cleanup"], len(demo)))

            t0 = time.perf_counter()
            save_debug_video(demo, video_out, args.image_size)
            t_video = time.perf_counter() - t0
            print("[TIME] video={:.2f}s ({} frames → {})".format(
                t_video, len(demo), video_out))

            t0 = time.perf_counter()
            save_debug_zarr(demo, zarr_path, group,
                            orbital_extrinsics=orbital_extrinsics,
                            image_size=args.image_size)
            t_zarr = time.perf_counter() - t0
            print("[TIME] zarr={:.2f}s → {}".format(t_zarr, zarr_path))

            t_collect = sum(timing[k] for k in ("reset", "sensors", "demos", "cleanup"))
            print("[TIME] total={:.2f}s (collect={:.2f}s save={:.2f}s)".format(
                t_collect + t_video + t_zarr, t_collect, t_video + t_zarr))

        else:
            base_path = os.path.join(args.save_path, args.task, group)

            ep_start = 0
            if os.path.exists(base_path):
                existing = [
                    d for d in os.listdir(base_path)
                    if d.startswith("episode_") and
                       os.path.isdir(os.path.join(base_path, d))
                ]
                ep_start = len(existing)
                if ep_start >= args.n_episodes:
                    print("[SKIP] {}/{} already has {} episodes.".format(
                        args.task, group, ep_start))
                    continue
                if ep_start > 0:
                    print("[RESUME] {}/{} from episode {}.".format(
                        args.task, group, ep_start))

            ep_times = []
            for ep_idx in range(ep_start, args.n_episodes):
                print("[INFO] {}/{} episode {}/{}".format(
                    args.task, group, ep_idx + 1, args.n_episodes))
                demo, orbital_extrinsics, timing = collect_one_episode(
                    task_env, scene, cam_left, cam_right, args.image_size, args.fov_deg)
                if demo is None:
                    continue

                t_collect = sum(timing[k] for k in ("reset", "sensors", "demos", "cleanup"))
                print("[TIME]  reset={:.2f}s sensors={:.2f}s demos={:.2f}s "
                      "cleanup={:.2f}s steps={}".format(
                    timing["reset"], timing["sensors"], timing["demos"],
                    timing["cleanup"], len(demo)))

                ep_path = os.path.join(base_path, "episode_{}".format(ep_idx))
                print("[STEP] Saving episode to {}...".format(ep_path))
                t0 = time.perf_counter()
                save_orbital_episode(demo, ep_path, group, orbital_extrinsics)
                t_save = time.perf_counter() - t0
                print("[TIME]  save={:.2f}s → {}".format(t_save, ep_path))

                t_total = t_collect + t_save
                ep_times.append(t_total)
                avg       = sum(ep_times) / len(ep_times)
                remaining = (args.n_episodes - ep_idx - 1) * avg
                print("[SAVED] {} | collect={:.1f}s save={:.1f}s "
                      "total={:.1f}s avg={:.1f}s eta={:.1f}s".format(
                    ep_path, t_collect, t_save, t_total, avg, remaining))

    env.shutdown()
    print("[DONE]")


if __name__ == "__main__":
    main()
