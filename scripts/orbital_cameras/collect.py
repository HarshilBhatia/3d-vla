"""
Collect RLBench PerAct rollouts using orbital camera groups.

Each episode records 4 cameras:
  [orbital_left, orbital_right, over_shoulder_left, over_shoulder_right]

Normal mode  → saves raw episodes to --save-path/{task}/{group}/episode_{N}/
               Variations cycle sequentially across episodes (like generate.py).
Video mode   → saves a single episode as MP4 for pipeline validation.

Example (headless):
    xvfb-run -a python scripts/orbital_cameras/collect.py \\
        --task close_jar --groups G1 --n-episodes 30 \\
        --save-path data/orbital_rollouts \\
        --cameras-file instructions/orbital_cameras_grouped.json

    # With episode offset (e.g. second SLURM node):
    xvfb-run -a python scripts/orbital_cameras/collect.py \\
        --task close_jar --groups G1 --n-episodes 30 --ep-start 30 \\
        --save-path data/orbital_rollouts \\
        --cameras-file instructions/orbital_cameras_grouped.json
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from data.generation.orbital.collection import (
    collect_one_episode,
    load_group_cameras,
    make_obs_config,
    save_debug_video,
    save_orbital_episode,
)


def parse_args():
    p = argparse.ArgumentParser(description="Collect RLBench orbital rollouts.")
    p.add_argument("--task",         required=True,
                   help="RLBench task name (e.g. close_jar)")
    p.add_argument("--groups",       nargs="+", required=True,
                   help="One or more camera groups (e.g. G1 G2 G3). "
                        "All groups are collected in a single CoppeliaSim session.")
    p.add_argument("--n-episodes",   type=int, default=30,
                   help="Number of episodes to collect (default: 30)")
    p.add_argument("--ep-start",     type=int, default=0,
                   help="Starting episode index, for parallel nodes writing to "
                        "the same directory (default: 0)")
    p.add_argument("--save-path",    default="data/orbital_rollouts",
                   help="Root directory for raw episode output")
    p.add_argument("--image-size",   type=int, default=256)
    p.add_argument("--fov-deg",      type=float, default=60.0,
                   help="FOV for orbital cameras (degrees)")
    p.add_argument("--cameras-file", default="instructions/orbital_cameras_grouped.json",
                   help="Path to orbital_cameras_grouped.json")
    p.add_argument("--video-only",   action="store_true",
                   help="Debug: collect 1 episode, save MP4")
    p.add_argument("--video-dir",    default="debug_videos",
                   help="Output directory for debug MP4s (used with --video-only)")
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

    from data.generation.orbital.scene import OrbitalEnvironment

    obs_config  = make_obs_config(args.image_size)
    action_mode = MoveArmThenGripper(JointVelocity(), Discrete())

    env = OrbitalEnvironment(
        action_mode=action_mode,
        obs_config=obs_config,
        headless=True,
        robot_setup="panda",
    )
    env.launch()

    task_class  = task_file_to_task_class(args.task)
    task_env    = env.get_task(task_class)
    n_variations = task_env.variation_count()
    scene       = env._scene

    print("[INFO] task={} groups={} n_variations={} mode={}".format(
        args.task, args.groups, n_variations,
        "video" if args.video_only else "collect"))

    for group in args.groups:
        cam_left, cam_right = load_group_cameras(args.cameras_file, group)

        if args.video_only:
            video_out = os.path.join(
                args.video_dir, "{}_{}.mp4".format(args.task, group))
            if os.path.exists(video_out):
                print("[SKIP] {} already exists.".format(video_out))
                continue

            task_env.set_variation(0)
            demo, orbital_extrinsics, timing = collect_one_episode(
                task_env, scene, cam_left, cam_right,
                args.image_size, args.fov_deg,
            )
            if demo is None:
                continue

            t_collect = timing["reset"] + timing["sensors"] + timing["demos"] + timing["cleanup"]
            print("[TIME] reset={:.2f}s sensors={:.2f}s demos={:.2f}s cleanup={:.2f}s steps={}".format(
                timing["reset"], timing["sensors"], timing["demos"], timing["cleanup"], len(demo)))

            t0 = time.perf_counter()
            save_debug_video(demo, video_out, args.image_size)
            print("[TIME] video={:.2f}s total={:.2f}s".format(
                time.perf_counter() - t0, t_collect + time.perf_counter() - t0))

        else:
            base_path = os.path.join(args.save_path, args.task, group)

            ep_end = args.ep_start + args.n_episodes
            ep_times = []
            for ep_idx in range(args.ep_start, ep_end):
                ep_path = os.path.join(base_path, "episode_{}".format(ep_idx))
                if os.path.exists(ep_path):
                    print("[SKIP] {} already exists.".format(ep_path))
                    continue

                # Cycle variations sequentially, same as generate.py.
                variation = ep_idx % n_variations
                task_env.set_variation(variation)

                print("[INFO] {}/{} episode {} (variation {}/{})".format(
                    args.task, group, ep_idx, variation, n_variations))

                demo, orbital_extrinsics, timing = collect_one_episode(
                    task_env, scene, cam_left, cam_right,
                    args.image_size, args.fov_deg,
                )
                if demo is None:
                    continue

                t_collect = timing["reset"] + timing["sensors"] + timing["demos"] + timing["cleanup"]
                print("[TIME] reset={:.2f}s sensors={:.2f}s demos={:.2f}s cleanup={:.2f}s steps={}".format(
                    timing["reset"], timing["sensors"], timing["demos"], timing["cleanup"], len(demo)))

                print("[STEP] Saving episode to {}...".format(ep_path))
                t0 = time.perf_counter()
                save_orbital_episode(demo, ep_path, group, orbital_extrinsics)
                # Record which variation this episode used.
                with open(os.path.join(ep_path, "variation.txt"), "w") as f:
                    f.write("{}\n".format(variation))
                t_save = time.perf_counter() - t0

                t_total = t_collect + t_save
                ep_times.append(t_total)
                avg       = sum(ep_times) / len(ep_times)
                remaining = (ep_end - ep_idx - 1) * avg
                print("[SAVED] {} | collect={:.1f}s save={:.1f}s total={:.1f}s avg={:.1f}s eta={:.1f}s".format(
                    ep_path, t_collect, t_save, t_total, avg, remaining))

    env.shutdown()
    print("[DONE]")


if __name__ == "__main__":
    main()
