"""Collect seeded bimanual orbital episodes (every frame kept) for the dense PerAct2 dataset."""

import argparse
import os
import random
import shutil
import sys
import time
import zlib

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from data.generation.orbital.collection import (
    collect_one_episode,
    load_group_cameras,
    make_obs_config,
    save_orbital_episode,
)
from data.generation.orbital.constants import PERACT2_PROFILE


def parse_args():
    p = argparse.ArgumentParser(description="Collect seeded orbital PerAct2 episodes.")
    p.add_argument("--task", required=True)
    p.add_argument("--groups", nargs="+", required=True)
    p.add_argument("--ep-start", type=int, default=0)
    p.add_argument("--n-episodes", type=int, default=30)
    p.add_argument("--save-path", required=True)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--fov-deg", type=float, default=60.0)
    p.add_argument("--cameras-file", default="instructions/orbital_cameras_grouped.json")
    p.add_argument("--max-seed-retries", type=int, default=5)
    return p.parse_args()


def episode_seed(task, group, ep_idx, retry):
    return zlib.crc32("{}/{}/{}/{}".format(task, group, ep_idx, retry).encode()) & 0x7FFFFFFF


def main():
    args = parse_args()
    profile = PERACT2_PROFILE

    from data.generation.orbital.scene import OrbitalEnvironment

    env = OrbitalEnvironment(
        action_mode=profile.make_action_mode(),
        obs_config=make_obs_config(args.image_size, profile),
        headless=True,
        robot_setup=profile.robot_setup,
    )
    env.launch()
    task_env = env.get_task(profile.task_loader(args.task))
    n_variations = task_env.variation_count()
    scene = env._scene
    print("[INFO] task={} groups={} n_variations={}".format(args.task, args.groups, n_variations))

    for group in args.groups:
        cam_left, cam_right = load_group_cameras(args.cameras_file, group)
        base_path = os.path.join(args.save_path, args.task, group)
        for ep_idx in range(args.ep_start, args.ep_start + args.n_episodes):
            ep_path = os.path.join(base_path, "episode_{}".format(ep_idx))
            if os.path.exists(os.path.join(ep_path, "seed.txt")):
                print("[SKIP] {} exists".format(ep_path))
                continue
            # Same variation cycling as the original orbital collection (collect.py).
            variation = ep_idx % n_variations
            t0 = time.perf_counter()
            for retry in range(args.max_seed_retries):
                seed = episode_seed(args.task, group, ep_idx, retry)
                np.random.seed(seed)
                random.seed(seed)
                task_env.set_variation(variation)
                demo, extr, timing = collect_one_episode(
                    task_env, scene, cam_left, cam_right, args.image_size, args.fov_deg)
                if demo is not None:
                    break
            if demo is None:
                print("[ERROR] {} failed after {} seeds".format(ep_path, args.max_seed_retries))
                continue
            tmp_path = ep_path + ".tmp"
            if os.path.exists(tmp_path):
                shutil.rmtree(tmp_path)
            save_orbital_episode(demo, tmp_path, group, extr, profile)
            with open(os.path.join(tmp_path, "variation.txt"), "w") as f:
                f.write("{}\n".format(variation))
            # seed.txt is written last and marks the episode complete.
            with open(os.path.join(tmp_path, "seed.txt"), "w") as f:
                f.write("{} {}\n".format(seed, retry))
            if os.path.exists(ep_path):
                shutil.rmtree(ep_path)
            os.rename(tmp_path, ep_path)
            print("[SAVED] {} var={} seed={} frames={} {:.1f}s".format(
                ep_path, variation, seed, len(demo), time.perf_counter() - t0))

    env.shutdown()
    print("[DONE]")


if __name__ == "__main__":
    main()
