"""
Collect RLBench demos and save only low_dim_obs.pkl (no images).

Output layout (compatible with _load_orbital_rollout_demos):
    {save-path}/{task}/GT/episode_{N}/
        low_dim_obs.pkl
        variation.txt

Much faster than collect.py — no cameras, no image saving.

Example:
    xvfb-run -a python scripts/orbital_cameras/collect_low_dim_only.py \\
        --tasks close_jar open_drawer --n-episodes 30 \\
        --save-path data/orbital_low_dim
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from data.generation.orbital.collection import (
    collect_one_episode_low_dim,
    make_obs_config_low_dim,
    save_low_dim_episode,
)

GT_GROUP = "GT"


def parse_args():
    p = argparse.ArgumentParser(description="Collect low_dim_obs.pkl only (no images).")
    p.add_argument("--tasks",      nargs="+", required=True,
                   help="RLBench task name(s) (e.g. close_jar open_drawer)")
    p.add_argument("--n-episodes", type=int, default=30)
    p.add_argument("--ep-start",   type=int, default=0,
                   help="Starting episode index for parallel collection (default: 0)")
    p.add_argument("--save-path",  default="data/orbital_low_dim")
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

    obs_config  = make_obs_config_low_dim()
    action_mode = MoveArmThenGripper(JointVelocity(), Discrete())

    env = OrbitalEnvironment(
        action_mode=action_mode,
        obs_config=obs_config,
        headless=True,
        robot_setup="panda",
    )
    env.launch()

    for task_str in args.tasks:
        task_class   = task_file_to_task_class(task_str)
        task_env     = env.get_task(task_class)
        n_variations = task_env.variation_count()
        base_path    = os.path.join(args.save_path, task_str, GT_GROUP)
        ep_end       = args.ep_start + args.n_episodes
        ep_times     = []

        print("[INFO] task={} n_variations={} episodes={}-{}".format(
            task_str, n_variations, args.ep_start, ep_end - 1))

        for ep_idx in range(args.ep_start, ep_end):
            ep_path = os.path.join(base_path, "episode_{}".format(ep_idx))
            if os.path.exists(ep_path):
                print("[SKIP] {}".format(ep_path))
                continue

            variation = ep_idx % n_variations
            task_env.set_variation(variation)

            t0 = time.perf_counter()
            demo, timing = collect_one_episode_low_dim(task_env)
            if demo is None:
                continue

            save_low_dim_episode(demo, ep_path, variation)
            t_total = time.perf_counter() - t0
            ep_times.append(t_total)
            avg       = sum(ep_times) / len(ep_times)
            remaining = (ep_end - ep_idx - 1) * avg
            print("[SAVED] {} | reset={:.2f}s total={:.2f}s avg={:.2f}s eta={:.1f}s".format(
                ep_path, timing["reset"], t_total, avg, remaining))

    env.shutdown()
    print("[DONE]")


if __name__ == "__main__":
    main()
