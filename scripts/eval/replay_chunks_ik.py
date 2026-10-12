"""Replay recorded demo poses through an eval controller in 8-step chunks; checks the controller, not a model."""
import argparse

import numpy as np
from rlbench.backend.exceptions import InvalidActionError

from evaluation.online.rlbench.backends.bimanual import IKMover, Mover, RLBenchEnv, task_file_to_task_class
from evaluation.online.rlbench.stored_demos import get_stored_demos


def pose(o):
    return np.stack([np.append(o.left.gripper_pose, o.left.gripper_open), np.append(o.right.gripper_pose, o.right.gripper_open)])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="bimanual_handover_item")
    p.add_argument("--data", default="/grogu/datasets/hbhatia/peract2_test/peract2_test")
    p.add_argument("--controller", default="ik")
    p.add_argument("--chunk", type=int, default=8)
    p.add_argument("--demos", type=int, default=3)
    p.add_argument("--substeps", type=int, default=10)
    p.add_argument("--no-stop-when-still", action="store_true")
    p.add_argument("--no-fallback", action="store_true")
    args = p.parse_args()
    env = RLBenchEnv(args.data, task_str=args.task, image_size=(128, 128), headless=True,
                     apply_cameras=("over_shoulder_left",), controller=args.controller)
    if args.controller == "ik":
        env.action_mode.arm_action_mode._max_substeps = args.substeps
        env.action_mode.arm_action_mode._stop_when_still = not args.no_stop_when_still
        if args.no_fallback:
            env.action_mode.arm_action_mode._fallback = None
    env._launch_env()
    task = env.env.get_task(task_file_to_task_class(args.task))
    task.set_variation(0)
    for d, demo in enumerate(get_stored_demos(amount=-1, dataset_root=args.data, variation_number=0, task_name=args.task)[:args.demos]):
        _, obs = task.reset_to_demo(demo)
        move = IKMover(task) if args.controller == "ik" else Mover(task, max_tries=1)
        T, fails, steps, reward, err = len(demo), 0, 0, 0.0, []
        try:
            move(pose(obs))
            print(f"demo {d}: IK to the current pose OK", flush=True)
        except InvalidActionError as e:
            print(f"demo {d}: IK to the current pose FAILED: {e}", flush=True)
        for t in range(0, T - 1, args.chunk):
            for k in range(1, args.chunk + 1):
                target = pose(demo[min(t + k, T - 1)])
                cur = pose(task._scene.get_observation())
                gap = np.linalg.norm(cur[:, :3] - target[:, :3], axis=-1).max()
                ang = np.degrees(2 * np.arccos(np.clip(np.abs((cur[:, 3:7] * target[:, 3:7]).sum(-1)), 0, 1))).max()
                fb0 = getattr(env.action_mode.arm_action_mode, "fallbacks", 0)
                try:
                    obs, reward, _ = move(target)
                except InvalidActionError:
                    fails += 1
                    if fails <= 3:
                        print(f"  fail at frame {t + k}: target {gap*100:.1f} cm, {ang:.1f} deg away", flush=True)
                if getattr(env.action_mode.arm_action_mode, "fallbacks", 0) > fb0 and env.action_mode.arm_action_mode.fallbacks <= 3:
                    print(f"  planner fallback at frame {t + k}: target {gap*100:.1f} cm, {ang:.1f} deg away", flush=True)
                steps += 1
                err.append(np.linalg.norm(pose(obs)[:, :3] - target[:, :3], axis=-1).max())
                if reward == 1:
                    break
            if reward == 1:
                break
        print(f"demo {d} [substeps={args.substeps} stop_when_still={not args.no_stop_when_still}]: frames {T}, steps {steps}, IK failures {fails}, planner fallbacks {getattr(env.action_mode.arm_action_mode, 'fallbacks', 0)}, success {reward == 1}, "
              f"tracking error median {np.median(err)*100:.1f} cm max {np.max(err)*100:.1f} cm", flush=True)
    env.env.shutdown()


if __name__ == "__main__":
    main()
