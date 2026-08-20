"""Online evaluation of a 3DFA checkpoint on LIBERO (closed-loop success rate).

Mirrors online_evaluation_rlbench/evaluate_policy.py but drives LIBERO's
OSC_POSE controller: the model predicts absolute EE keyposes
(xyz + quat_xyzw + gripper) from RGB-D + extrinsics-unprojected point clouds,
and a servo loop converts each keypose into clipped delta actions.

Observation processing matches data/processing/convert_to_zarr/libero_to_zarr.py:
256x256 renders, vertical flip to OpenCV convention, metric depth via
get_real_depth_map, K/E from robosuite camera_utils.

Run inside an env with torch-cu128 + robosuite/LIBERO (see
scripts/libero/setup_eval_env.sh), MUJOCO_GL=egl, PYTHONPATH += LIBERO repo:
  python online_evaluation_libero/evaluate_policy.py \
    checkpoint=... task=all output_file=eval.json num_demos=10
"""

import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("MUJOCO_GL", "egl")

# LIBERO's get_task_init_states calls bare torch.load, which fails under
# torch>=2.6 (weights_only defaults to True). These are trusted local files.
_torch_load = torch.load
torch.load = lambda *a, **k: _torch_load(*a, **{**k, "weights_only": k.get("weights_only", False)})

from scipy.spatial.transform import Rotation as R

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import robosuite.utils.camera_utils as CU

from online_evaluation_rlbench.evaluate_policy import load_models
from utils.common_utils import round_floats
from utils.hydra_utils import get_config, get_config_path
from modeling.encoder.text import fetch_tokenizers

CAMERAS = ["agentview", "robot0_eye_in_hand"]
IM_SIZE = 256
# OSC_POSE output limits (from LIBERO's controller config)
POS_STEP = 0.05     # m per action unit
ROT_STEP = 0.5      # rad per action unit
SERVO_STEPS = 300   # max control steps to reach one keypose (shared across sub-targets)
POS_TOL = 0.008     # m
ROT_TOL = 0.15      # rad
SERVO_GAIN = 4.0    # error multiplier before clipping — OSC under-shoots at
                    # small commanded deltas (impedance steady-state error)
SETTLE_STEPS = 10   # LIBERO protocol: let objects settle after reset
GRIPPER_STEPS = 12  # extra steps to actuate a gripper change


class Actioner:
    """Same contract as the RLBench Actioner."""

    def __init__(self, policy, backbone="clip", cfg_scale=None):
        self._policy = policy.cuda().eval()
        self.tokenizer = fetch_tokenizers(backbone)
        self._cfg_scale = float(cfg_scale) if cfg_scale is not None else None

    def load_episode(self, description):
        self._instr_str = description
        self._instr = self.tokenizer([description]).cuda(non_blocking=True)

    @torch.no_grad()
    def predict(self, rgbs, pcds, gripper, prediction_len=1):
        dtype = next(self._policy.parameters()).dtype
        return self._policy(
            None,
            torch.full([1, prediction_len, 1], False).cuda(non_blocking=True),
            rgbs.to(dtype),
            None,
            pcds.to(dtype),
            self._instr,
            gripper[:, :, None, :7].to(dtype),
            run_inference=True,
            cfg_scale=self._cfg_scale,
        ).view(1, prediction_len, 8)


def get_obs_tensors(env, obs):
    """(rgbs, pcds) as (1, ncam, 3, H, W) cuda float; eef state (8,) numpy."""
    rgbs, pcds = [], []
    for cam in CAMERAS:
        rgb = obs[f"{cam}_image"][::-1].astype(np.float32) / 255.0     # (H,W,3)
        depth = CU.get_real_depth_map(env.sim, obs[f"{cam}_depth"])[::-1, :, 0]
        K = CU.get_camera_intrinsic_matrix(env.sim, cam, IM_SIZE, IM_SIZE)
        E = CU.get_camera_extrinsic_matrix(env.sim, cam)
        u, v = np.meshgrid(np.arange(IM_SIZE), np.arange(IM_SIZE))
        pix = np.stack([u, v, np.ones_like(u)], 0).reshape(3, -1)
        cam_pts = np.linalg.inv(K) @ pix * depth.reshape(1, -1)
        world = E[:3, :3] @ cam_pts + E[:3, 3:4]
        rgbs.append(rgb.transpose(2, 0, 1))
        pcds.append(world.reshape(3, IM_SIZE, IM_SIZE))
    rgbs = torch.from_numpy(np.stack(rgbs)[None]).float().cuda()
    pcds = torch.from_numpy(np.stack(pcds).astype(np.float32)[None]).cuda()
    return rgbs, pcds


def eef_state(obs, gripper_open):
    return np.concatenate([
        obs["robot0_eef_pos"], obs["robot0_eef_quat"], [gripper_open]
    ]).astype(np.float32)


WAYPOINT_SPACING = 0.03  # m between interpolated sub-targets


def servo_to_keypose(env, obs, target, gripper_open_cmd, gripper_open_cur=None):
    """Drive OSC_POSE toward an absolute keypose along an interpolated path.

    Straight jumps between sparse keyposes leave the demo manifold and knock
    objects over; short sub-targets keep the interim motion close to linear
    (verified: dense GT waypoint replay succeeds where keypose jumps fail).
    The gripper HOLDS its current state during the motion and actuates the
    commanded state only after arriving — keypose semantics (a gripper close
    commanded mid-flight grabs air before the bowl).
    Returns (obs, done, success).
    """
    if gripper_open_cur is None:
        gripper_open_cur = gripper_open_cmd
    grip_act = -1.0 if gripper_open_cur else 1.0
    # Reject degenerate predictions: NaN or far outside the workspace would
    # otherwise blow up the sub-target count (observed: deterministic hang,
    # n_sub explosion spinning in numpy where SIGALRM can't interrupt).
    if not np.all(np.isfinite(target)):
        return obs, False, env.check_success()
    target = target.copy()
    target[:3] = np.clip(target[:3], [-0.6, -0.6, 0.4], [0.6, 0.6, 1.6])
    start = obs["robot0_eef_pos"].copy()
    r_start = R.from_quat(obs["robot0_eef_quat"])
    r_tgt = R.from_quat(target[3:7])
    n_sub = int(np.clip(
        np.ceil(np.linalg.norm(target[:3] - start) / WAYPOINT_SPACING), 1, 64))
    from scipy.spatial.transform import Slerp
    slerp = Slerp([0, 1], R.concatenate([r_start, r_tgt]))
    subs = [(start + (target[:3] - start) * f, slerp(f)) for f in
            np.linspace(1 / n_sub, 1.0, n_sub)]

    steps_left = SERVO_STEPS
    for sub_pos, sub_rot in subs:
        while steps_left > 0:
            steps_left -= 1
            pos_err = sub_pos - obs["robot0_eef_pos"]
            rot_err = (sub_rot * R.from_quat(obs["robot0_eef_quat"]).inv()).as_rotvec()
            if np.linalg.norm(pos_err) < POS_TOL and np.linalg.norm(rot_err) < ROT_TOL:
                break
            action = np.concatenate([
                np.clip(SERVO_GAIN * pos_err / POS_STEP, -1, 1),
                np.clip(SERVO_GAIN * rot_err / ROT_STEP, -1, 1),
                [grip_act],
            ])
            obs, _, done, info = env.step(action)
            if done:
                return obs, done, env.check_success()
        if steps_left <= 0:
            break
    # Arrived: actuate the commanded gripper state with the arm held.
    grip_cmd_act = -1.0 if gripper_open_cmd else 1.0
    for _ in range(GRIPPER_STEPS):
        obs, _, done, info = env.step(np.array([0, 0, 0, 0, 0, 0, grip_cmd_act]))
        if done:
            break
    return obs, False, env.check_success()


def rollout(env, actioner, init_state, max_keyposes, num_history,
            prediction_len=1, waypoint_stride=10):
    env.reset()
    obs = env.set_init_state(init_state)
    for _ in range(SETTLE_STEPS):
        obs, _, _, _ = env.step(np.zeros(7))

    gripper_open = 1.0
    # keypose history (oldest → current), padded with the initial state
    hist = [eef_state(obs, gripper_open)] * max(num_history, 1)

    for _ in range(max_keyposes):
        rgbs, pcds = get_obs_tensors(env, obs)
        gripper = torch.from_numpy(np.stack(hist[-num_history:])[None]).float().cuda()
        traj = actioner.predict(rgbs, pcds, gripper,
                                prediction_len=prediction_len)[0].cpu().numpy()
        # Keypose model (T=1): one target. Dense model (T=50): servo through
        # subsampled waypoints, always ending on the final pose.
        idxs = (list(range(waypoint_stride - 1, prediction_len - 1, waypoint_stride))
                + [prediction_len - 1]) if prediction_len > 1 else [0]
        for i in idxs:
            target = traj[i, :7].astype(np.float64)
            target[3:7] /= np.linalg.norm(target[3:7]) + 1e-8
            gripper_prev = gripper_open
            gripper_open = 1.0 if traj[i, 7] > 0.5 else 0.0
            obs, done, success = servo_to_keypose(
                env, obs, target, bool(gripper_open),
                gripper_open_cur=bool(gripper_prev))
            if success:
                return True
            if done:
                return env.check_success()
        hist.append(eef_state(obs, gripper_open))
    return bool(env.check_success())


def main():
    args = get_config(
        overrides=sys.argv[1:], config_name="config",
        config_path=get_config_path(),
    )
    suite = getattr(args, "libero_suite", None) or "libero_spatial"
    num_demos = int(args.num_demos) if args.num_demos else 10
    max_keyposes = int(args.max_steps) if args.max_steps else 10
    prediction_len = int(getattr(args, "prediction_len", 1) or 1)

    model = load_models(args)
    print("workspace_normalizer:", model.workspace_normalizer, flush=True)
    text_backbone = getattr(args, "text_backbone", None) or args.backbone
    actioner = Actioner(model, backbone=text_backbone,
                        cfg_scale=getattr(args, "cfg_scale", None))
    num_history = int(getattr(args, "num_history", 1))

    b = benchmark.get_benchmark_dict()[suite]()
    # task: null/"all" | "3" | "3,5,7" | "608-617" (inclusive range; avoids
    # hydra's comma-sweep ambiguity on CLI)
    if args.task in (None, "all"):
        task_ids = range(b.n_tasks)
    elif "-" in str(args.task):
        lo, hi = str(args.task).split("-")
        task_ids = range(int(lo), int(hi) + 1)
    else:
        task_ids = [int(t) for t in str(args.task).split(",")]

    results = {}
    for tid in task_ids:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

        task = b.get_task(tid)
        env = OffScreenRenderEnv(
            bddl_file_name=os.path.join(
                get_libero_path("bddl_files"), task.problem_folder, task.bddl_file),
            camera_heights=IM_SIZE, camera_widths=IM_SIZE, camera_depths=True,
        )
        env.seed(int(args.seed))
        actioner.load_episode(task.language)
        init_states = b.get_task_init_states(tid)

        succ = []
        for ep in range(min(num_demos, len(init_states))):
            # Watchdog: MuJoCo/EGL can deadlock mid-render; a stuck episode
            # must not stall the sweep.
            import signal

            def _timeout(signum, frame):
                raise TimeoutError("episode watchdog expired")

            signal.signal(signal.SIGALRM, _timeout)
            signal.alarm(600)
            try:
                s = rollout(env, actioner, init_states[ep], max_keyposes,
                            num_history, prediction_len=prediction_len)
            except TimeoutError:
                print(f"[warn] task {tid} ep {ep} hung; recreating env", flush=True)
                s = False
                env.close()
                env = OffScreenRenderEnv(
                    bddl_file_name=os.path.join(
                        get_libero_path("bddl_files"),
                        task.problem_folder, task.bddl_file),
                    camera_heights=IM_SIZE, camera_widths=IM_SIZE,
                    camera_depths=True,
                )
                env.seed(int(args.seed))
            except Exception as e:  # sim instability should not kill the sweep
                print(f"[warn] task {tid} ep {ep} crashed: {e}", flush=True)
                s = False
            finally:
                signal.alarm(0)
            succ.append(float(s))
            print(f"task {tid} ({task.name[:40]}) ep {ep}: "
                  f"{'OK' if s else 'fail'} (running {np.mean(succ):.2f})", flush=True)
        env.close()

        results[task.name] = {"success_rate": float(np.mean(succ)), "n": len(succ)}
        results["mean"] = float(np.mean(
            [v["success_rate"] for k, v in results.items() if k != "mean"]))
        with open(args.output_file, "w") as f:
            json.dump(round_floats(results), f, indent=2)
        print(f"[{suite}] running mean over {len(results)-1} tasks: "
              f"{results['mean']:.3f}", flush=True)

    print("FINAL:", json.dumps(round_floats(results), indent=2))


if __name__ == "__main__":
    main()
