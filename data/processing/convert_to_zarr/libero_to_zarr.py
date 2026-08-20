"""
LIBERO demo HDF5 → 3DFA zarr (keypose format, RGB-D + extrinsics/intrinsics).

The distributed LIBERO datasets store 128x128 RGB only, but every frame also
stores the flattened MuJoCo state. We re-render observations at keypose frames
via env.regenerate_obs_from_state with camera_depths=True, and read per-frame
camera intrinsics/extrinsics (OpenCV convention) from robosuite camera_utils.
The eye-in-hand camera moves with the arm, so its extrinsics are per-frame.

Keypose discovery mirrors data/processing/rlbench_utils.py: a frame is a
keypose if the gripper command changes, the arm is stopped (near-zero joint
velocities outside a refractory buffer), or it is the final frame.

Outputs {tgt}/train.zarr with the peract_collected schema:
  rgb            (N, NCAM, 3, S, S) uint8
  depth          (N, NCAM, S, S)    float16   metric
  proprioception (N, 3, 1, 8)       float32   (t-2, t-1, t keypose EEF)
  action         (N, 1, 1, 8)       float32   next keypose EEF (xyz+quat_xyzw+open)
  extrinsics     (N, NCAM, 4, 4)    float32   camera-to-world, OpenCV axes
  intrinsics     (N, NCAM, 3, 3)    float32
  task_id        (N,)               uint8
  variation      (N,)               uint8     always 0 (LIBERO has no variations)
  demo_id        (N,)               uint32

Also writes instructions/{suite}/instructions.json ({task: {"0": [lang]}}).

Run inside the LIBERO sim venv (MUJOCO_GL=egl, PYTHONPATH=<LIBERO repo>):
  python libero_to_zarr.py --suite libero_spatial --tgt /path/out --im_size 256
"""
import argparse
import json
import os

import h5py
import numpy as np
from numcodecs import Blosc
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import zarr

os.environ.setdefault("MUJOCO_GL", "egl")

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import robosuite.utils.camera_utils as CU

CAMERAS = ["agentview", "robot0_eye_in_hand"]
NCAM = len(CAMERAS)


def parse_arguments():
    p = argparse.ArgumentParser()
    p.add_argument('--suite', type=str, default='libero_spatial')
    p.add_argument('--datasets_root', type=str, default=None,
                   help='Folder with {suite}/*.hdf5 (default: LIBERO datasets path)')
    p.add_argument('--tgt', type=str, required=True,
                   help='Output directory; writes {tgt}/train.zarr')
    p.add_argument('--im_size', type=int, default=256)
    p.add_argument('--instr_out', type=str, default=None,
                   help='Instructions json path (default {tgt}/instructions.json)')
    p.add_argument('--max_demos', type=int, default=None,
                   help='Cap demos per task (debug)')
    p.add_argument('--task_ids', type=str, default=None,
                   help='Comma-separated task indices within the suite (default all)')
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--store_trajectory', action='store_true',
                   help='Store dense interpolated EEF trajectory per keypose '
                        'segment instead of next keypose only (keypose_only '
                        'ablation; obs stay at keyposes). Mirrors '
                        'orbital_to_zarr.py --store_trajectory.')
    p.add_argument('--interp_len', type=int, default=50,
                   help='Interpolated steps per keypose segment')
    return p.parse_args()


def keypose_frames(actions, joint_vel, stopping_delta=0.1, buffer_size=4):
    """RLBench-style keypose discovery on dense LIBERO frames.

    actions: (T, 7) OSC deltas; actions[:, -1] is the gripper command
    (robosuite: -1 open, +1 close). joint_vel: (T, 7).
    Returns sorted keypose frame indices (never frame 0).
    """
    T = len(actions)
    grip = (actions[:, -1] > 0).astype(np.int8)  # 1 = closing
    keyframes = []
    stopped_buffer = 0
    for i in range(T):
        last = i == T - 1
        grip_changed = i > 0 and grip[i] != grip[i - 1]
        next_is_not_final = i == T - 2
        grip_stable = (
            i < T - 2
            and grip[i] == grip[min(T - 1, i + 1)]
            and grip[i] == grip[max(0, i - 1)]
            and grip[max(0, i - 2)] == grip[max(0, i - 1)]
        )
        small_delta = np.allclose(joint_vel[i], 0, atol=stopping_delta)
        stopped = (
            stopped_buffer <= 0 and small_delta
            and (not next_is_not_final) and grip_stable
        )
        stopped_buffer = buffer_size if stopped else stopped_buffer - 1
        if i != 0 and (grip_changed or last or stopped):
            keyframes.append(i)
    if len(keyframes) > 1 and (keyframes[-1] - 1) == keyframes[-2]:
        keyframes.pop(-2)
    return keyframes


def interpolate_eef(states, num_steps):
    """Cubic-spline interpolate (T, 8) EEF states [xyz, quat_xyzw, open] to (num_steps, 8).

    Same scheme as data/processing/orbital_utils._interpolate_eef: spline in
    euler space, nearest-neighbor for the gripper bit.
    """
    from scipy.interpolate import CubicSpline, interp1d
    if len(states) == 1:
        return np.repeat(states, num_steps, axis=0)
    eul = R.from_quat(states[:, 3:7]).as_euler('xyz')
    traj = np.concatenate([states[:, :3], eul, states[:, 7:]], axis=1)
    t0 = np.linspace(0, 1, len(traj))
    t1 = np.linspace(0, 1, num_steps)
    main = CubicSpline(t0, traj[:, :6], axis=0)(t1)
    grip = interp1d(t0, traj[:, 6:], axis=0, kind='nearest')(t1)
    quat = R.from_euler('xyz', main[:, 3:6]).as_quat()
    return np.concatenate([main[:, :3], quat, grip], axis=1).astype(np.float32)


def dense_eef_from_h5(d, grip_open):
    """Per-frame EEF states (T, 8) from HDF5 obs (ee_ori is axis-angle)."""
    pos = d["obs/ee_pos"][:]
    quat = R.from_rotvec(d["obs/ee_ori"][:]).as_quat()
    return np.concatenate([pos, quat, grip_open[:, None]], axis=1).astype(np.float32)


def eef_state(obs, gripper_open):
    """xyz + quat_xyzw + open bit from a robosuite obs dict."""
    return np.concatenate([
        obs["robot0_eef_pos"], obs["robot0_eef_quat"], [gripper_open]
    ]).astype(np.float32)


def render_frame(env, state, im_size):
    """Set sim state and return per-camera rgb/depth/K/E in OpenCV convention."""
    obs = env.regenerate_obs_from_state(state)
    rgbs, depths, Ks, Es = [], [], [], []
    for cam in CAMERAS:
        # robosuite's default "opengl" image convention renders row 0 at the
        # bottom; flip vertically so pixel (0,0) is top-left, matching the
        # pinhole model used by camera_utils K/E and utils/depth2cloud.
        rgb = obs[f"{cam}_image"][::-1].copy()
        depth = CU.get_real_depth_map(env.sim, obs[f"{cam}_depth"])[::-1, :, 0]
        rgbs.append(rgb.transpose(2, 0, 1))
        depths.append(depth.astype(np.float16))
        Ks.append(CU.get_camera_intrinsic_matrix(env.sim, cam, im_size, im_size))
        Es.append(CU.get_camera_extrinsic_matrix(env.sim, cam))
    return (obs, np.stack(rgbs).astype(np.uint8), np.stack(depths),
            np.stack(Ks).astype(np.float32), np.stack(Es).astype(np.float32))


def convert(args):
    b = benchmark.get_benchmark_dict()[args.suite]()
    datasets_root = args.datasets_root or get_libero_path("datasets")
    task_ids = (list(range(b.n_tasks)) if args.task_ids is None
                else [int(t) for t in args.task_ids.split(',')])

    filename = os.path.join(args.tgt, "train.zarr")
    if os.path.exists(filename):
        if not args.overwrite:
            raise SystemExit(f"[SKIP] {filename} exists (--overwrite to rebuild)")
        import shutil
        shutil.rmtree(filename)
    os.makedirs(args.tgt, exist_ok=True)

    S = args.im_size
    compressor = Blosc(cname='lz4', clevel=1, shuffle=Blosc.SHUFFLE)
    instructions = {}

    with zarr.open_group(filename, mode="w") as zf:
        def _create(field, shape, dtype):
            zf.create_dataset(field, shape=(0,) + shape, chunks=(1,) + shape,
                              compressor=compressor, dtype=dtype)

        _create("rgb",            (NCAM, 3, S, S), "uint8")
        _create("depth",          (NCAM, S, S),    "float16")
        _create("proprioception", (3, 1, 8),       "float32")
        act_len = args.interp_len if args.store_trajectory else 1
        _create("action",         (act_len, 1, 8), "float32")
        _create("extrinsics",     (NCAM, 4, 4),    "float32")
        _create("intrinsics",     (NCAM, 3, 3),    "float32")
        _create("task_id",        (),              "uint8")
        _create("variation",      (),              "uint8")
        _create("demo_id",        (),              "uint32")

        n_rollouts = 0
        for tid in task_ids:
            task = b.get_task(tid)
            task_name = task.name
            instructions[task_name] = {"0": [task.language]}
            h5_path = os.path.join(datasets_root, b.get_task_demonstration(tid))
            if not os.path.exists(h5_path):
                print(f"[WARN] missing {h5_path}")
                continue

            env = OffScreenRenderEnv(
                bddl_file_name=os.path.join(
                    get_libero_path("bddl_files"),
                    task.problem_folder, task.bddl_file),
                camera_heights=S, camera_widths=S, camera_depths=True,
            )
            env.seed(0)
            env.reset()

            with h5py.File(h5_path, "r") as f:
                demos = sorted(f["data"].keys(), key=lambda x: int(x.split('_')[1]))
                if args.max_demos:
                    demos = demos[:args.max_demos]
                for demo_key in tqdm(demos, desc=f"{tid}:{task_name[:40]}"):
                    d = f[f"data/{demo_key}"]
                    actions = d["actions"][:]
                    joint_vel = np.gradient(d["obs/joint_states"][:], axis=0) * 20.0
                    states = d["states"][:]

                    kf = keypose_frames(actions, joint_vel)
                    key_frames = [0] + kf
                    if len(key_frames) < 2:
                        continue

                    # gripper_open at each keypose: -1 action = open
                    grip_open = (actions[:, -1] < 0).astype(np.float32)

                    frames = []
                    for k in key_frames:
                        obs, rgb, depth, K, E = render_frame(env, states[k], S)
                        frames.append((eef_state(obs, grip_open[k]), rgb, depth, K, E))

                    eefs = np.stack([fr[0] for fr in frames])       # (T+1, 8)
                    prop = eefs[:-1]
                    prop_1 = np.concatenate([prop[:1], prop[:-1]])
                    prop_2 = np.concatenate([prop_1[:1], prop_1[:-1]])
                    prop = np.stack([prop_2, prop_1, prop], axis=1) # (T, 3, 8)
                    T = len(prop)

                    if args.store_trajectory:
                        # Dense EEF path between consecutive keyposes,
                        # resampled to interp_len steps per segment.
                        dense = dense_eef_from_h5(d, grip_open)     # (Tdense, 8)
                        action = np.stack([
                            interpolate_eef(
                                dense[key_frames[i]:key_frames[i + 1] + 1],
                                args.interp_len)
                            for i in range(T)
                        ]).reshape(T, args.interp_len, 1, 8)
                    else:
                        action = eefs[1:].reshape(T, 1, 1, 8)

                    zf['rgb'].append(np.stack([fr[1] for fr in frames[:-1]]))
                    zf['depth'].append(np.stack([fr[2] for fr in frames[:-1]]))
                    zf['proprioception'].append(prop.reshape(T, 3, 1, 8))
                    zf['action'].append(action)
                    zf['extrinsics'].append(np.stack([fr[4] for fr in frames[:-1]]))
                    zf['intrinsics'].append(np.stack([fr[3] for fr in frames[:-1]]))
                    zf['task_id'].append(np.full(T, tid, dtype=np.uint8))
                    zf['variation'].append(np.zeros(T, dtype=np.uint8))
                    zf['demo_id'].append(np.full(T, n_rollouts, dtype=np.uint32))
                    n_rollouts += 1
            env.close()

        print(f"[DONE] {len(zf['action'])} keypose steps, {n_rollouts} demos → {filename}")

    instr_path = args.instr_out or os.path.join(args.tgt, "instructions.json")
    os.makedirs(os.path.dirname(instr_path), exist_ok=True)
    with open(instr_path, "w") as fp:
        json.dump(instructions, fp, indent=1)
    print(f"[DONE] instructions → {instr_path}")


if __name__ == "__main__":
    convert(parse_arguments())
