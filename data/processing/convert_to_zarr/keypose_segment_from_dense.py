"""Dense zarr -> keyframe-row zarr whose action is the reference 3DFA keypose-horizon trajectory (spline to the next keypose)."""
import argparse
import os

import numpy as np
import zarr
from numcodecs import Blosc
from scipy.interpolate import CubicSpline, interp1d
from scipy.spatial.transform import Rotation

COPY = ["rgb", "depth", "extrinsics", "intrinsics", "proprioception", "proprioception_joints", "task_id", "variation",
        "camera_group", "demo_id", "frame_id", "keypose_idx", "episode_idx"]


def interpolate_trajectory(traj, num_steps):
    """(T, 7) xyz+euler+gripper -> (num_steps, 7); reference 3DFA data_processing/rlbench_utils.py."""
    if traj.shape[0] == 1:
        return np.repeat(traj, num_steps, axis=0)
    t_orig, t_target = np.linspace(0, 1, traj.shape[0]), np.linspace(0, 1, num_steps)
    main = CubicSpline(t_orig, traj[..., :-1], axis=0)(t_target)
    last = interp1d(t_orig, traj[..., -1:], axis=0, kind="nearest")(t_target)
    return np.concatenate((main, last), 1)


def interpolate_segment(states, num_steps, unwrap=True):
    """(T, 8) xyz+quat(xyzw)+gripper -> (num_steps, 8), as the reference hiveformer_to_zarr._interpolate."""
    euler = Rotation.from_quat(states[:, 3:7]).as_euler("xyz")
    # The reference splines raw Euler angles, which sweep the long way round when an angle crosses +-pi.
    euler = np.unwrap(euler, axis=0) if unwrap else euler
    traj = np.concatenate((states[:, :3], euler, states[:, 7:]), 1)
    traj = interpolate_trajectory(traj, num_steps)
    return np.concatenate((traj[:, :3], Rotation.from_euler("xyz", traj[:, 3:6]).as_quat(), traj[:, 6:]), 1)


def convert(src, dst, num_steps, unwrap=True):
    z = zarr.open(src, "r")
    demo, frame, kf = (np.asarray(z[k][:]) for k in ("demo_id", "frame_id", "is_keyframe"))
    cur = np.asarray(z["proprioception"][:, -1])
    nxt = np.asarray(z["next_keypose"][:, 0])
    rows = np.flatnonzero(kf)
    ends = np.append(rows[1:], len(demo))
    actions = []
    for r, e in zip(rows, ends):
        # Rows are ordered by demo then frame, so a segment runs to the next keyframe row or the demo's end.
        seg = np.arange(r, e)
        seg = seg[demo[seg] == demo[r]]
        assert np.all(np.diff(frame[seg]) == 1), "dense rows must be contiguous frames"
        states = np.concatenate([cur[seg], nxt[r][None]], 0)
        actions.append(np.stack([interpolate_segment(states[:, h], num_steps, unwrap) for h in range(states.shape[1])], 1))
    out = zarr.open_group(dst, mode="w")
    comp = Blosc(cname="lz4", clevel=1, shuffle=Blosc.SHUFFLE)
    for k in COPY:
        if k in z:
            a = z[k]
            out.create_dataset(k, data=np.stack([a[int(r)] for r in rows]), chunks=(1,) + a.shape[1:], compressor=comp, dtype=a.dtype)
    out.create_dataset("action", data=np.stack(actions).astype(np.float32), chunks=(1, num_steps, actions[0].shape[1], 8), compressor=comp)
    out.create_dataset("next_keypose", data=np.asarray(z["next_keypose"][:])[rows], compressor=comp)
    out.attrs.update({"source": src, "target": "keypose_segment", "horizon": num_steps, "euler_unwrap": unwrap})
    print(dst, "rows", len(rows), "action", out["action"].shape)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--exact", action="store_true", help="no Euler unwrap, exactly as the reference")
    a = p.parse_args()
    for split in ("train", "val"):
        convert(os.path.join(a.src, split + ".zarr"), os.path.join(a.dst, split + ".zarr"), a.steps, not a.exact)
