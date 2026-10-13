"""Raw PerAct2 episodes (all frames) -> dense chunk zarr or keyframe zarr with the reference 50-point keypose trajectory."""
import argparse
import json
import os
import pickle
import sys
from multiprocessing import Pool

import numpy as np
from numcodecs import Blosc
from PIL import Image
import zarr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
from data.processing.rlbench_utils import (
    DEPTH_SCALE,
    euler_to_quat_np,
    image_to_float_array,
    interpolate_trajectory,
    keypoint_discovery,
    num2id,
    quat_to_euler_np,
)
from datasets.rlbench import PERACT2_TASKS

CAMERAS = ["front", "wrist_left", "wrist_right"]
NHAND = 2


def row_schema(im, horizon):
    return {
        "rgb": ((len(CAMERAS), 3, im, im), "uint8"),
        "depth": ((len(CAMERAS), im, im), "float16"),
        "extrinsics": ((len(CAMERAS), 4, 4), "float16"),
        "intrinsics": ((len(CAMERAS), 3, 3), "float16"),
        "proprioception": ((3, NHAND, 8), "float32"),
        "proprioception_keypose": ((3, NHAND, 8), "float32"),
        "action": ((horizon, NHAND, 8), "float32"),
        "next_keypose": ((1, NHAND, 8), "float32"),
        "proprioception_joints": ((1, NHAND, 8), "float32"),
        "action_joints": ((1, NHAND, 8), "float32"),
        "task_id": ((), "uint8"),
        "variation": ((), "uint8"),
        "demo_id": ((), "uint32"),
        "episode_idx": ((), "uint16"),
        "frame_id": ((), "uint16"),
        "keypose_idx": ((), "uint8"),
        "is_keyframe": ((), "bool"),
    }


def interpolate_segment(traj, num_steps, unwrap=False):
    """Reference hiveformer_to_zarr._interpolate: (T, 8) xyz+quat+gripper -> (num_steps, 8) via euler 'xyz' cubic spline."""
    euler = quat_to_euler_np(traj[:, 3:7])
    # The reference does not unwrap, so a +-pi crossing makes the spline spin through a large rotation.
    traj = np.concatenate((traj[:, :3], np.unwrap(euler, axis=0) if unwrap else euler, traj[:, 7:]), 1)
    traj = interpolate_trajectory(traj, num_steps)
    return np.concatenate((traj[:, :3], euler_to_quat_np(traj[:, 3:6]), traj[:, 6:]), 1)


def load_frame(ep_path, demo, t, im):
    """rgb/depth/extrinsics/intrinsics at frame t, exactly as the reference peract2_to_zarr.py."""
    rgb = np.stack([np.array(Image.open(f"{ep_path}/{c}_rgb/rgb_{num2id(t)}.png")) for c in CAMERAS]).transpose(0, 3, 1, 2)
    depth = []
    for c in CAMERAS:
        d = image_to_float_array(Image.open(f"{ep_path}/{c}_depth/depth_{num2id(t)}.png"), DEPTH_SCALE)
        near, far = demo[t].misc[f"{c}_camera_near"], demo[t].misc[f"{c}_camera_far"]
        depth.append(near + d * (far - near))
    extr = np.stack([demo[t].misc[f"{c}_camera_extrinsics"].astype(np.float16) for c in CAMERAS])
    intr = np.stack([demo[t].misc[f"{c}_camera_intrinsics"].astype(np.float16) for c in CAMERAS])
    assert rgb.shape[-1] == im
    return rgb, np.stack(depth).astype(np.float16), extr, intr


def process_episode(job):
    ep_path, task_id, im, cfg = job
    with open(os.path.join(ep_path, "low_dim_obs.pkl"), "rb") as f:
        demo = pickle.load(f)
    with open(os.path.join(ep_path, "variation_number.pkl"), "rb") as f:
        variation = int(pickle.load(f))
    kfs = keypoint_discovery(demo, bimanual=True)
    kfs.insert(0, 0)
    states = np.stack([np.concatenate([o.left.gripper_pose, [o.left.gripper_open], o.right.gripper_pose, [o.right.gripper_open]])
                       for o in demo]).astype(np.float32).reshape(len(demo), NHAND, 8)
    joints = np.stack([np.concatenate([o.left.joint_positions, [o.left.gripper_open], o.right.joint_positions, [o.right.gripper_open]])
                       for o in demo]).astype(np.float32).reshape(len(demo), NHAND, 8)
    ps = cfg["proprio_stride"]
    rows = {k: [] for k in row_schema(im, cfg["horizon"])}
    for i in range(len(kfs) - 1):
        k0, k1 = kfs[i], kfs[i + 1]
        frames = [k0] if cfg["target"] == "keypose_segment" else range(k0, k1)
        for t in frames:
            rgb, depth, extr, intr = load_frame(ep_path, demo, t, im)
            rows["rgb"].append(rgb)
            rows["depth"].append(depth)
            rows["extrinsics"].append(extr)
            rows["intrinsics"].append(intr)
            # Reference keypose history: the two latest keyframes strictly before t (padded with the first), then s_t.
            prev = [k for k in kfs if k < t]
            p1 = prev[-1] if prev else t
            p2 = prev[-2] if len(prev) > 1 else p1
            rows["proprioception_keypose"].append(states[[p2, p1, t]])
            rows["proprioception"].append(states[[p2, p1, t]] if cfg["proprio"] == "keypose"
                                          else states[[max(0, t - 2 * ps), max(0, t - ps), t]])
            if cfg["target"] == "chunk":
                rows["action"].append(states[np.minimum(t + cfg["chunk_stride"] * np.arange(1, cfg["horizon"] + 1), len(demo) - 1)])
            else:
                rows["action"].append(np.stack([interpolate_segment(states[k0:k1 + 1, h], cfg["horizon"], cfg["unwrap_euler"]) for h in range(NHAND)], 1))
            rows["next_keypose"].append(states[[k1]])
            rows["proprioception_joints"].append(joints[[t]])
            rows["action_joints"].append(joints[[k1]])
            rows["task_id"].append(task_id)
            rows["variation"].append(variation)
            rows["episode_idx"].append(int(os.path.basename(ep_path).replace("episode", "")))
            rows["frame_id"].append(t)
            rows["keypose_idx"].append(i)
            rows["is_keyframe"].append(t == k0)
    return rows


def parse_args():
    p = argparse.ArgumentParser(description="Raw PerAct2 episodes -> dense train/val zarr (standard cameras)")
    p.add_argument("--root", required=True, help="raw root with {split}/{task}/all_variations/episodes/episodeN")
    p.add_argument("--out", required=True)
    p.add_argument("--tasks", required=True, help="comma-separated; task_id indexes PERACT2_TASKS")
    p.add_argument("--target", choices=["chunk", "keypose_segment"], required=True)
    p.add_argument("--horizon", type=int, required=True)
    p.add_argument("--chunk-stride", type=int, default=1)
    p.add_argument("--proprio", choices=["timestep", "keypose"], default="timestep")
    p.add_argument("--proprio-stride", type=int, default=1)
    p.add_argument("--unwrap-euler", action="store_true", help="keypose_segment: unwrap euler angles before the spline (not in the reference)")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--splits", default="train,val")
    p.add_argument("--max-episodes", type=int, default=None, help="tests only")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    tasks = [t.strip() for t in args.tasks.split(",")]
    cfg = {"target": args.target, "horizon": args.horizon, "chunk_stride": args.chunk_stride,
           "proprio": args.proprio, "proprio_stride": args.proprio_stride, "unwrap_euler": args.unwrap_euler}
    schema = row_schema(args.image_size, args.horizon)
    compressor = Blosc(cname="lz4", clevel=1, shuffle=Blosc.SHUFFLE)
    os.makedirs(args.out, exist_ok=True)
    summary = {"config": cfg, "tasks": tasks}
    for split in args.splits.split(","):
        path = os.path.join(args.out, split + ".zarr")
        if os.path.exists(path) and not args.overwrite:
            sys.exit("[ERROR] {} exists; pass --overwrite".format(path))
        jobs = []
        for task in tasks:
            folder = f"{args.root}/{split}/{task}/all_variations/episodes"
            # Same sorted order as the reference converter, so demo_id order matches Peract2_zarr.
            eps = sorted(os.listdir(folder))[:args.max_episodes]
            jobs += [(os.path.join(folder, ep), PERACT2_TASKS.index(task), args.image_size, cfg) for ep in eps]
        zf = zarr.open_group(path, mode="w")
        for key, (shape, dtype) in schema.items():
            zf.create_dataset(key, shape=(0,) + shape, chunks=(1,) + shape, compressor=compressor, dtype=dtype)
        zf.attrs.update(dict(cfg, tasks=tasks, cameras=CAMERAS))
        n_rows, episodes = 0, []
        with Pool(args.workers) as pool:
            # One batch of episodes in memory at a time.
            batched = (r for i in range(0, len(jobs), args.workers) for r in pool.map(process_episode, jobs[i:i + args.workers]))
            for demo_id, rows in enumerate(batched):
                n = len(rows["frame_id"])
                rows["demo_id"] = [demo_id] * n
                for key, (shape, dtype) in schema.items():
                    zf[key].append(np.asarray(rows[key], dtype=dtype).reshape((n,) + shape))
                n_rows += n
                episodes.append([jobs[demo_id][0], n])
                print("[{}] demo {} {} rows={} total={}".format(split, demo_id, jobs[demo_id][0], n, n_rows), flush=True)
        summary[split] = {"rows": n_rows, "episodes": episodes}
    with open(os.path.join(args.out, "build_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print("[DONE]", {s: summary[s]["rows"] for s in args.splits.split(",")})


if __name__ == "__main__":
    main()
