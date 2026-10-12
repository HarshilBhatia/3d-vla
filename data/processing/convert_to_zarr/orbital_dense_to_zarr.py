"""Convert raw seeded orbital PerAct2 episodes into dense (per-timestep) train.zarr / val.zarr."""
import argparse
import json
import os
import pickle
import sys
from multiprocessing import Pool

import numpy as np
from numcodecs import Blosc
from scipy.spatial.transform import Rotation, Slerp
import zarr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
from data.processing.rlbench_utils import CustomUnpickler, keypoint_discovery
from data.processing.orbital_utils import (
    _load_wrist_depth,
    get_group_id,
    load_extrinsics_from_misc,
    load_intrinsics_from_misc,
    load_orbital_extrinsics,
    load_rgb,
)
from data.generation.orbital.constants import PERACT2_PROFILE

PROFILE = PERACT2_PROFILE
CAMERAS = ["orbital_left", "orbital_right"] + PROFILE.wrist_cameras
NHAND = PROFILE.nhand


def row_schema(ncam, im, horizon):
    return {
        "rgb": ((ncam, 3, im, im), "uint8"),
        "depth": ((ncam, im, im), "float16"),
        "extrinsics": ((ncam, 4, 4), "float16"),
        "intrinsics": ((ncam, 3, 3), "float16"),
        "proprioception": ((3, NHAND, 8), "float32"),
        "proprioception_keypose": ((3, NHAND, 8), "float32"),
        "action": ((horizon, NHAND, 8), "float32"),
        "next_keypose": ((1, NHAND, 8), "float32"),
        "proprioception_joints": ((1, NHAND, 8), "float32"),
        "action_joints": ((1, NHAND, 8), "float32"),
        "task_id": ((), "uint8"),
        "variation": ((), "uint8"),
        "camera_group": ((), "uint8"),
        "demo_id": ((), "uint32"),
        "episode_idx": ((), "uint16"),
        "frame_id": ((), "uint16"),
        "keypose_idx": ((), "uint8"),
        "is_keyframe": ((), "bool"),
    }


def resample_segment(states, t, t_end, horizon):
    """(horizon, nhand, 8) poses at t + j/H*(t_end-t), j=1..H; xyz linear, quat slerp, gripper from the later frame."""
    pos = t + (t_end - t) * np.arange(1, horizon + 1) / horizon
    lo = np.clip(np.floor(pos).astype(int), t, t_end)
    hi = np.clip(np.ceil(pos).astype(int), t, t_end)
    w = (pos - lo)[:, None]
    out = np.empty((horizon, NHAND, 8), np.float32)
    for h in range(NHAND):
        s = states[:, h]
        out[:, h, :3] = (1 - w) * s[lo, :3] + w * s[hi, :3]
        frames = np.arange(t, t_end + 1)
        out[:, h, 3:7] = Slerp(frames, Rotation.from_quat(s[t:t_end + 1, 3:7]))(pos).as_quat()
        out[:, h, 7] = s[hi, 7]
        # Integer positions copy the recorded state exactly, so the last step equals the next keypose bit-for-bit.
        exact = lo == hi
        out[exact, h] = s[lo[exact]]
    return out


def process_episode(job):
    ep_path, task_id, group_str, im, horizon, stride, (target, chunk_stride, proprio, pstride) = job
    with open(os.path.join(ep_path, "low_dim_obs.pkl"), "rb") as f:
        demo = CustomUnpickler(f).load()
    with open(os.path.join(ep_path, "variation.txt")) as f:
        variation = int(f.read().strip())
    kfs = keypoint_discovery(demo, bimanual=True)
    kfs.insert(0, 0)
    states = np.stack([PROFILE.eef_fn(o) for o in demo]).reshape(len(demo), NHAND, 8)
    joints = np.stack([PROFILE.joints_fn(o) for o in demo]).reshape(len(demo), NHAND, 8)
    E_ol, E_or, K_ol, K_or = load_orbital_extrinsics(ep_path)
    group_id = get_group_id(ep_path, group_str)

    rows = {k: [] for k in row_schema(len(CAMERAS), im, horizon)}
    for i in range(len(kfs) - 1):
        k0, k1 = kfs[i], kfs[i + 1]
        for t in range(k0, k1, stride):
            obs = demo[t]
            rows["rgb"].append(np.stack([load_rgb(ep_path, c, t).transpose(2, 0, 1) for c in CAMERAS]))
            rows["depth"].append(np.stack(
                [obs.orbital_left_depth.astype(np.float32), obs.orbital_right_depth.astype(np.float32)]
                + [_load_wrist_depth(obs, c) for c in PROFILE.wrist_cameras]).astype(np.float16))
            rows["extrinsics"].append(np.stack(
                [E_ol, E_or] + [load_extrinsics_from_misc(obs, c) for c in PROFILE.wrist_cameras]).astype(np.float16))
            rows["intrinsics"].append(np.stack(
                [K_ol, K_or] + [load_intrinsics_from_misc(obs, c) for c in PROFILE.wrist_cameras]).astype(np.float16))
            # Keypose-style history: the two latest keyframes strictly before t, then the current state.
            prev = [k for k in kfs if k < t]
            p1 = prev[-1] if prev else t
            p2 = prev[-2] if len(prev) > 1 else p1
            rows["proprioception_keypose"].append(states[[p2, p1, t]])
            rows["proprioception"].append(states[[p2, p1, t]] if proprio == "keypose"
                                          else states[[max(0, t - 2 * pstride), max(0, t - pstride), t]])
            if target == "chunk":
                # Fixed-time chunk: the next horizon recorded states at chunk_stride, held at the last frame past the end.
                rows["action"].append(states[np.minimum(t + chunk_stride * np.arange(1, horizon + 1), len(demo) - 1)])
            else:
                rows["action"].append(resample_segment(states, t, k1, horizon))
            rows["next_keypose"].append(states[[k1]])
            rows["proprioception_joints"].append(joints[[t]])
            rows["action_joints"].append(joints[[k1]])
            rows["task_id"].append(task_id)
            rows["variation"].append(variation)
            rows["camera_group"].append(group_id)
            rows["episode_idx"].append(int(os.path.basename(ep_path).split("_")[-1]))
            rows["frame_id"].append(t)
            rows["keypose_idx"].append(i)
            rows["is_keyframe"].append(t == k0)
    return rows


def init_zarr(path, schema, compressor):
    zf = zarr.open_group(path, mode="w")
    for key, (shape, dtype) in schema.items():
        zf.create_dataset(key, shape=(0,) + shape, chunks=(1,) + shape, compressor=compressor, dtype=dtype)
    return zf


def list_jobs(args, task_list):
    with open(args.mapping) as f:
        mapping = json.load(f)["tasks"]
    shards = sorted("{}__{}".format(t, g) for t in args.tasks for g in mapping[t]["train_groups"])
    jobs = {"train": [], "val": []}
    for shard in shards:
        task, group = shard.split("__")
        group_root = os.path.join(args.raw, task, group)
        if not os.path.isdir(group_root) and args.allow_missing:
            print("[WARN] {}: no raw episodes".format(shard))
            continue
        # Lexicographic order and first-N-train split match orbital_to_zarr.py, so val = episodes 7, 8, 9.
        eps = sorted(d for d in os.listdir(group_root)
                     if d.startswith("episode_") and not d.endswith(".tmp")
                     and os.path.exists(os.path.join(group_root, d, "seed.txt")))
        if len(eps) != args.expected_episodes:
            msg = "{}: {} complete episodes (expected {})".format(shard, len(eps), args.expected_episodes)
            if not args.allow_missing:
                sys.exit("[ERROR] " + msg + "; a missing episode would shift the train/val split")
            print("[WARN] " + msg)
        for i, ep in enumerate(eps):
            split = "train" if i < args.train_episodes else "val"
            jobs[split].append((os.path.join(group_root, ep), task_list.index(task), group,
                                args.image_size, args.horizon, args.frame_stride,
                                (args.target, args.chunk_stride, args.proprio, args.proprio_stride or args.frame_stride)))
    return jobs


def parse_args():
    p = argparse.ArgumentParser(description="Raw orbital PerAct2 episodes -> dense train/val zarr")
    p.add_argument("--raw", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tasks", required=True, help="Comma-separated task names")
    p.add_argument("--mapping", default="instructions/peract2_orbital_task_group_mapping.json")
    p.add_argument("--train-episodes", type=int, default=27)
    p.add_argument("--expected-episodes", type=int, default=30)
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--target", choices=["keypose_path", "chunk"], default="keypose_path",
                   help="keypose_path: horizon points resampled to the next keypose; chunk: the next horizon states at --chunk-stride")
    p.add_argument("--chunk-stride", type=int, default=1)
    p.add_argument("--proprio", choices=["timestep", "keypose"], default="timestep",
                   help="what the 'proprioception' field holds: states at --proprio-stride, or the keypose history")
    p.add_argument("--proprio-stride", type=int, default=None, help="defaults to --frame-stride")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--allow-missing", action="store_true", help="Build even if a shard is incomplete (tests only)")
    return p.parse_args()


def main():
    args = parse_args()
    args.tasks = [t.strip() for t in args.tasks.split(",")]
    jobs = list_jobs(args, PROFILE.task_list)
    schema = row_schema(len(CAMERAS), args.image_size, args.horizon)
    compressor = Blosc(cname="lz4", clevel=1, shuffle=Blosc.SHUFFLE)
    os.makedirs(args.out, exist_ok=True)
    summary = {}
    for split, split_jobs in jobs.items():
        path = os.path.join(args.out, split + ".zarr")
        if os.path.exists(path) and not args.overwrite:
            sys.exit("[ERROR] {} exists; pass --overwrite".format(path))
        zf = init_zarr(path, schema, compressor)
        zf.attrs.update({"horizon": args.horizon, "frame_stride": args.frame_stride, "tasks": args.tasks, "target": args.target,
                         "chunk_stride": args.chunk_stride, "proprio": args.proprio, "proprio_stride": args.proprio_stride or args.frame_stride})
        n_rows, episodes = 0, []
        with Pool(args.workers) as pool:
            # One batch of episodes in memory at a time; imap let finished episodes pile up behind the writer.
            batched = (r for i in range(0, len(split_jobs), args.workers)
                       for r in pool.map(process_episode, split_jobs[i:i + args.workers]))
            for demo_id, rows in enumerate(batched):
                n = len(rows["frame_id"])
                rows["demo_id"] = [demo_id] * n
                for key, (shape, dtype) in schema.items():
                    zf[key].append(np.asarray(rows[key], dtype=dtype).reshape((n,) + shape))
                n_rows += n
                episodes.append([split_jobs[demo_id][0], n])
                print("[{}] demo {} {} rows={} total={}".format(split, demo_id, split_jobs[demo_id][0], n, n_rows), flush=True)
        summary[split] = {"rows": n_rows, "episodes": episodes}
    with open(os.path.join(args.out, "build_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print("[DONE] train rows={} val rows={}".format(summary["train"]["rows"], summary["val"]["rows"]))


if __name__ == "__main__":
    main()
