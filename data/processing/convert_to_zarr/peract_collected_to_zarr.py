"""
Self-collected PerAct unimanual raw demos → zarr.

Handles the RLBench output format from data/generation/generate.py:
  {root}/{task}/variation{N}/episodes/episode{M}/
    {cam}_rgb/{k}.png
    {cam}_depth/{k}.png
    low_dim_obs.pkl

Outputs {tgt}/train.zarr with keys:
  rgb          (N, NCAM, 3, H, W) uint8
  depth        (N, NCAM, H, W)    float16
  proprioception (N, 3, NHAND, 8) float32   (t-2, t-1, t EEF)
  action       (N, 1, NHAND, 8)   float32   (next keypose EEF)
  proprioception_joints (N, 1, NHAND, 8) float32
  action_joints        (N, 1, NHAND, 8)  float32
  extrinsics   (N, NCAM, 4, 4)   float16
  intrinsics   (N, NCAM, 3, 3)   float16
  task_id      (N,)               uint8
  variation    (N,)               uint8
"""
import argparse
import os
import pickle
import re
import sys

import numpy as np
from numcodecs import Blosc
from PIL import Image
from tqdm import tqdm
import zarr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

RAW_ROOT = "peract2_raw"
ZARR_ROOT = "Peract2_zarr"
from data.processing.rlbench_utils import (
    keypoint_discovery,
    image_to_float_array,
    store_instructions,
    CustomUnpickler,
    DEPTH_SCALE,
    PERACT_TASKS,
)

CAMERAS = ["left_shoulder", "right_shoulder", "wrist", "front"]
NCAM = 4
NHAND = 1
IM_SIZE = 128  # generate.py default; override with --im_size


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default=RAW_ROOT,
                        help='Path to raw collected data root')
    parser.add_argument('--tgt', type=str, default=ZARR_ROOT,
                        help='Output directory; writes {tgt}/train.zarr')
    parser.add_argument('--tasks', type=str, default=None,
                        help='Comma-separated task names (default: all PERACT_TASKS found under root)')
    parser.add_argument('--im_size', type=int, default=IM_SIZE,
                        help='Image size (default 128, matching generate.py default)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Remove existing zarr and rebuild')
    return parser.parse_args()


def _read_rgb(ep_path, cam, frame_idx, im_size):
    img = Image.open(f"{ep_path}/{cam}_rgb/{frame_idx}.png").convert("RGB")
    if img.size != (im_size, im_size):
        img = img.resize((im_size, im_size), Image.BILINEAR)
    arr = np.array(img, dtype=np.uint8)  # (H, W, 3)
    return arr.transpose(2, 0, 1)  # (3, H, W)


def _read_depth(ep_path, cam, frame_idx, near, far):
    img = Image.open(f"{ep_path}/{cam}_depth/{frame_idx}.png")
    d = image_to_float_array(img, DEPTH_SCALE)
    return (near + d * (far - near)).astype(np.float16)


def convert(root, store_path, tasks, im_size, overwrite):
    filename = f"{store_path}/train.zarr"

    if os.path.exists(filename):
        if not overwrite:
            print(f"[SKIP] {filename} already exists (--overwrite to rebuild)")
            return
        import shutil
        shutil.rmtree(filename)
        print(f"[INFO] Removed existing {filename}")

    os.makedirs(store_path, exist_ok=True)
    task2id = {t: i for i, t in enumerate(PERACT_TASKS)}
    compressor = Blosc(cname='lz4', clevel=1, shuffle=Blosc.SHUFFLE)

    with zarr.open_group(filename, mode="w") as zf:

        def _create(field, shape, dtype):
            zf.create_dataset(field, shape=(0,) + shape,
                              chunks=(1,) + shape,
                              compressor=compressor, dtype=dtype)

        _create("rgb",                   (NCAM, 3, im_size, im_size), "uint8")
        _create("depth",                 (NCAM, im_size, im_size),    "float16")
        _create("proprioception",        (3, NHAND, 8),               "float32")
        _create("action",                (1, NHAND, 8),               "float32")
        _create("proprioception_joints", (1, NHAND, 8),               "float32")
        _create("action_joints",         (1, NHAND, 8),               "float32")
        _create("extrinsics",            (NCAM, 4, 4),                "float16")
        _create("intrinsics",            (NCAM, 3, 3),                "float16")
        _create("task_id",               (),                          "uint8")
        _create("variation",             (),                          "uint8")

        for task in tasks:
            task_base = os.path.join(root, task)
            if not os.path.isdir(task_base):
                print(f"[WARN] Task folder not found: {task_base}")
                continue

            var_dirs = sorted(
                [d for d in os.listdir(task_base) if d.startswith("variation")],
                key=lambda x: int(re.search(r'\d+', x).group())
            )
            if not var_dirs:
                print(f"[WARN] No variation folders under {task_base}")
                continue

            all_episodes = []
            for var_dir in var_dirs:
                ep_base = os.path.join(task_base, var_dir, "episodes")
                if not os.path.isdir(ep_base):
                    continue
                var_id = int(re.search(r'\d+', var_dir).group())
                for ep in sorted(os.listdir(ep_base),
                                 key=lambda x: int(re.search(r'\d+', x).group())):
                    all_episodes.append((os.path.join(ep_base, ep), var_id))

            print(f"[{task}] {len(all_episodes)} episodes across {len(var_dirs)} variations")

            for ep_path, var_id in tqdm(all_episodes, desc=task):
                try:
                    with open(f"{ep_path}/low_dim_obs.pkl", 'rb') as f:
                        demo = CustomUnpickler(f).load()
                except Exception as e:
                    print(f"[WARN] Could not read {ep_path}: {e}")
                    continue

                key_frames = keypoint_discovery(demo, bimanual=False)
                key_frames.insert(0, 0)
                if len(key_frames) < 2:
                    continue

                obs_frames = key_frames[:-1]   # input frames (t)
                act_frames = key_frames[1:]    # action frames (t+1)

                # RGB: (T, NCAM, 3, H, W)
                rgb = np.stack([
                    np.stack([_read_rgb(ep_path, cam, k, im_size) for cam in CAMERAS])
                    for k in obs_frames
                ]).astype(np.uint8)

                # Depth: (T, NCAM, H, W)
                depth = np.stack([
                    np.stack([
                        _read_depth(
                            ep_path, cam, k,
                            demo[k].misc[f'{cam}_camera_near'],
                            demo[k].misc[f'{cam}_camera_far']
                        )
                        for cam in CAMERAS
                    ])
                    for k in obs_frames
                ])

                # EEF proprioception + actions (gripper_pose=7D + gripper_open=1D)
                def _eef(obs):
                    return np.concatenate([obs.gripper_pose, [obs.gripper_open]]).astype(np.float32)

                states = np.stack([_eef(demo[k]) for k in key_frames])  # (T+1, 8)
                prop = states[:-1]                                        # (T, 8)
                prop_1 = np.concatenate([prop[:1], prop[:-1]])
                prop_2 = np.concatenate([prop_1[:1], prop_1[:-1]])
                prop = np.stack([prop_2, prop_1, prop], axis=1)           # (T, 3, 8)
                prop = prop.reshape(len(prop), 3, NHAND, 8)
                actions = states[1:].reshape(len(states[1:]), 1, NHAND, 8)

                # Joint space
                def _joints(obs):
                    return np.concatenate([obs.joint_positions, [obs.gripper_open]]).astype(np.float32)

                # joint_positions is 7D, pad to 8 with gripper_open
                states_j = np.stack([_joints(demo[k]) for k in key_frames])  # (T+1, 8)
                prop_j = states_j[:-1].reshape(len(states_j[:-1]), 1, NHAND, 8)
                act_j  = states_j[1:].reshape(len(states_j[1:]),  1, NHAND, 8)

                # Extrinsics / intrinsics: (T, NCAM, 4, 4) and (T, NCAM, 3, 3)
                extrinsics = np.stack([
                    np.stack([demo[k].misc[f'{cam}_camera_extrinsics'] for cam in CAMERAS])
                    for k in obs_frames
                ]).astype(np.float16)

                intrinsics = np.stack([
                    np.stack([demo[k].misc[f'{cam}_camera_intrinsics'] for cam in CAMERAS])
                    for k in obs_frames
                ]).astype(np.float16)

                T = len(obs_frames)
                task_id = np.full(T, task2id[task], dtype=np.uint8)
                variation = np.full(T, var_id, dtype=np.uint8)

                zf['rgb'].append(rgb)
                zf['depth'].append(depth)
                zf['proprioception'].append(prop)
                zf['action'].append(actions)
                zf['proprioception_joints'].append(prop_j)
                zf['action_joints'].append(act_j)
                zf['extrinsics'].append(extrinsics)
                zf['intrinsics'].append(intrinsics)
                zf['task_id'].append(task_id)
                zf['variation'].append(variation)

        print(f"\n[DONE] {len(zf['action'])} total keypose steps → {filename}")


if __name__ == "__main__":
    args = parse_arguments()

    if args.tasks is not None:
        tasks = [t.strip() for t in args.tasks.split(',')]
    else:
        # Auto-discover: any PERACT_TASKS folder present under root
        tasks = [t for t in PERACT_TASKS if os.path.isdir(os.path.join(args.root, t))]
        if not tasks:
            raise SystemExit(f"No PERACT_TASKS found under {args.root}")
        print(f"[INFO] Auto-discovered tasks: {tasks}")

    convert(args.root, args.tgt, tasks, args.im_size, args.overwrite)

    # Instructions
    os.makedirs('instructions/peract', exist_ok=True)
    instr = store_instructions(args.root, tasks, ['train', 'val', 'test'])
    if any(len(v) > 0 for v in instr.values()):
        import json
        with open('instructions/peract/instructions.json', 'w') as f:
            json.dump(instr, f)
        print("[INFO] Instructions written to instructions/peract/instructions.json")
