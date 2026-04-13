"""
Convert raw orbital rollout episodes → single train.zarr.

Directory layout expected (from collect_orbital_rollouts.py):
  {root}/{task}/{group}/episode_{N}/
      orbital_left_rgb/   {0000..}.png
      orbital_left_depth/ {0000..}.png   (RGB-encoded float, RLBench convention)
      orbital_right_rgb/  {0000..}.png
      orbital_right_depth/{0000..}.png
      over_shoulder_left_rgb/   {0000..}.png
      over_shoulder_left_depth/ {0000..}.png
      over_shoulder_right_rgb/  {0000..}.png
      over_shoulder_right_depth/{0000..}.png
      low_dim_obs.pkl
      camera_group.txt

Zarr schema (matches Peract2_zarr / peract2_to_zarr.py):
  rgb              (N, NCAM=4, 3, H, W)   uint8
  depth            (N, NCAM=4, H, W)      float16
  extrinsics       (N, NCAM=4, 4, 4)      float16  cam-to-world
  intrinsics       (N, NCAM=4, 3, 3)      float16
  proprioception   (N, 3, NHAND=1, 8)     float32
  action           (N, 1, NHAND=1, 8)     float32
  proprioception_joints (N, 1, NHAND=1, 8) float32
  action_joints    (N, 1, NHAND=1, 8)     float32
  task_id          (N,)                   uint8
  variation        (N,)                   uint8
  camera_group     (N,)                   uint8   (1-6)

Camera order:  [orbital_left, orbital_right, wrist]
"""
import argparse
import json
import os
import pickle
import sys
from pathlib import Path

# Stub unpickler to avoid RLBench/PyRep/CoppeliaSim import chain
class Stub:
    def __init__(self, *args, **kwargs):
        pass
    def __setstate__(self, state):
        self.__dict__.update(state)
    def __getattr__(self, name):
        return self.__dict__.get(name, Stub())
    def __len__(self):
        if hasattr(self, '_observations'):
            return len(self._observations)
        return 0
    def __getitem__(self, key):
        if hasattr(self, '_observations'):
            return self._observations[key]
        if isinstance(key, int):
            return Stub()
        return self.__dict__.get(key, Stub())

class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'list': return list
        if name == 'dict': return dict
        try:
            return super().find_class(module, name)
        except (AttributeError, ModuleNotFoundError, ImportError):
            return Stub

import numpy as np
from numcodecs import Blosc
from PIL import Image
from tqdm import tqdm
import zarr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data_processing.rlbench_utils import keypoint_discovery, image_to_float_array

NCAM = 3
NHAND = 1
IM_SIZE = 256
DEPTH_SCALE = 2 ** 24 - 1

PERACT_TASKS = [
    "place_cups", "close_jar", "insert_onto_square_peg",
    "light_bulb_in", "meat_off_grill", "open_drawer",
    "place_shape_in_shape_sorter", "place_wine_at_rack_location",
    "push_buttons", "put_groceries_in_cupboard",
    "put_item_in_drawer", "put_money_in_safe", "reach_and_drag",
    "slide_block_to_color_target", "stack_blocks", "stack_cups",
    "sweep_to_dustpan_of_size", "turn_tap",
]

CAMERAS = [
    "orbital_left",
    "orbital_right",
    "wrist",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _num2id(i):
    return str(i).zfill(4)


def load_rgb(ep_path, cam, frame_id):
    """Load a PNG frame as (H, W, 3) uint8."""
    path = os.path.join(ep_path, "{}_rgb".format(cam),
                        "{}.png".format(_num2id(frame_id)))
    return np.array(Image.open(path).convert("RGB"))


def load_depth_metres(ep_path, cam, frame_id, obs, cam_key):
    """
    Load depth image as float32 metres.

    For over_shoulder cameras the PNG is RGB-encoded in the [0,1] range scaled by
    (far-near); near/far are stored in the demo observation's misc dict.
    For orbital cameras the PNG is RGB-encoded at DEPTH_SCALE (absolute metres).
    """
    path = os.path.join(ep_path, "{}_depth".format(cam),
                        "{}.png".format(_num2id(frame_id)))
    d_raw = image_to_float_array(Image.open(path), DEPTH_SCALE)
    if cam_key is not None:
        # over_shoulder cameras: value in [0,1], need to unpack near/far
        near = obs.misc.get("{}_camera_near".format(cam_key), 0.0)
        far  = obs.misc.get("{}_camera_far".format(cam_key), 4.0)
        return (near + d_raw * (far - near)).astype(np.float32)
    else:
        # orbital cameras: value already in metres (absolute scale)
        return d_raw.astype(np.float32)


def load_extrinsics_from_misc(obs, cam_key):
    """4×4 cam-to-world from obs.misc (over_shoulder cameras)."""
    key = "{}_camera_extrinsics".format(cam_key)
    E = obs.misc.get(key, None)
    if E is None:
        return np.eye(4, dtype=np.float32)
    return np.array(E, dtype=np.float32)


def load_intrinsics_from_misc(obs, cam_key):
    """3×3 intrinsic matrix from obs.misc (over_shoulder cameras)."""
    key = "{}_camera_intrinsics".format(cam_key)
    K = obs.misc.get(key, None)
    if K is None:
        return np.eye(3, dtype=np.float32)
    return np.array(K, dtype=np.float32)


def load_orbital_extrinsics(ep_path):
    """
    Load pre-saved orbital camera extrinsics from episode folder.
    The collector saves these as a small pkl if present, otherwise returns eyes.
    """
    path = os.path.join(ep_path, "orbital_extrinsics.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        return (
            np.array(data["left_extrinsics"],  dtype=np.float32),
            np.array(data["right_extrinsics"], dtype=np.float32),
            np.array(data["left_intrinsics"],  dtype=np.float32),
            np.array(data["right_intrinsics"], dtype=np.float32),
        )
    # Fallback: identity (should not happen with correct collection)
    eye4 = np.eye(4, dtype=np.float32)
    eye3 = np.eye(3, dtype=np.float32)
    return eye4, eye4, eye3, eye3


def get_group_id(ep_path, group_str):
    """Return integer group id (1-6) from camera_group.txt or group string."""
    txt = os.path.join(ep_path, "camera_group.txt")
    if os.path.exists(txt):
        with open(txt) as f:
            return int(f.read().strip()[1])  # "G3" → 3
    return int(group_str[1])


# ---------------------------------------------------------------------------
# Episode → zarr rows
# ---------------------------------------------------------------------------

def process_episode(ep_path, task_id, group_str, zarr_file, im_size=IM_SIZE):
    """
    Extract keyframes from one episode and append rows to zarr_file.
    Returns number of keyframes written.
    """
    low_dim_path = os.path.join(ep_path, "low_dim_obs.pkl")
    if not os.path.exists(low_dim_path):
        print("[WARN] Missing low_dim_obs.pkl in {}".format(ep_path))
        return 0

    with open(low_dim_path, "rb") as f:
        demo = CustomUnpickler(f).load()

    key_frames = keypoint_discovery(demo, bimanual=False)
    key_frames.insert(0, 0)
    if len(key_frames) < 2:
        print("[WARN] Not enough keyframes in {}".format(ep_path))
        return 0

    # Orbital extrinsics (fixed per episode)
    E_ol, E_or, K_ol, K_or = load_orbital_extrinsics(ep_path)

    group_id = get_group_id(ep_path, group_str)

    n_written = 0
    for idx, k in enumerate(key_frames[:-1]):
        obs      = demo[k]
        obs_next = demo[key_frames[idx + 1]]

        # ── RGB ───────────────────────────────────────────────────────────────
        rgb_list = []
        for cam in CAMERAS:
            img = load_rgb(ep_path, cam, k)          # (H, W, 3) uint8
            rgb_list.append(img.transpose(2, 0, 1))  # (3, H, W)
        rgb = np.stack(rgb_list)[np.newaxis]          # (1, NCAM, 3, H, W)

        # ── Depth ─────────────────────────────────────────────────────────────
        depth_list = []
        cam_keys = [None, None, "wrist"]
        for cam, cam_key in zip(CAMERAS, cam_keys):
            d = load_depth_metres(ep_path, cam, k, obs, cam_key)
            depth_list.append(d)
        depth = np.stack(depth_list).astype(np.float16)[np.newaxis]  # (1, NCAM, H, W)

        # ── Extrinsics ────────────────────────────────────────────────────────
        E_wrist = load_extrinsics_from_misc(obs, "wrist")
        extr  = np.stack([E_ol, E_or, E_wrist]).astype(np.float16)
        extr  = extr[np.newaxis]  # (1, NCAM, 4, 4)

        # ── Intrinsics ────────────────────────────────────────────────────────
        K_wrist = load_intrinsics_from_misc(obs, "wrist")
        intr  = np.stack([K_ol, K_or, K_wrist]).astype(np.float16)
        intr  = intr[np.newaxis]  # (1, NCAM, 3, 3)

        # ── Proprioception (EEF pose) ─────────────────────────────────────────
        def _eef_state(o):
            return np.concatenate([o.gripper_pose, [o.gripper_open]]).astype(np.float32)

        state      = _eef_state(obs)
        state_prev = _eef_state(demo[key_frames[max(0, idx - 1)]])
        state_pp   = _eef_state(demo[key_frames[max(0, idx - 2)]])
        prop = np.stack([state_pp, state_prev, state]).reshape(3, NHAND, 8)[np.newaxis]

        action = _eef_state(obs_next).reshape(1, NHAND, 8)[np.newaxis]

        # ── Joint space ───────────────────────────────────────────────────────
        def _joint_state(o):
            return np.concatenate([
                o.joint_positions, [o.gripper_open]
            ]).astype(np.float32)

        prop_j = _joint_state(obs).reshape(1, NHAND, 8)[np.newaxis]
        act_j  = _joint_state(obs_next).reshape(1, NHAND, 8)[np.newaxis]

        # ── Scalars ───────────────────────────────────────────────────────────
        tid    = np.array([task_id],  dtype=np.uint8)
        var    = np.array([0],        dtype=np.uint8)
        grp    = np.array([group_id], dtype=np.uint8)

        # ── Append ────────────────────────────────────────────────────────────
        zarr_file["rgb"].append(rgb)
        zarr_file["depth"].append(depth)
        zarr_file["extrinsics"].append(extr)
        zarr_file["intrinsics"].append(intr)
        zarr_file["proprioception"].append(prop)
        zarr_file["action"].append(action)
        zarr_file["proprioception_joints"].append(prop_j)
        zarr_file["action_joints"].append(act_j)
        zarr_file["task_id"].append(tid)
        zarr_file["variation"].append(var)
        zarr_file["camera_group"].append(grp)
        n_written += 1

    return n_written


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Convert orbital rollouts to train.zarr"
    )
    p.add_argument("--root",       required=True,
                   help="Root dir containing task/group/episode_* folders")
    p.add_argument("--out",        required=True,
                   help="Output zarr path (e.g. data/orbital_train.zarr)")
    p.add_argument("--image-size", type=int, default=IM_SIZE)
    p.add_argument("--tasks",      default=None,
                   help="Comma-separated task list (default: all 18 PerAct)")
    p.add_argument("--groups",     default=None,
                   help="Comma-separated camera groups to include (e.g. G2,G3). Default: all groups present.")
    p.add_argument("--overwrite",  action="store_true",
                   help="Remove existing zarr and rebuild")
    return p.parse_args()


def main():
    args = parse_args()

    tasks = PERACT_TASKS
    if args.tasks:
        tasks = [t.strip() for t in args.tasks.split(",")]
    task2id = {t: i for i, t in enumerate(PERACT_TASKS)}
    allowed_groups = None
    if args.groups:
        allowed_groups = set(g.strip() for g in args.groups.split(","))

    if os.path.exists(args.out):
        if args.overwrite:
            import shutil
            shutil.rmtree(args.out)
            print("[INFO] Removed existing zarr at {}".format(args.out))
        else:
            print("[SKIP] {} already exists. Use --overwrite to rebuild.".format(
                args.out))
            return

    im = args.image_size
    compressor = Blosc(cname="lz4", clevel=1, shuffle=Blosc.SHUFFLE)

    with zarr.open_group(args.out, mode="w") as zf:

        def _create(name, shape, dtype):
            zf.create_dataset(
                name, shape=(0,) + shape,
                chunks=(1,) + shape,
                compressor=compressor, dtype=dtype,
            )

        _create("rgb",                   (NCAM, 3, im, im), "uint8")
        _create("depth",                 (NCAM, im, im),    "float16")
        _create("extrinsics",            (NCAM, 4, 4),      "float16")
        _create("intrinsics",            (NCAM, 3, 3),      "float16")
        _create("proprioception",        (3, NHAND, 8),     "float32")
        _create("action",                (1, NHAND, 8),     "float32")
        _create("proprioception_joints", (1, NHAND, 8),     "float32")
        _create("action_joints",         (1, NHAND, 8),     "float32")
        _create("task_id",               (),                "uint8")
        _create("variation",             (),                "uint8")
        _create("camera_group",          (),                "uint8")

        total = 0
        for task in tasks:
            tid = task2id.get(task, 0)
            task_root = os.path.join(args.root, task)
            if not os.path.isdir(task_root):
                print("[SKIP] No data for task {}".format(task))
                continue

            groups = sorted(os.listdir(task_root))
            for group_str in groups:
                if allowed_groups is not None and group_str not in allowed_groups:
                    continue
                group_root = os.path.join(task_root, group_str)
                if not os.path.isdir(group_root):
                    continue
                episodes = sorted([
                    d for d in os.listdir(group_root)
                    if d.startswith("episode_") and
                       os.path.isdir(os.path.join(group_root, d))
                ])
                print("[{}] {} — {} episodes".format(
                    task, group_str, len(episodes)))
                for ep in tqdm(episodes, desc="{}/{}".format(task, group_str)):
                    ep_path = os.path.join(group_root, ep)
                    n = process_episode(ep_path, tid, group_str, zf, im)
                    total += n

        print("\n[DONE] Wrote {} keyframe rows to {}".format(total, args.out))
        for key in zf.keys():
            print("  {}: {}".format(key, zf[key].shape))


if __name__ == "__main__":
    main()
