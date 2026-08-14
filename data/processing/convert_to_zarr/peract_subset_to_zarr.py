"""
PerAct (unimanual) subset → zarr.

Extracts only the 6 tasks used in our main experiments:
  open_drawer, meat_off_grill, put_money_in_safe,
  slide_block_to_color_target, sweep_to_dustpan_of_size, turn_tap

Optional transform (applied to all cameras):
  --rotate_x_deg, --rotate_y_deg, --rotate_z_deg  (degrees, default 0)
  --translate "dx,dy,dz"  (meters, default "0,0,0")
"""
import argparse
import json
import os
from pathlib import Path
import pickle

import blosc
from numcodecs import Blosc
import zarr
import numpy as np
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import rotate as ndimage_rotate

RAW_ROOT = "peract2_raw"
ZARR_ROOT = "Peract_subset_zarr"

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from data.processing.rlbench_utils import store_instructions

SUBSET_TASKS = [
    "open_drawer",
    "meat_off_grill",
    "put_money_in_safe",
    "slide_block_to_color_target",
    "sweep_to_dustpan_of_size",
    "turn_tap",
]

STORE_EVERY = 1
NCAM = 4
NHAND = 1
IM_SIZE = 256


def parse_arguments():
    parser = argparse.ArgumentParser()
    arguments = [
        ('root', str, RAW_ROOT),
        ('tgt', str, ZARR_ROOT),
        ('rotate_x_deg', float, 0.0),
        ('rotate_y_deg', float, 0.0),
        ('rotate_z_deg', float, 0.0),
        ('translate', str, "0,0,0"),
    ]
    for arg in arguments:
        parser.add_argument(f'--{arg[0]}', type=arg[1], default=arg[2])
    parser.add_argument('--overwrite', action='store_true', help='Remove existing zarrs and rebuild')
    return parser.parse_args()


def _parse_translate(s: str):
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 3:
        raise ValueError(f"translate must be 'dx,dy,dz', got {s!r}")
    return tuple(float(x) for x in parts)


def apply_transform_to_episode(rgb, pcd, rotate_x_deg, rotate_y_deg, rotate_z_deg, translate):
    """Apply rotation (x,y,z in degrees) and translation to one episode's rgb and pcd.

    rgb: (NCAM, 3, H, W) uint8. pcd: (NCAM, 3, H, W) float16.
    """
    if rotate_x_deg == 0 and rotate_y_deg == 0 and rotate_z_deg == 0 and translate == (0.0, 0.0, 0.0):
        return rgb, pcd
    angle_2d = rotate_z_deg
    out_rgb = np.empty_like(rgb)
    out_pcd = np.empty_like(pcd)
    R_3d = R.from_euler("xyz", [np.deg2rad(rotate_x_deg), np.deg2rad(rotate_y_deg), np.deg2rad(rotate_z_deg)]).as_matrix()
    t = np.array(translate, dtype=np.float32)
    for c in range(rgb.shape[0]):
        for ch in range(3):
            out_rgb[c, ch] = ndimage_rotate(
                rgb[c, ch], angle_2d, axes=(0, 1), reshape=False, order=1, mode="constant", cval=0
            )
        p = pcd[c]
        if p.ndim != 3 or p.shape[0] != 3:
            out_pcd[c] = p
            continue
        p_flat = p.reshape(3, -1)
        p_rot = (R_3d @ p_flat).reshape(p.shape)
        p_rot += t.reshape(3, *([1] * (p.ndim - 1)))
        for ch in range(3):
            out_pcd[c, ch] = ndimage_rotate(
                p_rot[ch], angle_2d, axes=(0, 1), reshape=False, order=1, mode="constant", cval=0
            )
    return out_rgb, out_pcd


def to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    return np.array(x)


def all_tasks_main(split, tasks, rotate_x_deg=0.0, rotate_y_deg=0.0, rotate_z_deg=0.0, translate=(0.0, 0.0, 0.0), overwrite=False):
    filename = f"{STORE_PATH}/{split}.zarr"
    if os.path.exists(filename) and not overwrite:
        print(f"Zarr file {filename} already exists. Skipping... (use --overwrite to replace)")
        return 0
    if os.path.exists(filename) and overwrite:
        import shutil
        shutil.rmtree(filename)
        print(f"Removed existing {filename}")

    cameras = ["left_shoulder", "right_shoulder", "wrist", "front"]
    task2id = {task: t for t, task in enumerate(tasks)}

    compressor = Blosc(cname='lz4', clevel=1, shuffle=Blosc.SHUFFLE)
    n_rollouts = 0

    with zarr.open_group(filename, mode="w") as zarr_file:

        def _create(field, shape, dtype):
            zarr_file.create_dataset(
                field,
                shape=(0,) + shape,
                chunks=(STORE_EVERY,) + shape,
                compressor=compressor,
                dtype=dtype
            )

        _create("rgb", (NCAM, 3, IM_SIZE, IM_SIZE), "uint8")
        _create("pcd", (NCAM, 3, IM_SIZE, IM_SIZE), "float16")
        _create("proprioception", (3, NHAND, 8), "float32")
        _create("action", (1, NHAND, 8), "float32")
        _create("task_id", (), "uint8")
        _create("variation", (), "uint8")
        _create("demo_id", (), "uint32")

        for task in tasks:
            print(f"[{split}] Processing task: {task}")
            episodes = []
            for var in range(0, 199):
                _path = Path(f'{ROOT}{split}/{task}+{var}/')
                if not _path.is_dir():
                    continue
                episodes.extend([
                    (ep, var) for ep in sorted(_path.glob("*.dat"))
                ])

            if not episodes:
                print(f"[WARN] No episodes found for {task} in {ROOT}{split}/")
                continue

            for ep, var in tqdm(episodes, desc=task):
                with open(ep, "rb") as f:
                    content = pickle.loads(blosc.decompress(f.read()))

                rgb = (127.5 * (content[1][:, :, 0] + 1)).astype(np.uint8)
                pcd = content[1][:, :, 1].astype(np.float16)
                if rgb.ndim == 3:
                    rgb = rgb[np.newaxis, ...]
                if pcd.ndim == 3:
                    pcd = pcd[np.newaxis, ...]
                rgb, pcd = apply_transform_to_episode(
                    rgb, pcd, rotate_x_deg, rotate_y_deg, rotate_z_deg, translate
                )

                prop = np.stack([
                    to_numpy(tens).astype(np.float32) for tens in content[4]
                ])
                prop_1 = np.concatenate([prop[:1], prop[:-1]])
                prop_2 = np.concatenate([prop_1[:1], prop_1[:-1]])
                prop = np.concatenate([prop_2, prop_1, prop], 1)
                prop = prop.reshape(len(prop), 3, NHAND, 8)

                actions = np.stack([
                    to_numpy(tens).astype(np.float32) for tens in content[2]
                ]).reshape(len(content[2]), 1, NHAND, 8)

                tids = np.array([task2id[task]] * len(content[0])).astype(np.uint8)
                _vars = np.array([var] * len(content[0])).astype(np.uint8)

                zarr_file['rgb'].append(rgb)
                zarr_file['pcd'].append(pcd)
                zarr_file['proprioception'].append(prop)
                zarr_file['action'].append(actions)
                zarr_file['task_id'].append(tids)
                zarr_file['variation'].append(_vars)
                zarr_file['demo_id'].append(np.full(len(content[0]), n_rollouts, dtype=np.uint32))
                n_rollouts += 1

    return n_rollouts


if __name__ == "__main__":
    args = parse_arguments()
    ROOT = args.root
    STORE_PATH = args.tgt

    try:
        translate = _parse_translate(args.translate)
    except ValueError as e:
        raise SystemExit(f"Invalid --translate: {e}") from e

    if args.rotate_x_deg != 0 or args.rotate_y_deg != 0 or args.rotate_z_deg != 0 or translate != (0.0, 0.0, 0.0):
        print(f"[INFO] Transform: rotate_x={args.rotate_x_deg}° rotate_y={args.rotate_y_deg}° rotate_z={args.rotate_z_deg}° translate={translate}")
    if args.overwrite:
        print("[INFO] Overwrite: existing zarrs will be removed and rebuilt")

    print(f"[INFO] Tasks: {SUBSET_TASKS}")
    os.makedirs(STORE_PATH, exist_ok=True)

    total = 0
    for split in ['train', 'val']:
        n = all_tasks_main(
            split, SUBSET_TASKS,
            rotate_x_deg=args.rotate_x_deg,
            rotate_y_deg=args.rotate_y_deg,
            rotate_z_deg=args.rotate_z_deg,
            translate=translate,
            overwrite=args.overwrite,
        )
        print(f"[{split}] {n} rollouts written")
        total += n
    print(f"[TOTAL] {total} rollouts")

    os.makedirs('instructions/peract_subset', exist_ok=True)
    instr_dict = store_instructions(ROOT, SUBSET_TASKS, ['train', 'val', 'test'])
    has_content = any(len(v) > 0 for v in instr_dict.values())
    if has_content:
        with open('instructions/peract_subset/instructions.json', 'w') as fid:
            json.dump(instr_dict, fid)
        print("[INFO] Instructions written to instructions/peract_subset/instructions.json")
    else:
        print(f"[INFO] No variation descriptions found at {ROOT} — instructions/peract_subset/instructions.json unchanged")
