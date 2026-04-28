"""
PerAct (unimanual, 18 tasks) raw .dat data → zarr.

Optional transform (applied to all cameras):
  --rotate_x_deg, --rotate_y_deg, --rotate_z_deg  (degrees, default 0)
  --translate "dx,dy,dz"  (meters, default "0,0,0")
When rotation is set: RGB and PCD get 2D in-plane rotation; PCD also gets 3D rotation + translation.
Note: Post-hoc rotation can look bad (black corners, blur). For clean rotated views, collect
data with RLBench using --camera_rig_rotation_deg and build zarr without any --rotate_* / --translate.
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
ZARR_ROOT = "Peract2_zarr"
from data.processing.rlbench_utils import store_instructions, PERACT_TASKS


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


def apply_transform_to_peract_episode(rgb, pcd, rotate_x_deg, rotate_y_deg, rotate_z_deg, translate):
    """Apply rotation (x,y,z in degrees) and translation to one episode's rgb and pcd.

    rgb: (NCAM, 3, H, W) uint8. pcd: (NCAM, 3, H, W) float16.
    - RGB: 2D in-plane rotation by +rotate_z_deg (CCW) so the view matches the rotated scene.
    - PCD: 3D rotation R = Rz @ Ry @ Rx applied to (x,y,z) at each pixel, then +translate, then same 2D grid rotation.
    """
    if rotate_x_deg == 0 and rotate_y_deg == 0 and rotate_z_deg == 0 and translate == (0.0, 0.0, 0.0):
        return rgb, pcd
    # Match 3D Rz: positive rotate_z_deg = CCW in xy; scipy.ndimage.rotate uses positive = CCW.
    angle_2d = rotate_z_deg
    out_rgb = np.empty_like(rgb)
    out_pcd = np.empty_like(pcd)
    # 3D rotation matrix (intrinsic: apply Rx then Ry then Rz)
    R_3d = R.from_euler("xyz", [np.deg2rad(rotate_x_deg), np.deg2rad(rotate_y_deg), np.deg2rad(rotate_z_deg)]).as_matrix()
    t = np.array(translate, dtype=np.float32)
    for c in range(rgb.shape[0]):
        for ch in range(3):
            out_rgb[c, ch] = ndimage_rotate(
                rgb[c, ch], angle_2d, axes=(0, 1), reshape=False, order=1, mode="constant", cval=0
            )
        # PCD: (3, H, W) or (3, ...) -> apply R to each pixel's (x,y,z), add t, then 2D rotate grid
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
        return None
    if os.path.exists(filename) and overwrite:
        import shutil
        shutil.rmtree(filename)
        print(f"Removed existing {filename}")

    cameras = ["left_shoulder", "right_shoulder", "wrist", "front"]
    task2id = {task: t for t, task in enumerate(tasks)}

    # Initialize zarr
    compressor = Blosc(cname='lz4', clevel=1, shuffle=Blosc.SHUFFLE)
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

        n_rollouts = 0
        # Loop through episodes
        for task in tasks:
            print(task)
            episodes = []
            for var in range(0, 199):
                _path = Path(f'{ROOT}{split}/{task}+{var}/')
                if not _path.is_dir():
                    continue
                episodes.extend([
                    (ep, var) for ep in sorted(_path.glob("*.dat"))
                ])
            for ep, var in tqdm(episodes):
                # Read
                with open(ep, "rb") as f:
                    content = pickle.loads(blosc.decompress(f.read()))
                # Map [-1, 1] to [0, 255] uint8. Expect (NCAM, 3, H, W)
                rgb = (127.5 * (content[1][:, :, 0] + 1)).astype(np.uint8)
                pcd = content[1][:, :, 1].astype(np.float16)
                if rgb.ndim == 3:
                    rgb = rgb[np.newaxis, ...]  # (3,H,W) -> (1,3,H,W)
                if pcd.ndim == 3:
                    pcd = pcd[np.newaxis, ...]
                rgb, pcd = apply_transform_to_peract_episode(
                    rgb, pcd, rotate_x_deg, rotate_y_deg, rotate_z_deg, translate
                )
                # Store current eef pose as well as two previous ones
                prop = np.stack([
                    to_numpy(tens).astype(np.float32) for tens in content[4]
                ])
                prop_1 = np.concatenate([prop[:1], prop[:-1]])
                prop_2 = np.concatenate([prop_1[:1], prop_1[:-1]])
                prop = np.concatenate([prop_2, prop_1, prop], 1)
                prop = prop.reshape(len(prop), 3, NHAND, 8)
                # Next keypose (concatenate curr eef to form a "trajectory")
                actions = np.stack([
                    to_numpy(tens).astype(np.float32) for tens in content[2]
                ]).reshape(len(content[2]), 1, NHAND, 8)
                # Task ids and variation ids
                tids = np.array([task2id[task]] * len(content[0])).astype(np.uint8)
                _vars = np.array([var] * len(content[0])).astype(np.uint8)

                # write
                zarr_file['rgb'].append(rgb)
                zarr_file['pcd'].append(pcd)
                zarr_file['proprioception'].append(prop)
                zarr_file['action'].append(actions)
                zarr_file['task_id'].append(tids)
                zarr_file['variation'].append(_vars)
                zarr_file['demo_id'].append(np.full(len(content[0]), n_rollouts, dtype=np.uint32))
                n_rollouts += 1
                assert all(
                    len(zarr_file['action']) == len(zarr_file[key])
                    for key in zarr_file.keys()
                )


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
    if getattr(args, 'overwrite', False):
        print("[INFO] Overwrite: existing zarrs will be removed and rebuilt")
    for split in ['train', 'val']:
        all_tasks_main(
            split, PERACT_TASKS,
            rotate_x_deg=args.rotate_x_deg,
            rotate_y_deg=args.rotate_y_deg,
            rotate_z_deg=args.rotate_z_deg,
            translate=translate,
            overwrite=getattr(args, 'overwrite', False),
        )
    # Store instructions as json (can be run independently).
    # Only overwrite if the result is non-empty (prevents clobbering existing data
    # when ROOT points to Peract_packaged format which has no variation_descriptions.pkl).
    os.makedirs('instructions/peract', exist_ok=True)
    instr_dict = store_instructions(ROOT, PERACT_TASKS, ['train', 'val', 'test'])
    has_content = any(len(v) > 0 for v in instr_dict.values())
    if has_content:
        with open('instructions/peract/instructions.json', 'w') as fid:
            json.dump(instr_dict, fid)
        print(f"[INFO] Instructions written to instructions/peract/instructions.json")
    else:
        print(f"[INFO] No variation descriptions found at {ROOT} — instructions/peract/instructions.json unchanged")
