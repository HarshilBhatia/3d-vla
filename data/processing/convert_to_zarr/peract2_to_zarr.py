"""
PerAct2 raw demos → zarr (same layout as 3dfa/3D-VLA).

Camera extrinsics: 4x4 camera-to-world per camera, from RLBench/PyRep
(VisionSensor.get_matrix()). See PERACT2_CAMERA_EXTRINSICS.md.

Optional transform (RGB/depth unchanged): --rotate-extrinsics-deg (e.g. 10 around world Z)
and --translate-extrinsics "dx,dy,dz" in meters apply E_new = T_global @ E to all extrinsics.
"""
import argparse
import json
import os
import pickle
import sys
import time

from numcodecs import Blosc
import zarr
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

RAW_ROOT = "peract2_raw"
ZARR_ROOT = "Peract2_zarr"
from data.processing.rlbench_utils import (
    keypoint_discovery,
    image_to_float_array,
    store_instructions,
    CustomUnpickler,
    DEPTH_SCALE,
    num2id,
)
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import rotate as ndimage_rotate

# =========================
# CONSTANTS
# =========================
NCAM = 3
NHAND = 2
IM_SIZE = 256

CAMERAS = ["front", "wrist_left", "wrist_right"]

DEFAULT_TASKS = [
    'bimanual_push_box',
    'bimanual_lift_ball',
    'bimanual_dual_push_buttons',
    'bimanual_pick_plate',
    'bimanual_put_item_in_drawer',
    'bimanual_put_bottle_in_fridge',
    'bimanual_handover_item',
    'bimanual_pick_laptop',
    'bimanual_straighten_rope',
    'bimanual_sweep_to_dustpan',
    'bimanual_lift_tray',
    'bimanual_handover_item_easy',
    'bimanual_take_tray_out_of_oven'
]

# =========================
# ARGUMENTS
# =========================
def parse_arguments():
    parser = argparse.ArgumentParser()

    arguments = [
        ('root', str, RAW_ROOT),
        ('tgt', str, ZARR_ROOT),
        ('tasks', str, None),  # comma-separated list
        ('rotate_extrinsics_deg', float, 0.0),  # e.g. 10 for +10 deg around world Z
        ('translate_extrinsics', str, "0,0,0"),  # dx,dy,dz in meters (world frame)
    ]

    for name, typ, default in arguments:
        parser.add_argument(f'--{name}', type=typ, default=default)

    return parser.parse_args()


def _parse_translate(s: str):
    """Parse 'dx,dy,dz' into (dx, dy, dz) floats."""
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 3:
        raise ValueError(f"translate must be 'dx,dy,dz', got {s!r}")
    return tuple(float(x) for x in parts)


def apply_extrinsics_transform(
    extrinsics: np.ndarray,
    rotate_deg: float = 0.0,
    rotate_axis: str = "z",
    translate: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Apply a global transform to camera-to-world matrices: E_new = T_global @ E.

    T_global = [R|t; 0 0 0 1]: rotation (degrees around axis) then translation (meters in world).
    extrinsics: (T, NCAM, 4, 4). Returns same shape, float16.
    """
    if rotate_deg == 0 and translate == (0.0, 0.0, 0.0):
        return extrinsics.astype(np.float16)
    T = np.eye(4, dtype=np.float64)
    if rotate_deg != 0:
        T[:3, :3] = R.from_euler(rotate_axis, np.deg2rad(rotate_deg)).as_matrix()
    T[:3, 3] = np.array(translate, dtype=np.float64)
    out = np.einsum("ij,...jk->...ik", T, extrinsics.astype(np.float64))
    return out.astype(np.float16)


def rotate_images_for_transform(
    rgb: np.ndarray,
    depth: np.ndarray,
    rotate_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate RGB and depth so the view matches rotated extrinsics.

    Camera +rotate_deg around Z => image content rotates -rotate_deg in the image plane.
    rgb: (T, NCAM, 3, H, W) uint8. depth: (T, NCAM, H, W) float16.
    """
    if rotate_deg == 0:
        return rgb, depth
    angle = -rotate_deg
    out_rgb = np.empty_like(rgb)
    out_depth = np.empty_like(depth)
    for t in range(rgb.shape[0]):
        for c in range(rgb.shape[1]):
            for ch in range(3):
                out_rgb[t, c, ch] = ndimage_rotate(
                    rgb[t, c, ch], angle, axes=(0, 1), reshape=False, order=1,
                    mode="constant", cval=0
                )
            out_depth[t, c] = ndimage_rotate(
                depth[t, c], angle, axes=(0, 1), reshape=False, order=1,
                mode="constant", cval=0
            )
    return out_rgb, out_depth


# =========================
# ZARR CREATION
# =========================
def all_tasks_main(split, tasks, rotate_extrinsics_deg: float = 0.0, translate_extrinsics: tuple[float, float, float] = (0.0, 0.0, 0.0)):
    """Process all tasks for one split; write to zarr. Returns number of rollouts (episodes) written."""
    filename = f"{STORE_PATH}/{split}.zarr"

    if os.path.exists(filename):
        print(f"[SKIP] Zarr file {filename} already exists.")
        return 0

    task2id = {task: i for i, task in enumerate(tasks)}
    n_rollouts = 0
    compressor = Blosc(cname='lz4', clevel=1, shuffle=Blosc.SHUFFLE)

    with zarr.open_group(filename, mode="w") as zarr_file:

        def _create(field, shape, dtype):
            zarr_file.create_dataset(
                field,
                shape=(0,) + shape,
                chunks=(1,) + shape,
                compressor=compressor,
                dtype=dtype
            )

        _create("rgb", (NCAM, 3, IM_SIZE, IM_SIZE), "uint8")
        _create("depth", (NCAM, IM_SIZE, IM_SIZE), "float16")
        _create("proprioception", (3, NHAND, 8), "float32")
        _create("action", (1, NHAND, 8), "float32")
        _create("proprioception_joints", (1, NHAND, 8), "float32")
        _create("action_joints", (1, NHAND, 8), "float32")
        _create("extrinsics", (NCAM, 4, 4), "float16")
        _create("intrinsics", (NCAM, 3, 3), "float16")
        _create("task_id", (), "uint8")
        _create("variation", (), "uint8")
        _create("demo_id", (), "uint32")

        for task in tasks:
            print(f"[{split}] Processing task: {task}")
            
            # Try different path structures
            possible_task_folders = [
                f'{ROOT}/{split}/{task}/all_variations/episodes',
                f'{ROOT}/{task}/all_variations/episodes',
            ]
            
            task_folder = None
            for folder in possible_task_folders:
                if os.path.exists(folder):
                    task_folder = folder
                    break
            
            if task_folder is None:
                # Try variation structure: ROOT/task/variation0/episodes
                task_base = f'{ROOT}/{task}'
                if not os.path.exists(task_base):
                    # Try with split prefix
                    task_base = f'{ROOT}/{split}/{task}'
                
                if os.path.exists(task_base):
                    variation_folders = sorted([
                        f for f in os.listdir(task_base) 
                        if f.startswith('variation')
                    ])
                    
                    if variation_folders:
                        all_episodes = []
                        for var_dir in variation_folders:
                            v_folder = f'{task_base}/{var_dir}/episodes'
                            if os.path.exists(v_folder):
                                for ep in sorted(os.listdir(v_folder)):
                                    all_episodes.append((v_folder, ep))
                    else:
                        print(f"[WARN] Could not find episodes for task {task} in {ROOT}")
                        continue
                else:
                    print(f"[WARN] Could not find task folder for {task} in {ROOT}")
                    continue
            else:
                all_episodes = [(task_folder, ep) for ep in sorted(os.listdir(task_folder))]

            for ep_folder, ep in tqdm(all_episodes):
                with open(f"{ep_folder}/{ep}/low_dim_obs.pkl", 'rb') as f:
                    demo = CustomUnpickler(f).load()

                key_frames = keypoint_discovery(demo, bimanual=True)
                key_frames.insert(0, 0)

                # RGB
                rgb = np.stack([
                    np.stack([
                        np.array(Image.open(
                            f"{ep_folder}/{ep}/{cam}_rgb/rgb_{num2id(k)}.png"
                        ))
                        for cam in CAMERAS
                    ])
                    for k in key_frames[:-1]
                ]).transpose(0, 1, 4, 2, 3)

                # Depth
                depth = []
                for k in key_frames[:-1]:
                    cam_d = []
                    for cam in CAMERAS:
                        d = image_to_float_array(
                            Image.open(
                                f"{ep_folder}/{ep}/{cam}_depth/depth_{num2id(k)}.png"
                            ),
                            DEPTH_SCALE
                        )
                        near = demo[k].misc[f'{cam}_camera_near']
                        far = demo[k].misc[f'{cam}_camera_far']
                        cam_d.append(near + d * (far - near))
                    depth.append(np.stack(cam_d))
                depth = np.stack(depth).astype(np.float16)

                # When applying rotation to extrinsics, rotate images so the view matches
                if rotate_extrinsics_deg != 0:
                    rgb, depth = rotate_images_for_transform(rgb, depth, rotate_extrinsics_deg)

                # Proprioception (EEF)
                states = np.stack([
                    np.concatenate([
                        demo[k].left.gripper_pose, [demo[k].left.gripper_open],
                        demo[k].right.gripper_pose, [demo[k].right.gripper_open]
                    ])
                    for k in key_frames
                ]).astype(np.float32)

                prop = states[:-1]
                prop_1 = np.concatenate([prop[:1], prop[:-1]])
                prop_2 = np.concatenate([prop_1[:1], prop_1[:-1]])
                prop = np.concatenate([prop_2, prop_1, prop], axis=1)
                prop = prop.reshape(len(prop), 3, NHAND, 8)

                actions = states[1:].reshape(len(states[1:]), 1, NHAND, 8)

                # Joint space
                states_j = np.stack([
                    np.concatenate([
                        demo[k].left.joint_positions, [demo[k].left.gripper_open],
                        demo[k].right.joint_positions, [demo[k].right.gripper_open]
                    ])
                    for k in key_frames
                ]).astype(np.float32)

                prop_j = states_j[:-1].reshape(len(states_j[:-1]), 1, NHAND, 8)
                act_j = states_j[1:].reshape(len(states_j[1:]), 1, NHAND, 8)

                # Cameras
                extrinsics = np.stack([
                    np.stack([demo[k].misc[f'{cam}_camera_extrinsics'] for cam in CAMERAS])
                    for k in key_frames[:-1]
                ])
                extrinsics = apply_extrinsics_transform(
                    extrinsics,
                    rotate_deg=rotate_extrinsics_deg,
                    translate=translate_extrinsics,
                )

                intrinsics = np.stack([
                    np.stack([demo[k].misc[f'{cam}_camera_intrinsics'] for cam in CAMERAS])
                    for k in key_frames[:-1]
                ]).astype(np.float16)

                task_id = np.full(len(key_frames[:-1]), task2id[task], dtype=np.uint8)

                var_file = f"{ep_folder}/{ep}/variation_number.pkl"
                if os.path.exists(var_file):
                    with open(var_file, 'rb') as f:
                        var = int(pickle.load(f))
                else:
                    # Fallback to variation folder name if available
                    import re
                    match = re.search(r'variation(\d+)', ep_folder)
                    var = int(match.group(1)) if match else 0
                
                variation = np.full(len(key_frames[:-1]), var, dtype=np.uint8)

                # Write
                zarr_file['rgb'].append(rgb)
                zarr_file['depth'].append(depth)
                zarr_file['proprioception'].append(prop)
                zarr_file['action'].append(actions)
                zarr_file['proprioception_joints'].append(prop_j)
                zarr_file['action_joints'].append(act_j)
                zarr_file['extrinsics'].append(extrinsics)
                zarr_file['intrinsics'].append(intrinsics)
                zarr_file['task_id'].append(task_id)
                zarr_file['variation'].append(variation)
                zarr_file['demo_id'].append(np.full(len(key_frames[:-1]), n_rollouts, dtype=np.uint32))
                n_rollouts += 1
        return n_rollouts

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    args = parse_arguments()
    ROOT = args.root
    STORE_PATH = args.tgt

    if args.tasks is not None:
        tasks = args.tasks.split(',')
        for t in tasks:
            if t not in DEFAULT_TASKS:
                raise ValueError(f"Unknown task: {t}")
        print(f"[INFO] Using tasks from CLI: {tasks}")
    else:
        tasks = DEFAULT_TASKS
        print("[INFO] Using default PerAct2 task list")

    os.makedirs(STORE_PATH, exist_ok=True)

    translate = (0.0, 0.0, 0.0)
    try:
        translate = _parse_translate(args.translate_extrinsics)
    except ValueError as e:
        raise SystemExit(f"Invalid --translate_extrinsics: {e}") from e
    if args.rotate_extrinsics_deg != 0 or translate != (0.0, 0.0, 0.0):
        print(f"[INFO] Transform: rotate {args.rotate_extrinsics_deg}° around Z, translate {translate} (m)")

    total_rollouts = 0
    total_start = time.perf_counter()
    for split in ['train', 'val']:
        split_start = time.perf_counter()
        n = all_tasks_main(
            split, tasks,
            rotate_extrinsics_deg=args.rotate_extrinsics_deg,
            translate_extrinsics=translate,
        )
        total_rollouts += n
        elapsed = time.perf_counter() - split_start
        print(f"[TIME] {split}: {n} rollouts in {elapsed:.1f}s")

    total_elapsed = time.perf_counter() - total_start
    print(f"[TOTAL] {total_rollouts} rollouts in {total_elapsed:.1f}s ({total_elapsed / 60:.1f} min)")

    os.makedirs('instructions/peract2', exist_ok=True)
    instr_dict = store_instructions(ROOT, tasks)
    with open('instructions/peract2/instructions.json', 'w') as f:
        json.dump(instr_dict, f, indent=2)
