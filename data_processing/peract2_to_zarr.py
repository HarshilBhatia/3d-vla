import argparse
import json
import os
import pickle
import sys

# Centralized Stub and Unpickler to avoid backend dependencies
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

from numcodecs import Blosc
import zarr
import numpy as np
from PIL import Image
from tqdm import tqdm
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from paths import RAW_ROOT, ZARR_ROOT
from data_processing.rlbench_utils import (
    keypoint_discovery,
    image_to_float_array,
    store_instructions
)

# =========================
# CONSTANTS
# =========================
NCAM = 3
NHAND = 2
IM_SIZE = 256
DEPTH_SCALE = 2**24 - 1

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
    ]

    for name, typ, default in arguments:
        parser.add_argument(f'--{name}', type=typ, default=default)

    return parser.parse_args()

# =========================
# ZARR CREATION
# =========================
def all_tasks_main(split, tasks):
    filename = f"{STORE_PATH}/{split}.zarr"

    if os.path.exists(filename):
        print(f"[SKIP] Zarr file {filename} already exists.")
        return

    task2id = {task: i for i, task in enumerate(tasks)}
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
                            f"{ep_folder}/{ep}/{cam}_rgb/rgb_{_num2id(k)}.png"
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
                                f"{ep_folder}/{ep}/{cam}_depth/depth_{_num2id(k)}.png"
                            ),
                            DEPTH_SCALE
                        )
                        near = demo[k].misc[f'{cam}_camera_near']
                        far = demo[k].misc[f'{cam}_camera_far']
                        cam_d.append(near + d * (far - near))
                    depth.append(np.stack(cam_d))
                depth = np.stack(depth).astype(np.float16)

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
                ]).astype(np.float16)

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

# =========================
# UTILS
# =========================
def _num2id(i):
    return str(i).zfill(4)

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

    for split in ['train', 'val']:
        all_tasks_main(split, tasks)

    os.makedirs('instructions/peract2', exist_ok=True)
    instr_dict = store_instructions(ROOT, tasks)
    with open('instructions/peract2/instructions.json', 'w') as f:
        json.dump(instr_dict, f, indent=2)
