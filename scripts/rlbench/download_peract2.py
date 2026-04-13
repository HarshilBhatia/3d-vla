import argparse
import os
import sys
import subprocess

# Add project root to sys.path to find 'paths' module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from paths import RAW_ROOT


def parse_arguments():
    parser = argparse.ArgumentParser()
    # Tuples: (name, type, default)
    arguments = [
        ('root', str, RAW_ROOT),
        ('tasks', str, None)  # comma-separated list
    ]
    for arg in arguments:
        parser.add_argument(f'--{arg[0]}', type=arg[1], default=arg[2])

    return parser.parse_args()


args = parse_arguments()
STORE_PATH = args.root
LINK = 'https://dataset.cs.washington.edu/fox/bimanual/image_size_256'

ALL_TASKS = [
    'bimanual_push_box',
    # 'bimanual_lift_ball',
    # 'bimanual_dual_push_buttons',
    # 'bimanual_pick_plate',
    # 'bimanual_put_item_in_drawer',
    # 'bimanual_put_bottle_in_fridge',
    # 'bimanual_handover_item',
    # 'bimanual_pick_laptop',
    # 'bimanual_straighten_rope',
    # 'bimanual_sweep_to_dustpan',
    # 'bimanual_lift_tray',
    # 'bimanual_handover_item_easy',
    # 'bimanual_take_tray_out_of_oven'
]

if args.tasks is not None:
    tasks = args.tasks.split(',')
    for t in tasks:
        if t not in ALL_TASKS:
            print(f"[WARN] Task {t} not in default list.")
else:
    tasks = ALL_TASKS


# for split in ['train', 'val']:
for split in ['val']:
    os.makedirs(f'{STORE_PATH}/{split}', exist_ok=True)
    for task in tasks:
        if os.path.exists(f'{STORE_PATH}/{split}/{task}'):
            print(f"[SKIP] Task {task} ({split}) already exists in {STORE_PATH}")
            continue
        
        print(f"[INFO] Downloading {task} ({split})...")
        subprocess.run(
            f"wget --no-check-certificate -O {task}.{split}.squashfs {LINK}/{task}.{split}.squashfs",
            shell=True,
            check=True,
            cwd=f"{STORE_PATH}/{split}"
        )
        
        print(f"[INFO] Extracting {task} ({split})...")
        subprocess.run(
            f"unsquashfs {task}.{split}.squashfs",
            shell=True,
            check=True,
            cwd=f"{STORE_PATH}/{split}"
        )
        
        subprocess.run(
            f"mv squashfs-root/ {task}",
            shell=True,
            check=True,
            cwd=f"{STORE_PATH}/{split}"
        )
        
        subprocess.run(
            f"rm {task}.{split}.squashfs",
            shell=True,
            check=True,
            cwd=f"{STORE_PATH}/{split}"
        )
