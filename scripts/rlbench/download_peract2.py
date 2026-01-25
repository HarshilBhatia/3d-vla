import argparse
import os
import subprocess


def parse_arguments():
    parser = argparse.ArgumentParser()
    # Tuples: (name, type, default)
    arguments = [
        ('root', str, '/data/group_data/katefgroup/VLA/peract2_raw_squash')
    ]
    for arg in arguments:
        parser.add_argument(f'--{arg[0]}', type=arg[1], default=arg[2])

    return parser.parse_args()


args = parse_arguments()
STORE_PATH = args.root
LINK = 'https://dataset.cs.washington.edu/fox/bimanual/image_size_256'

tasks = [
    # 'bimanual_push_box',
    # 'bimanual_lift_ball',
    # 'bimanual_dual_push_buttons',
    # 'bimanual_pick_plate',
    # 'bimanual_put_item_in_drawer',
    # 'bimanual_put_bottle_in_fridge',
    # 'bimanual_handover_item',
    # 'bimanual_pick_laptop',
    # 'bimanual_straighten_rope',
    # 'bimanual_sweep_to_dustpan',
    'bimanual_lift_tray',
    # 'bimanual_handover_item_easy',
    # 'bimanual_take_tray_out_of_oven'
]


def run_step(label, command, cwd):
    print(f"[{label}] {command}", flush=True)
    subprocess.run(command, shell=True, check=True, cwd=cwd)


for split in ['train', 'val']:
    os.makedirs(f'{STORE_PATH}/{split}', exist_ok=True)
    for task in tasks:
        print(f"\n== {task} ({split}) ==", flush=True)
        if os.path.exists(f'{STORE_PATH}/{split}/{task}'):
            print("Already exists, skipping.", flush=True)
            continue
        run_step(
            "download",
            f"wget --no-check-certificate {LINK}/{task}.{split}.squashfs",
            cwd=f"{STORE_PATH}/{split}",
        )
        run_step(
            "unsquash",
            f"unsquashfs {task}.{split}.squashfs",
            cwd=f"{STORE_PATH}/{split}",
        )
        run_step(
            "rename",
            f"mv squashfs-root/ {task}",
            cwd=f"{STORE_PATH}/{split}",
        )
        run_step(
            "cleanup",
            f"rm {task}.{split}.squashfs",
            cwd=f"{STORE_PATH}/{split}",
        )
