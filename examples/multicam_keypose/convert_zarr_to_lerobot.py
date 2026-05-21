"""
Convert keypose zarr dataset to LeRobot format for pi_0.5 finetuning.

Usage:
    uv run examples/multicam_keypose/convert_zarr_to_lerobot.py

The script reads:
  - ZARR_PATH: zarr file with rgb, action, proprioception, demo_id, task_id, variation
  - INSTRUCTION_JSON: {task_name: {variation_str: [instructions]}} — RLBench format

Output is written to $HF_LEROBOT_HOME/local/multicam_keypose.
"""

import json
import shutil

import einops
import numpy as np
import zarr
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

REPO_NAME = "local/multicam_keypose"
ZARR_PATH = "/work/nvme/bgkz/harshilb/multi_cam/train.zarr"
INSTRUCTION_JSON = "/work/nvme/bgkz/harshilb/multi_cam/instruction.json"


def main():
    output_path = HF_LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        shutil.rmtree(output_path)

    with open(INSTRUCTION_JSON) as f:
        instructions = json.load(f)

    # task_id is an index into the ordered list of task names in the JSON.
    task_names = list(instructions.keys())

    z = zarr.open(ZARR_PATH, "r")
    rgb = z["rgb"][:]                  # (N, 3, 3, 256, 256) — (N, cams, C, H, W)
    action = z["action"][:]            # (N, 1, 1, 8)
    proprio = z["proprioception"][:]   # (N, 3, 1, 8)
    demo_ids = z["demo_id"][:]         # (N,)
    task_ids = z["task_id"][:]         # (N,) — index into task_names
    variations = z["variation"][:]     # (N,) — variation index within task

    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="panda",
        fps=1,
        features={
            "base_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "left_wrist_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "right_wrist_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    unique_demos = np.unique(demo_ids)
    print(f"Converting {len(unique_demos)} demos...")

    for demo_id in unique_demos:
        mask = demo_ids == demo_id
        indices = np.where(mask)[0]

        for i in indices:
            # rgb[i]: (3, 3, 256, 256) — (cams, C, H, W) → (H, W, C) per cam
            base_img = einops.rearrange(rgb[i, 0], "c h w -> h w c")
            left_img = einops.rearrange(rgb[i, 1], "c h w -> h w c")
            right_img = einops.rearrange(rgb[i, 2], "c h w -> h w c")

            state = proprio[i, -1, 0, :].astype(np.float32)  # latest history step
            act = action[i, 0, 0, :].astype(np.float32)

            task_name = task_names[task_ids[i]]
            variation_str = str(variations[i])
            # Pick the first of the available instruction paraphrases.
            prompt = instructions[task_name][variation_str][0]

            dataset.add_frame({
                "base_image": base_img,
                "left_wrist_image": left_img,
                "right_wrist_image": right_img,
                "state": state,
                "actions": act,
                "task": prompt,
            })

        dataset.save_episode()
        if (demo_id + 1) % 50 == 0:
            print(f"  {demo_id + 1}/{len(unique_demos)} demos done")

    print(f"Done. Dataset written to {output_path}")
    print(f"  Episodes: {len(unique_demos)}")
    print(f"  Total frames: {len(dataset)}")


if __name__ == "__main__":
    main()
