#!/bin/bash
# Example script to visualize the dataset with Rerun

# First, install rerun if not already installed:
# pip install rerun-sdk

# Example 1: Visualize Peract dataset (save to file for server without display)
python visualize_rerun.py \
    --data_dir Peract2_zarr/bimanual_lift_tray/train.zarr \
    --instructions instructions/peract2/instructions_bimanual_lift_tray.json \
    --dataset Peract2_3dfront_3dwrist \
    --num_samples 10 \

# Example 2: Visualize RLBench dataset
# python visualize_rerun.py \
#     --data_dir /path/to/rlbench/data \
#     --instructions /path/to/rlbench/instructions.json \
#     --dataset Hiveformer \
#     --num_samples 10

# Example 3: Save visualization to file
# python visualize_rerun.py \
#     --data_dir /path/to/data \
#     --instructions /path/to/instructions.json \
#     --dataset Peract \
#     --num_samples 10 \
#     --save visualization.rrd

echo "Edit this script with your actual data paths and run it!"
echo "Or run directly: python visualize_rerun.py --data_dir <path> --instructions <path> --dataset <name>"
