#!/bin/bash
#SBATCH --job-name=debug_cfg_diff
#SBATCH --output=logs/debug_cfg_diff_%j.out
#SBATCH --error=logs/debug_cfg_diff_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=0:10:00
#SBATCH --partition=all
#SBATCH --exclude=grogu-0-[14,19,29],grogu-1-[6,9,14],grogu-2-22,grogu-3-25

set -euo pipefail

cd /home/harshilb/3d_flowmatch_actor
source ~/miniconda3/etc/profile.d/conda.sh
conda activate 3dfa

python scripts/eval/eval_checkpoints_mse_cfg.py \
    checkpoints=train_logs/final/siglip_multi_group/best.pth \
    data_paths=/grogu/user/harshilb/open_drawer/multi/train.zarr \
    val_instructions=instructions/peract/instructions.json \
    bimanual=false \
    cfg_scale=1 \
    num_batches=1 \
    batch_size_val=4 \
    output_csv=/tmp/cfg_debug_mg_2.csv
