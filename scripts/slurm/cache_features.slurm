#!/bin/bash
#SBATCH --job-name=droid_small_v2
#SBATCH --output=logs/droid_small_v2_%j.log
#SBATCH --error=logs/droid_small_v2_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1

source ~/.bashrc
conda activate groot

cd /ocean/projects/cis240058p/hbhatia1/Isaac-GR00T
python data/cache_backbone_features.py \
    --dataset-path /ocean/projects/cis240058p/hbhatia1/data/droid_raw_large_superset \
    --output-dir /ocean/projects/cis240058p/hbhatia1/data/droid_large_superset_cache \
    --batch-size 128
