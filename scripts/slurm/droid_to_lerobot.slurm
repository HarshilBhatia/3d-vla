#!/bin/bash
#SBATCH --job-name=droid_small_v2
#SBATCH --output=logs/droid_small_v2_%j.log
#SBATCH --error=logs/droid_small_v2_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=RM-shared
# Uncomment and set if isaac-groot is a partition or account:
# #SBATCH --partition=isaac-groot
# #SBATCH --account=...

# Same small-dataset pipeline: droid_100 -> LeRobot v2. Run on isaac-groot (or current partition) to verify.
# Optional: limit episodes for a quick test.
#   MAX_EPISODES=5 sbatch run_small_droid_to_lerobot.slurm

# source ~/.bashrc
# conda activate groot

# PROJECT_ROOT=/ocean/projects/cis240058p/hbhatia1
# DATA_ROOT="${PROJECT_ROOT}/data"
# SOURCE_DIR="${DATA_ROOT}/droid_raw_small"
# OUTPUT_DIR="${DATA_ROOT}/droid_v2_small"
# MAX_EPISODES="${MAX_EPISODES:-}"

source ~/.bashrc
conda activate groot

# Quick test (1 episode)
# python /ocean/projects/cis240058p/hbhatia1/convert_droid_rlds_to_lerobot_v2.py \
#   --source-dir /ocean/projects/cis240058p/hbhatia1/data/droid_raw_small \
#   --output-dir /ocean/projects/cis240058p/hbhatia1/data/droid_v2_small \
#   --max-episodes 1

# Full small set
# python /ocean/projects/cis240058p/hbhatia1/convert_droid_rlds_to_lerobot_v2.py \
#   --source-dir /ocean/projects/cis240058p/hbhatia1/data/droid_raw_small \
#   --output-dir /ocean/projects/cis240058p/hbhatia1/data/droid_v2_small

python cache_backbone_features.py   --dataset-path /ocean/projects/cis240058p/hbhatia1/data/droid_raw_large_superset   --output-dir /ocean/projects/cis240058p/hbhatia1/data/droid_v2_large_superset_cache   --batch-size 128
