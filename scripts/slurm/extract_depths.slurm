#!/bin/bash
#SBATCH --job-name=extract_depths
#SBATCH --output=logs/extract_depths_%A_%a.log
#SBATCH --error=logs/extract_depths_%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --account=bgkz-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --gres=gpu:1
#SBATCH --array=0-31

mkdir -p /u/hbhatia1/3d-vla/logs

apptainer exec --nv \
  --bind /work/nvme/bgkz:/work/nvme/bgkz \
  --bind /u/hbhatia1/.local/share/stereolabs/settings:/usr/local/zed/settings \
  --bind /u/hbhatia1/3d-vla:/u/hbhatia1/3d-vla \
  /u/hbhatia1/3d-vla/zed_4.0.sif \
  python3 /u/hbhatia1/3d-vla/data_processing/extract_svo_depth.py \
    --raw-dir /work/nvme/bgkz/droid_rail_raw \
    --output-dir /work/nvme/bgkz/droid_rail_depths \
    --rank ${SLURM_ARRAY_TASK_ID} \
    --world-size ${SLURM_ARRAY_TASK_COUNT}
