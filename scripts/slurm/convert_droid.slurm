#!/bin/bash

#SBATCH --job-name=convert
#SBATCH --output=logs/convert%j.log
#SBATCH --error=logs/convert%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=64
#SBATCH --partition=RM
#SBATCH --nodes=1

source ~/.bashrc
conda activate groot  # env with lerobot installed

export PYTHONUNBUFFERED=1

python -u convert_droid_rlds_to_lerobot_v2.py \
  --source-dir /ocean/projects/cis240058p/hbhatia1/data/droid_raw_large \
  --output-dir /ocean/projects/cis240058p/hbhatia1/data/droid_raw_large_superset \
  --cam2base-superset /ocean/projects/cis240058p/hbhatia1/data/droid_annotations/cam2base_extrinsic_superset.json \
  --num-workers 20