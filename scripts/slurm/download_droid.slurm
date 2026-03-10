#!/bin/bash
#SBATCH --job-name=droid_download
#SBATCH --output=logs/droid_download_%j.log
#SBATCH --error=logs/droid_download_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=32
#SBATCH --partition=RM-shared   # adjust to your cluster's CPU partition name

source ~/.bashrc
conda activate groot  # env with lerobot installed

# ------------------------------------------------------------
# DROID RLDS download
# Usage:
#   SUBSET=small sbatch download_droid.slurm   # droid_100 sample subset
#   SUBSET=large sbatch download_droid.slurm   # larger/full DROID RLDS
# ------------------------------------------------------------

SUBSET=large 

DATA_ROOT=/ocean/projects/cis240058p/hbhatia1/data
SUBSET=${SUBSET:-small}  # small (droid_100) or large (full/large subset)

if [[ "$SUBSET" == "small" ]]; then
  SRC_URI="gs://gresearch/robotics/droid_100"
  TARGET_DIR="${DATA_ROOT}/droid_raw_small"
elif [[ "$SUBSET" == "large" ]]; then
  SRC_URI="gs://gresearch/robotics/droid"
  TARGET_DIR="${DATA_ROOT}/droid_raw_large"
else
  echo "Unknown SUBSET value: ${SUBSET} (expected 'small' or 'large')" >&2
  exit 1
fi

mkdir -p "${TARGET_DIR}"

if command -v gsutil >/dev/null 2>&1; then
  echo "Downloading DROID from ${SRC_URI} to ${TARGET_DIR}"
  gsutil -m cp -r "${SRC_URI}" "${TARGET_DIR}"
else
  echo "ERROR: gsutil not found in PATH. Install Google Cloud SDK or load the appropriate module, then re-run this job." >&2
  exit 1

fi


# python inspect_droid_rlds_and_annotations.py \
#     --source-dir /ocean/projects/cis240058p/hbhatia1/data/droid_raw_small \
#     --annotations-dir /ocean/projects/cis240058p/hbhatia1/data/droid_annotations \
#     --max-episodes 3





# ------------------------------------------------------------
# Legacy / alternative Hugging Face-based downloads (commented)
# ------------------------------------------------------------

# huggingface-cli login --token YOUR_HF_TOKEN_HERE

# annotations first (small)
# hf download \
#   --repo-type model KarlP/droid \
#   --local-dir ./data/droid_annotations

# small subset
# hf download \
#   --repo-type dataset lerobot/droid_100 \
#   --local-dir ./data/droid_100

# /ocean/projects/cis240058p/hbhatia1/data/droid_1.0.1_v2

# cd /ocean/projects/cis240058p/hbhatia1/data

# python /jet/home/hbhatia1/.conda/envs/groot/lib/python3.10/site-packages/lerobot/datasets/v2/convert_dataset_v1_to_v2.py \
#     --repo-id lerobot/droid_100 \
#     --local-dir data/droid_100 \
#     --push-to-hub 0

# # Step 2: v2 → v2.1
# python /jet/home/hbhatia1/.conda/envs/groot/lib/python3.10/site-packages/lerobot/datasets/v21/convert_dataset_v20_to_v21.py \
#     --repo-id lerobot/droid_100 \
#     --local-dir data/droid_100 \
#     --push-to-hub 0

# python -m lerobot.common.datasets.v21.convert_dataset_v1_to_v2 --repo-id lerobot/droid_100 --local-dir ./data/droid_100 --push-to-hub 0

# python -c "
# from lerobot.common.datasets.v21.convert_dataset_v1_to_v2 import convert_dataset
# convert_dataset(
#     repo_id='lerobot/droid_100',
#     local_dir='./data/droid_100',
#     push_to_hub=False,
# )
# "


# cd /ocean/projects/cis240058p/hbhatia1/data

# # Delete the current v3.0 copy
# rm -rf droid_100

# # Download specifically from the v2.1 branch
# huggingface-cli download \
#     --repo-type dataset \
#     --revision v2.1 \
#     lerobot/droid_100 \
#     --local-dir ./droid_100