#!/bin/bash
#SBATCH --job-name=transfer_to_anvil
#SBATCH --output=logs/transfer_to_anvil_%A_%a.log
#SBATCH --error=logs/transfer_to_anvil_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --account=bgkz-delta-cpu
#SBATCH --partition=cpu
#SBATCH --array=0-5

DEST="x-hbhatia1@anvil.rcac.purdue.edu:/anvil/scratch/x-hbhatia1/"

DIRS=(
    /work/nvme/bgkz/droid_annotations
    /work/nvme/bgkz/droid_multilab_raw
    /work/nvme/bgkz/droid_multilab_depths
    /work/nvme/bgkz/droid_multilab_cache_ext2
    /work/nvme/bgkz/droid_multilab_depth_cam2cam_ext2
    /work/nvme/bgkz/droid_multilab_depth_metadata_ext2
)

DIR=${DIRS[$SLURM_ARRAY_TASK_ID]}
NAME=$(basename "$DIR")

echo "=== [$NAME] start: $(date) ==="
rsync -az --stats "$DIR" "$DEST"
echo "=== [$NAME] done: $(date) ==="
