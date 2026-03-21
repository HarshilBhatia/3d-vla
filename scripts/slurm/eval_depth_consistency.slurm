#!/bin/bash
#SBATCH --job-name=eval_depth_consistency
#SBATCH --output=/u/hbhatia1/3d-vla/logs/eval_depth_consistency_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --account=bgkz-delta-cpu
#SBATCH --partition=cpu

source ~/.bashrc
micromamba activate gr00t
cd /u/hbhatia1/3d-vla
mkdir -p logs

echo "======== metadata ========"
python data_processing/eval_depth_consistency.py \
    --raw-dir /work/nvme/bgkz/droid_multilab_raw \
    --depth-dir /work/nvme/bgkz/droid_multilab_depths \
    --extr-source metadata

echo ""
echo "======== cam2cam ========"
python data_processing/eval_depth_consistency.py \
    --raw-dir /work/nvme/bgkz/droid_multilab_raw \
    --depth-dir /work/nvme/bgkz/droid_multilab_depths \
    --cam2cam-json /work/nvme/bgkz/droid_annotations/cam2cam_extrinsics.json \
    --extr-source cam2cam
