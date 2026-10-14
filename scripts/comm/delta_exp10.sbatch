#!/usr/bin/env bash
# Resume one exp 10 run on Delta (2 x A100, batch 32/GPU like grogu). Usage: sbatch scripts/comm/delta_exp10.sbatch <suffix> <experiment>
# e.g. sbatch scripts/comm/delta_exp10.sbatch pooled_refine peract2_orbital_handover_dense_chunk8_pooled_refine
#SBATCH --account=bgkz-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --output=logs/delta/%x-%j.out
set -euo pipefail
SUFFIX="${1:?run suffix}"; EXPERIMENT="${2:?experiment name}"
source /u/hbhatia1/micromamba/etc/profile.d/mamba.sh && micromamba activate 3dfa
cd "$SLURM_SUBMIT_DIR"; export PYTHONPATH="$PWD:${PYTHONPATH:-}" BLOSC_NTHREADS=1
RUN=peract2_orbital_handover_dense_chunk8_$SUFFIX
D=/projects/bgkz/3dfa_dense/handover_chunk8
torchrun --nproc_per_node=2 --master_port=$((27500 + SLURM_JOB_ID % 1000)) main.py \
  data=orbital_peract2_nfs experiment="$EXPERIMENT" train_data_dir=$D/train.zarr eval_data_dir=$D/val.zarr \
  exp_log_dir=PerAct2 run_log_dir="$RUN" \
  miscal_mode=group miscal_group_level=medium miscal_group_file=instructions/orbital_miscalibration_noise_bimanual_external_only.json \
  miscal_cameras=[0,1] perturbation_noise_rot_deg=3.0 perturbation_noise_trans_m=0.01 video_deltam_cache_batches=0 \
  num_workers=4 prefetch_factor=2 memory_limit=2 checkpoint=train_logs/PerAct2/$RUN/last.pth
