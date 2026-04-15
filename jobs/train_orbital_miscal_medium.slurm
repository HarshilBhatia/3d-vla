#!/bin/bash
#SBATCH --job-name=miscal_medium
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=50
#SBATCH --gres=gpu:4
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --partition=shubhamlong


echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

cd /home/harshilb/3d_flowmatch_actor
source ~/miniconda3/etc/profile.d/conda.sh
conda activate 3dfa

MASTER_PORT=$((27500 + RANDOM % 1000))
ngpus=$(nvidia-smi -L | wc -l)

data=orbital
main_dir=Orbital
experiment=orb_deltaM_full

run_log_dir=miscal_medium_deltaM_full
checkpoint=train_logs/${main_dir}/${run_log_dir}/interm260000.pth

WANDB_API_KEY=$WANDB_API_KEY torchrun --nproc_per_node $ngpus --master_port $MASTER_PORT \
    main.py \
    data=$data \
    experiment=$experiment \
    exp_log_dir=$main_dir \
    run_log_dir=$run_log_dir \
    checkpoint=$checkpoint \
    miscalibration_noise_level=medium
