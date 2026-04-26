#!/bin/bash
#SBATCH --job-name=orbital_mini
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=50
#SBATCH --gres=gpu:4
#SBATCH --mem=96G
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
experiment=default
run_log_dir=open_drawer_G1
train_iters=45000

checkpoint=train_logs/${main_dir}/${run_log_dir}/last.pth

TORCHELASTIC_ERROR_FILE=/tmp/err.json WANDB_API_KEY=$WANDB_API_KEY torchrun --nproc_per_node $ngpus --master_port $MASTER_PORT \
    main.py \
    data=$data \
    experiment=$experiment \
    exp_log_dir=$main_dir \
    run_log_dir=$run_log_dir \
    train_iters=$train_iters \
    train_data_dir=/grogu/user/harshilb/orbital_rollouts_mini/1task.zarr \
    eval_data_dir=/grogu/user/harshilb/orbital_rollouts_mini/1task.zarr \
    checkpoint=$checkpoint
