#!/bin/bash
#SBATCH --job-name=orb_recursive
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=50
#SBATCH --gres=gpu:4
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --partition=all
#SBATCH --exclude=grogu-0-[14,19,24,29],grogu-1-14


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
data_dir='/grogu/user/harshilb/orbital_rollouts/4task_new.zarr'

run_log_dir=recursive_hist3
checkpoint=None
# train_logs/${main_dir}/${run_log_dir}/last.pth

WANDB_API_KEY=$WANDB_API_KEY torchrun --nproc_per_node $ngpus --master_port $MASTER_PORT \
    main.py \
    data=$data \
    experiment=$experiment \
    exp_log_dir=$main_dir \
    run_log_dir=$run_log_dir \
    num_history=3 \
    use_recursive_set_encoder=true \
    recursive_set_encoder_ncam=3 \
    train_data_dir=$data_dir \
    eval_data_dir=$data_dir \
    checkpoint=$checkpoint
    
