#!/bin/bash
#SBATCH --job-name=3dfa_profile
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=50
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --time=00:30:00
#SBATCH --partition=shubhamlong
#SBATCH --exclude=grogu-0-[14,19,24,29],grogu-1-14,grogu-3-25

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

cd /home/harshilb/3d_flowmatch_actor
source ~/miniconda3/etc/profile.d/conda.sh
conda activate 3dfa

MASTER_PORT=$((27500 + RANDOM % 1000))

data=orbital
main_dir=Orbital
experiment=debug
run_log_dir=profile_run

# Warmup steps before profiler captures, then capture this many steps
WARMUP=20
PROFILE_STEPS=10

train_data_dir=/grogu/user/harshilb/multi_cam/train.zarr
eval_data_dir=/grogu/user/harshilb/multi_cam/val.zarr

source /home/harshilb/3d_flowmatch_actor/scripts/helpers/slurm_utils.sh
to_scratch train_data_dir
to_scratch eval_data_dir

# No checkpoint — fresh start so we profile from step 0+warmup
torchrun --nproc_per_node 1 --master_port $MASTER_PORT \
    main.py \
    data=$data \
    experiment=$experiment \
    exp_log_dir=$main_dir \
    run_log_dir=$run_log_dir \
    train_iters=$((WARMUP + PROFILE_STEPS + 5)) \
    train_data_dir=$train_data_dir \
    eval_data_dir=$eval_data_dir \
    batch_size=64 \
    "orbital_miscal_noise_levels=[medium]" \
    use_compile=true \
    benchmark=true \
    benchmark_warmup_steps=$WARMUP \
    benchmark_profile_steps=$PROFILE_STEPS \
    benchmark_dummy_data=false
