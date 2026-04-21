#!/bin/bash
# Train on orbital_mini.zarr (unimanual, peract tasks)

cd /home/harshilb/3d_flowmatch_actor
source ~/miniconda3/etc/profile.d/conda.sh
conda activate 3dfa

MASTER_PORT=$((27500 + RANDOM % 1000))
ngpus=$(nvidia-smi -L | wc -l)

data=orbital
main_dir=Orbital
experiment=default
run_log_dir=orbital_mini

checkpoint=train_logs/${main_dir}/${run_log_dir}/last.pth

WANDB_API_KEY=$WANDB_API_KEY torchrun --nproc_per_node $ngpus --master_port $MASTER_PORT \
    main.py \
    data=$data \
    experiment=$experiment \
    exp_log_dir=$main_dir \
    run_log_dir=$run_log_dir \
    train_data_dir=/home/harshilb/data/orbital_mini.zarr \
    eval_data_dir=/home/harshilb/data/orbital_mini.zarr \
    checkpoint=$checkpoint
