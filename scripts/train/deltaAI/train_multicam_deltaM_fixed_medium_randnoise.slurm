#!/bin/bash
#SBATCH --job-name=multicam_deltaM_med
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --partition=ghx4
#SBATCH --account=bgkz-dtai-gh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:2
#SBATCH --mem=96G
#SBATCH --time=48:00:00

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

export PYTHONPATH=/projects/bgkz/hbhatia1/vla/3d-vla:$PYTHONPATH
export BLOSC_NTHREADS=1

cd /work/nvme/bgkz/harshilb/3d-vla
source /u/hbhatia1/micromamba/etc/profile.d/mamba.sh
micromamba activate 3dfa

train_data_dir=/work/nvme/bgkz/harshilb/multi_cam/train.zarr
eval_data_dir=/work/nvme/bgkz/harshilb/multi_cam/val.zarr

LOCAL_DATA=/tmp/$SLURM_JOB_ID
echo "Staging data to $LOCAL_DATA ..."
mkdir -p $LOCAL_DATA
rsync -a --info=progress2 $train_data_dir $LOCAL_DATA/
rsync -a --info=progress2 $eval_data_dir $LOCAL_DATA/
echo "Data staging complete."

train_data_dir=$LOCAL_DATA/train.zarr
eval_data_dir=$LOCAL_DATA/val.zarr

MASTER_PORT=$((27500 + RANDOM % 1000))
ngpus=${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}

data=orbital
main_dir=Orbital
experiment=orb_deltaM_full_fixed_medium_randnoise
run_log_dir=multicam_deltaM_fixed_medium_randnoise
train_epochs=5000
checkpoint="${CHECKPOINT:-}"

WANDB_API_KEY=$WANDB_API_KEY torchrun --nproc_per_node $ngpus --master_port $MASTER_PORT \
    main.py \
    data=$data \
    experiment=$experiment \
    exp_log_dir=$main_dir \
    run_log_dir=$run_log_dir \
    train_epochs=$train_epochs \
    base_log_dir=/work/hdd/bgkz/hbhatia1/train_logs \
    train_data_dir=$train_data_dir \
    eval_data_dir=$eval_data_dir \
    batch_size=512 \
    ${checkpoint:+checkpoint=$checkpoint} \
    "$@"
