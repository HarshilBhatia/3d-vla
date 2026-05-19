#!/bin/bash
#SBATCH --job-name=G3G4_ft_deltaM_eef
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=50
#SBATCH --gres=gpu:4
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --partition=shubhamlong
#SBATCH --exclude=grogu-0-[14,19,24,29],grogu-1-[6,9,14],grogu-3-25

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

cd /home/harshilb/3d_flowmatch_actor
source ~/miniconda3/etc/profile.d/conda.sh
conda activate 3dfa

MASTER_PORT=$((27500 + RANDOM % 1000))
ngpus=${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}

data=orbital
main_dir=final
experiment=orb_deltaM_full
run_log_dir=G3G4_finetune_deltaM_eef_final


train_epochs=5000

# Fine-tune from the G3G4 checkpoint (weights only, fresh optimizer + iter)
pretrained_checkpoint=train_logs/${main_dir}/multi_cam_G3G4/last.pth

# Resume within this run (after first launch, switch to checkpoint= for restarts)
checkpoint=train_logs/${main_dir}/${run_log_dir}/last.pth

train_data_dir=/grogu/user/harshilb/multi_cam/train.zarr
eval_data_dir=/grogu/user/harshilb/multi_cam/val.zarr

source /home/harshilb/3d_flowmatch_actor/scripts/helpers/slurm_utils.sh
to_scratch train_data_dir
to_scratch eval_data_dir

TORCHELASTIC_ERROR_FILE=/tmp/err.json WANDB_API_KEY=$WANDB_API_KEY torchrun --nproc_per_node $ngpus --master_port $MASTER_PORT \
    main.py \
    data=$data \
    experiment=$experiment \
    exp_log_dir=$main_dir \
    run_log_dir=$run_log_dir \
    train_iters=$train_iters \
    train_data_dir=$train_data_dir \
    eval_data_dir=$eval_data_dir \
    pretrained_checkpoint=$pretrained_checkpoint \
    checkpoint=$checkpoint \
    batch_size=256 \
    predict_ee_aux=true \
    train_epochs=$train_epochs \
    lambda_aux=1.0 \
    "ee_aux_cam_ids=[0,1]" \
    "cotrain_miscal_group_ids=[1,2]" \
    miscal_max_angle_deg=5.0 \
    miscal_max_translation_m=0.02 \
    "$@"
