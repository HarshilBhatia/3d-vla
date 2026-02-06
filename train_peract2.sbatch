#!/bin/bash
#SBATCH --job-name=3dvla
#SBATCH --partition=shubhamlong
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=logs/train-%j.out
#SBATCH --error=logs/train-%j.err

# Print some info
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Working directory: $(pwd)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Change to the project directory
cd /home/lzaceria/mscv/3dvla/3d-vla

# Activate conda environment
source /home/lzaceria/miniconda3/etc/profile.d/conda.sh
conda activate 3dfa

# Set environment variables if needed
export USER_NAME=LUQMAN  # For paths.py configuration
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Ensure log folders exist
mkdir -p logs

# Print GPU info
echo "GPU Info:"
nvidia-smi

# -----------------------
# Training config (override via sbatch --export=ALL,TASK=...,SEMANTIC_DPS_WEIGHT=...)
# -----------------------
main_dir=Peract2
TASK=${TASK:-bimanual_lift_tray}
DATA_PATH=$(pwd)

train_data_dir=$DATA_PATH/Peract2_zarr/$TASK/train.zarr
eval_data_dir=$DATA_PATH/Peract2_zarr/$TASK/val.zarr
train_instructions=instructions/peract2/instructions.json
val_instructions=instructions/peract2/instructions.json

# Fail fast if the dataset paths are wrong/missing (otherwise Zarr errors later).
if [ ! -d "$train_data_dir" ] || [ ! -f "$train_data_dir/.zgroup" ]; then
    echo "[ERROR] train_data_dir is not a valid zarr group: $train_data_dir"
    echo "        (Did you generate this task's zarr? If not, set TASK to one that exists.)"
    exit 1
fi
if [ ! -d "$eval_data_dir" ] || [ ! -f "$eval_data_dir/.zgroup" ]; then
    echo "[ERROR] eval_data_dir is not a valid zarr group: $eval_data_dir"
    exit 1
fi
if [ ! -f "$train_instructions" ]; then
    echo "[ERROR] train_instructions not found: $train_instructions"
    exit 1
fi
if [ ! -f "$val_instructions" ]; then
    echo "[ERROR] val_instructions not found: $val_instructions"
    exit 1
fi

dataset=Peract2_3dfront_3dwrist
num_workers=4
B=16
B_val=16
chunk_size=1
memory_limit=8

val_freq=500
eval_only=false
lr=1e-4
backbone_lr=1e-6
lr_scheduler=constant
wd=1e-10
train_iters=30000
use_compile=false
use_ema=false
lv2_batch_size=1

model_type=denoise3d
bimanual=true
keypose_only=true
pre_tokenize=true
workspace_normalizer_buffer=0.05

backbone=clip
finetune_backbone=false
finetune_text_encoder=false
fps_subsampling_factor=4

# LAST-Lifting
semantic_dps_weight=${SEMANTIC_DPS_WEIGHT:-0.7}

C=120
num_attn_heads=8
num_vis_instr_attn_layers=3
num_history=3

num_shared_attn_layers=4
relative_action=false
rotation_format=quat_xyzw
denoise_timesteps=5
denoise_model=rectified_flow

wandb_project=3dvla
wandb_name=last_lifting_w${semantic_dps_weight}

run_log_dir=$model_type-$dataset-C$C-B$B-lr$lr-$lr_scheduler-H$num_history-$denoise_model-LASTw${semantic_dps_weight}
checkpoint=train_logs/${main_dir}/${run_log_dir}/last.pth

ngpus=1

echo "Config:"
echo "  TASK=$TASK"
echo "  semantic_dps_weight=$semantic_dps_weight"

# Run the training script
echo "Starting training..."
torchrun --nproc_per_node $ngpus --master_port $RANDOM \
    main.py \
    --train_data_dir $train_data_dir \
    --eval_data_dir $eval_data_dir \
    --train_instructions $train_instructions \
    --val_instructions $val_instructions \
    --dataset $dataset \
    --num_workers $num_workers \
    --batch_size $B \
    --batch_size_val $B_val \
    --chunk_size $chunk_size \
    --memory_limit $memory_limit \
    --exp_log_dir $main_dir \
    --run_log_dir ${run_log_dir} \
    --checkpoint $checkpoint \
    --val_freq $val_freq \
    --eval_only $eval_only \
    --lr $lr \
    --backbone_lr $backbone_lr \
    --lr_scheduler $lr_scheduler \
    --wd $wd \
    --train_iters $train_iters \
    --use_compile $use_compile \
    --use_ema $use_ema \
    --lv2_batch_size $lv2_batch_size \
    --model_type $model_type \
    --bimanual $bimanual \
    --keypose_only $keypose_only \
    --pre_tokenize $pre_tokenize \
    --backbone $backbone \
    --finetune_backbone $finetune_backbone \
    --finetune_text_encoder $finetune_text_encoder \
    --fps_subsampling_factor $fps_subsampling_factor \
    --semantic_dps_weight $semantic_dps_weight \
    --embedding_dim $C \
    --num_attn_heads $num_attn_heads \
    --num_vis_instr_attn_layers $num_vis_instr_attn_layers \
    --num_history $num_history \
    --num_shared_attn_layers $num_shared_attn_layers \
    --workspace_normalizer_buffer $workspace_normalizer_buffer \
    --relative_action $relative_action \
    --rotation_format $rotation_format \
    --denoise_timesteps $denoise_timesteps \
    --denoise_model $denoise_model \
    --wandb_project $wandb_project \
    ${wandb_name:+--wandb_name $wandb_name} \
    --filter_tasks $TASK

echo "Job finished at: $(date)"
