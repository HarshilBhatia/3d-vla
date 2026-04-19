#!/usr/bin/env bash
# Quick benchmark run: 10 training steps, logs timing/memory to
# train_logs/Orbital/benchmark/<rank>.txt
#
# Usage:
#   bash scripts/rlbench/benchmark_orbital.sh
#   bash scripts/rlbench/benchmark_orbital.sh path/to/existing.pth   # skip workspace normalizer

main_dir=Orbital

# --- Optionally copy data from NFS to local scratch (SYNC_TO_SCRATCH=1 to enable) ---
GROGU_BASE=/grogu/user/harshilb/datasets/Peract2_zarr/bimanual_lift_tray
train_data_dir=$GROGU_BASE/train.zarr
eval_data_dir=$GROGU_BASE/val.zarr

if [ "${SYNC_TO_SCRATCH:-0}" = "1" ]; then
    SCRATCH=/scratch/$USER/3dfa_benchmark
    echo "Syncing data to scratch ($SCRATCH)..."
    mkdir -p "$SCRATCH"
    rsync -a --info=progress2 "$GROGU_BASE/train.zarr" "$SCRATCH/"
    rsync -a --info=progress2 "$GROGU_BASE/val.zarr"   "$SCRATCH/"
    echo "Data ready on scratch."
    train_data_dir=$SCRATCH/train.zarr
    eval_data_dir=$SCRATCH/val.zarr
fi

train_instructions=instructions/peract/instructions.json
val_instructions=instructions/peract/instructions.json

dataset=Peract2_3dfront_3dwrist
num_workers=4
B=64
B_val=32
chunk_size=1
memory_limit=8

lr=1e-4
backbone_lr=1e-6
lr_scheduler=constant
wd=1e-10
use_compile=true
use_ema=false
lv2_batch_size=1

model_type=denoise3d
bimanual=true
keypose_only=true
pre_tokenize=true
workspace_normalizer_buffer=0.04

backbone=clip
finetune_backbone=false
finetune_text_encoder=false
fps_subsampling_factor=4

C=120
num_attn_heads=8
num_vis_instr_attn_layers=2
num_history=3
num_shared_attn_layers=4
relative_action=false
rotation_format=quat_xyzw
denoise_timesteps=5
denoise_model=rectified_flow

learn_extrinsics=False
predict_extrinsics=false
extrinsics_prediction_mode=delta_m
dynamic_rope_from_camtoken=false
use_front_camera_frame=false
pc_rotate_by_front_camera=false
traj_scene_rope=true
rope_type=normal
rope_schedule_type=linear

# --- benchmark-specific ---
BENCH_WARMUP=10
BENCH_STEPS=20   # first 10 = warmup (discarded), last 10 = measured
run_log_dir="lal_benchmark"
checkpoint=""

ngpus=$(nvidia-smi -L | wc -l)

echo "Running ${BENCH_STEPS}-step benchmark on ${ngpus} GPU(s)"
echo "Checkpoint: ${checkpoint}"
echo "Results → train_logs/${main_dir}/${run_log_dir}/benchmark_rank*.txt"

NCCL_DEBUG=WARN torchrun --nproc_per_node $ngpus --master_port $RANDOM \
    main.py \
    train_data_dir=$train_data_dir \
    eval_data_dir=$eval_data_dir \
    train_instructions=$train_instructions \
    val_instructions=$val_instructions \
    dataset=$dataset \
    num_workers=$num_workers \
    batch_size=$B \
    batch_size_val=$B_val \
    chunk_size=$chunk_size \
    memory_limit=$memory_limit \
    exp_log_dir=$main_dir \
    run_log_dir="${run_log_dir}" \
    eval_only=false \
    checkpoint=$checkpoint \
    lr=$lr \
    backbone_lr=$backbone_lr \
    lr_scheduler=$lr_scheduler \
    wd=$wd \
    use_compile=$use_compile \
    use_ema=$use_ema \
    lv2_batch_size=$lv2_batch_size \
    model_type=$model_type \
    bimanual=$bimanual \
    keypose_only=$keypose_only \
    pre_tokenize=$pre_tokenize \
    backbone=$backbone \
    finetune_backbone=$finetune_backbone \
    finetune_text_encoder=$finetune_text_encoder \
    fps_subsampling_factor=$fps_subsampling_factor \
    embedding_dim=$C \
    num_attn_heads=$num_attn_heads \
    num_vis_instr_attn_layers=$num_vis_instr_attn_layers \
    num_history=$num_history \
    num_shared_attn_layers=$num_shared_attn_layers \
    workspace_normalizer_buffer=$workspace_normalizer_buffer \
    relative_action=$relative_action \
    rotation_format=$rotation_format \
    denoise_timesteps=$denoise_timesteps \
    denoise_model=$denoise_model \
    learn_extrinsics=$learn_extrinsics \
    use_front_camera_frame=$use_front_camera_frame \
    pc_rotate_by_front_camera=$pc_rotate_by_front_camera \
    traj_scene_rope=$traj_scene_rope \
    predict_extrinsics=$predict_extrinsics \
    extrinsics_prediction_mode=$extrinsics_prediction_mode \
    dynamic_rope_from_camtoken=$dynamic_rope_from_camtoken \
    rope_type=$rope_type \
    rope_schedule_type=$rope_schedule_type \
    rope_schedule_steps=$BENCH_STEPS \
    use_wandb=false \
    train_iters=$BENCH_STEPS \
    benchmark=true \
    benchmark_warmup_steps=$BENCH_WARMUP \
    benchmark_log_freq=$BENCH_STEPS 
