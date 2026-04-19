#!/bin/bash
#SBATCH --job-name=eval_orbital
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --partition=shubhamlong

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

cd /home/harshilb/3d_flowmatch_actor
source ~/miniconda3/etc/profile.d/conda.sh
conda activate 3dfa

MASTER_PORT=$((27500 + RANDOM % 1000))

COMMON="
    train_data_dir=/grogu/user/harshilb/orbital_train.zarr
    eval_data_dir=/grogu/user/harshilb/orbital_train.zarr
    train_instructions=instructions/peract/instructions.json
    val_instructions=instructions/peract/instructions.json
    dataset=Peract2_3dfront_3dwrist
    num_workers=4
    batch_size=32
    batch_size_val=32
    chunk_size=1
    memory_limit=8
    exp_log_dir=Orbital
    eval_only=true
    lr=1e-4
    backbone_lr=1e-6
    lr_scheduler=constant
    wd=1e-10
    train_iters=100000
    use_compile=false
    use_ema=false
    lv2_batch_size=1
    model_type=denoise3d
    bimanual=false
    keypose_only=true
    pre_tokenize=true
    backbone=clip
    finetune_backbone=false
    finetune_text_encoder=false
    fps_subsampling_factor=4
    embedding_dim=120
    num_attn_heads=8
    num_vis_instr_attn_layers=2
    num_history=3
    num_shared_attn_layers=4
    workspace_normalizer_buffer=0.04
    relative_action=false
    rotation_format=quat_xyzw
    denoise_timesteps=5
    denoise_model=rectified_flow
    use_wandb=false
    learn_extrinsics=false
    predict_extrinsics=false
    extrinsics_prediction_mode=delta_m
    dynamic_rope_from_camtoken=false
    use_front_camera_frame=false
    pc_rotate_by_front_camera=true
    rope_type=normal
    rope_schedule_type=linear
    rope_schedule_steps=100000
"

echo "===== Evaluating Nlabs_3dfa (traj_scene_rope=true) ====="
torchrun --nproc_per_node 1 --master_port $MASTER_PORT \
    main.py $COMMON \
    run_log_dir=Nlabs_3dfa \
    checkpoint=train_logs/Orbital/Nlabs_3dfa/last.pth \
    traj_scene_rope=true

MASTER_PORT=$((27500 + RANDOM % 1000))

echo "===== Evaluating Nlabs_baselin (traj_scene_rope=false) ====="
torchrun --nproc_per_node 1 --master_port $MASTER_PORT \
    main.py $COMMON \
    run_log_dir=Nlabs_baselin \
    checkpoint=train_logs/Orbital/Nlabs_baselin/last.pth \
    traj_scene_rope=false
