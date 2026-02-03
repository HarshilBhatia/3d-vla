"""Script for eval-only run and RoPE 3D frequency norm analysis."""

import argparse
import os
from pathlib import Path
import sys
from copy import deepcopy

import torch
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist

from datasets import fetch_dataset_class
from modeling.policy import fetch_model_class
from utils.common_utils import str2bool, str_none
from utils.trainers import fetch_train_tester
from utils.ema import EMA
from modeling.encoder.text import fetch_tokenizers


def parse_arguments():
    parser = argparse.ArgumentParser("Parse arguments for main.py")
    # Tuples: (name, type, default)
    arguments = [
        # Dataset/loader arguments
        ('train_data_dir', Path, ''),
        ('eval_data_dir', Path, ''),
        ('train_instructions', Path, ''),
        ('val_instructions', Path, ''),
        ('dataset', str, "Peract"),
        ('num_workers', int, 4),
        ('batch_size', int, 64),
        ('batch_size_val', int, 64),
        ('chunk_size', int, 1),
        ('memory_limit', float, 8),  # cache limit in GB
        # Logging arguments
        ('base_log_dir', Path, Path(__file__).parent / "train_logs"),
        ('exp_log_dir', Path, "exp"),
        ('run_log_dir', Path, "run"),
        # Wandb arguments
        ('use_wandb', str2bool, True),
        ('wandb_project', str, '3d_flowmatch_actor'),
        ('wandb_run_name', str_none, None),
        ('wandb_run_id', str_none, None),
        ('wandb_save_checkpoints', str2bool, True),
        ('wandb_watch_model', str2bool, False),
        # Training and testing arguments
        ('checkpoint', str_none, None),
        ('val_freq', int, 4000),
        ('interm_ckpt_freq', int, 1000000),
        ('eval_only', str2bool, True),  # Only run EVAL
        ('lr', float, 1e-4),
        ('backbone_lr', float, 1e-4),
        ('lr_scheduler', str, "constant"),
        ('wd', float, 5e-3),
        ('train_iters', int, 600000),
        ('use_compile', str2bool, False),
        ('use_ema', str2bool, False),
        ('lv2_batch_size', int, 1),
        # Model arguments: general policy type
        ('model_type', str, 'denoise3d'),
        ('bimanual', str2bool, False),
        ('keypose_only', str2bool, True),
        ('pre_tokenize', str2bool, True),
        ('custom_img_size', int, None),
        ('workspace_normalizer_buffer', float, 0.04),
        # Model arguments: encoder
        ('backbone', str, "clip"),
        ('finetune_backbone', str2bool, False),
        ('finetune_text_encoder', str2bool, False),
        ('fps_subsampling_factor', int, 5),
        # Model arguments: encoder and head
        ('embedding_dim', int, 120),  # divisible by num_attn_heads
        ('num_attn_heads', int, 8),
        ('num_vis_instr_attn_layers', int, 3),
        ('num_history', int, 1),
        # Model arguments: head
        ('num_shared_attn_layers', int, 4),
        ('relative_action', str2bool, False),
        ('rotation_format', str, 'quat_xyzw'),
        ('denoise_timesteps', int, 10),
        ('denoise_model', str, "rectified_flow"),
        ('learn_extrinsics', str2bool, False),
        ('predict_extrinsics', str2bool, True),
        ('use_front_camera_frame', str2bool, False),
        ('pc_rotate_by_front_camera', str2bool, False),
        ('traj_scene_rope', str2bool, True),
        ('rope_type', str, 'normal'),  # 'adam', 'normal', or 'stopgrad'
        # RoPE stopgrad schedule arguments
        ('rope_schedule_type', str, 'linear'),  # 'linear' or 'cosine'
        ('rope_schedule_start_k', int, 0),  # initial bins to zero out # NOT USED
        ('rope_schedule_end_k', int, 0),  # final bins to zero out
        ('rope_schedule_steps', int, 100000),  # training steps for schedule
        ('sa_blocks_use_rope', str2bool, True),  # must match checkpoint training config
        # RoPE frequency analysis
        ('rope_analysis_max_batches', int, 50),
        ('rope_analysis_save_name', str, 'rope3d_frequency_norms.pt'),
    ]
    for arg in arguments:
        parser.add_argument(f'--{arg[0]}', type=arg[1], default=arg[2])

    return parser.parse_args()


def suppress_output_on_non_main():
    if int(os.environ.get("RANK", 0)) != 0:
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")


if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    # Ensure repo root and scripts on path for rope3d_frequency_analysis_runner
    _script_dir = Path(__file__).resolve().parent
    _repo_root = _script_dir.parent
    for _p in (_repo_root, _script_dir):
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
    from scripts.rope3d_frequency_analysis_runner import run_rope_frequency_analysis
    # Arguments
    args = parse_arguments()
    print("Arguments:")
    print(args)
    print("-" * 100)

    log_dir = args.base_log_dir / args.exp_log_dir / args.run_log_dir
    args.log_dir = log_dir
    log_dir.mkdir(exist_ok=True, parents=True)
    print("Logging:", log_dir)
    print(
        "Available devices (CUDA_VISIBLE_DEVICES):",
        os.environ.get("CUDA_VISIBLE_DEVICES")
    )
    print("Device count:", torch.cuda.device_count())
    args.local_rank = int(os.environ["LOCAL_RANK"])
    suppress_output_on_non_main()

    # DDP initialization
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Select dataset and model classes
    dataset_class = fetch_dataset_class(args.dataset)
    model_class = fetch_model_class(args.model_type)

    # Eval-only + RoPE frequency analysis: use dataset trainer (has prepare_batch) with eval_only
    args.eval_only = True
    TrainTester = fetch_train_tester(args.dataset)
    train_tester = TrainTester(args, dataset_class, model_class)
    train_loader, val_loader, train_sampler = train_tester.get_loaders()
    model = train_tester.get_model()
    train_tester.tokenizer = fetch_tokenizers(args.backbone)
    if not os.path.exists(args.checkpoint):
        normalizer = train_tester.get_workspace_normalizer()
        model.workspace_normalizer.copy_(normalizer)
        dist.barrier(device_ids=[torch.cuda.current_device()])

    if torch.cuda.is_available():
        model = model.cuda()
        torch.set_float32_matmul_precision('high')
    model = DistributedDataParallel(
        model, device_ids=[args.local_rank],
        broadcast_buffers=False, find_unused_parameters=True
    )
    optimizer = train_tester.get_optimizer(model)
    from utils.schedulers import fetch_scheduler
    _ = fetch_scheduler(args.lr_scheduler, optimizer, args.train_iters)
    ema_model = deepcopy(model)
    train_tester.ema = EMA()

    if args.checkpoint:
        train_tester.load_checkpoint(model, ema_model, optimizer)
    print(model.module.workspace_normalizer)

    # RoPE 3D frequency norm analysis (Steps 3–5)
    run_rope_frequency_analysis(
        ema_model if args.use_ema else model,
        val_loader,
        train_tester,
        feature_dim=args.embedding_dim,
        num_attn_heads=args.num_attn_heads,
        log_dir=log_dir,
        max_batches=args.rope_analysis_max_batches,
        save_name=args.rope_analysis_save_name,
    )

    # Test evaluation
    if dist.get_rank() == 0:
        print("Test evaluation.......")
        model.eval()
        train_tester.evaluate_nsteps(
            ema_model if args.use_ema else model,
            val_loader, step_id=-1,
            val_iters=-1
        )
    dist.barrier(device_ids=[torch.cuda.current_device()])

    # Safe program termination
    if torch.distributed.is_initialized():
        torch.cuda.empty_cache()
        torch.distributed.destroy_process_group()
