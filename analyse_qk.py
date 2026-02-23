"""Script for eval-only run and RoPE 3D frequency norm analysis."""

import os
import sys
from copy import deepcopy
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from datasets import fetch_dataset_class
from modeling.encoder.text import fetch_tokenizers
from modeling.policy import fetch_model_class
from utils.ema import EMA
from utils.hydra_utils import get_config, get_config_path
from utils.trainers import fetch_train_tester


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

    # Compose config from config/config.yaml + CLI overrides (eval_only=true by default for this script)
    args = get_config(
        overrides=sys.argv[1:] + ["eval_only=true"],
        config_name="config",
        config_path=get_config_path(),
    )
    if not args.base_log_dir.is_absolute():
        args.base_log_dir = Path(__file__).resolve().parent / args.base_log_dir

    print("Arguments:")
    for k, v in sorted(vars(args).items()):
        print(f"  {k}: {v}")
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
