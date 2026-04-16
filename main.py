"""Main script for training and testing."""

import os
from datetime import timedelta
from pathlib import Path
import sys

import torch

from datasets import fetch_dataset_class
from modeling.policy import fetch_model_class
from utils.trainers import fetch_train_tester
from utils.hydra_utils import get_config, get_config_path


def redirect_non_main_output(log_dir: Path):
    """Send non-rank-0 output to per-rank log files instead of /dev/null so errors are visible."""
    rank = int(os.environ.get("RANK", 0))
    if rank != 0:
        log_dir.mkdir(exist_ok=True, parents=True)
        f = open(log_dir / f"rank_{rank}.log", "w", buffering=1)
        sys.stdout = f
        sys.stderr = f


if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    # Compose config from config/config.yaml + CLI overrides (e.g. batch_size=32 eval_only=true)
    args = get_config(
        overrides=sys.argv[1:],
        config_name="config",
        config_path=get_config_path(),
    )
    # Resolve relative base_log_dir relative to this script's directory
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

    # Redirect non-rank-0 output to per-rank log files (not /dev/null) so errors are visible
    redirect_non_main_output(log_dir / "rank_logs")

    # Short NCCL timeout: detect hung ranks in 120s instead of the default 600s.
    # When a rank dies the job fails fast, torchrun --max-restarts restarts from checkpoint.
    os.environ.setdefault("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", "120")

    # DDP initialization
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(
        backend='nccl', init_method='env://',
        timeout=timedelta(seconds=120),
    )
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Select dataset and model classes
    dataset_class = fetch_dataset_class(args.dataset)
    model_class = fetch_model_class(args.model_type)

    # Run
    TrainTester = fetch_train_tester(args.dataset)
    train_tester = TrainTester(args, dataset_class, model_class)
    train_tester.main()

    # Safe program termination
    if torch.distributed.is_initialized():
        torch.cuda.empty_cache()
        torch.distributed.destroy_process_group()
