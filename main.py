"""Main script for training and testing."""

import os
from datetime import timedelta
from pathlib import Path
import sys

import torch
from torch.distributed.elastic.multiprocessing.errors import record

from datasets import fetch_dataset_class
from modeling.policy import fetch_model_class
from training import fetch_train_tester
from utils.hydra_utils import get_config, get_config_path, write_experiment_manifest


def redirect_non_main_output(log_dir: Path):
    """Send non-rank-0 output to per-rank log files instead of /dev/null so errors are visible."""
    rank = int(os.environ.get("RANK", 0))
    if rank != 0:
        log_dir.mkdir(exist_ok=True, parents=True)
        f = open(log_dir / f"rank_{rank}.log", "w", buffering=1)
        sys.stdout = f
        sys.stderr = f


@record
def main():
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

    log_dir = args.base_log_dir / args.exp_log_dir / args.run_log_dir
    args.log_dir = log_dir
    log_dir.mkdir(exist_ok=True, parents=True)
    args.local_rank = int(os.environ["LOCAL_RANK"])

    # A concise, stable description for paper/result tooling. The full config is
    # still printed below; this manifest uses the canonical public vocabulary.
    if int(os.environ.get("RANK", 0)) == 0:
        write_experiment_manifest(args, log_dir / "experiment_manifest.json")

    # Redirect non-rank-0 output to per-rank log files (not /dev/null) so errors are visible
    redirect_non_main_output(log_dir / "rank_logs")

    rank = int(os.environ.get("RANK", 0))
    if rank == 0:
        print("Arguments:")
        for k, v in sorted(vars(args).items()):
            print(f"  {k}: {v}")
        print("-" * 100)
        print("Logging:", log_dir)
        print(
            "Available devices (CUDA_VISIBLE_DEVICES):",
            os.environ.get("CUDA_VISIBLE_DEVICES")
        )
        print("Device count:", torch.cuda.device_count())

    # NCCL timeout. 120s fails fast on a hung rank, but the launcher passes no
    # torchrun --max-restarts, so nothing restarts and a merely contended node
    # (step time 0.5s -> 14s) kills the run. Override with NCCL_TIMEOUT_SEC.
    _nccl_timeout = int(os.environ.get("NCCL_TIMEOUT_SEC", "1800"))
    os.environ.setdefault("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", str(_nccl_timeout))

    # DDP initialization
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(
        backend='nccl', init_method='env://',
        timeout=timedelta(seconds=_nccl_timeout),
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


if __name__ == '__main__':
    main()
