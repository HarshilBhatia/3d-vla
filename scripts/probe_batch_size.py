"""Standalone per-GPU batch-size / throughput probe for 3DFA on B200.

Builds the same model + real data pipeline the trainer builds (via BaseTrainTester
subclass hooks), then times N training steps and reports peak CUDA memory and
samples/sec. Exists because config `benchmark=true` records nothing: the trainer's
`train_one_step` always returns None, and the benchmark path is gated on that
return value being a timing dict.

Run under torchrun --nproc_per_node 1.
"""
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from datasets import fetch_dataset_class
from modeling.policy import fetch_model_class
from utils.trainers import fetch_train_tester
from utils.hydra_utils import get_config, get_config_path


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = get_config(overrides=sys.argv[1:], config_name="config",
                      config_path=get_config_path())
    if not args.base_log_dir.is_absolute():
        args.base_log_dir = Path(__file__).resolve().parent / args.base_log_dir
    args.log_dir = args.base_log_dir / args.exp_log_dir / args.run_log_dir
    args.log_dir.mkdir(exist_ok=True, parents=True)
    args.local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend="nccl", init_method="env://",
                            timeout=timedelta(seconds=300))
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    tt = fetch_train_tester(args.dataset)(
        args, fetch_dataset_class(args.dataset), fetch_model_class(args.model_type)
    )
    train_loader, _, sampler = tt.get_loaders()
    model = tt.get_model().cuda()
    from modeling.encoder.text import fetch_tokenizers
    tt.tokenizer = fetch_tokenizers(getattr(args, "text_backbone", None) or args.backbone)

    if args.use_compile:
        model.compute_loss = torch.compile(model.compute_loss, fullgraph=True)
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[args.local_rank], static_graph=True,
        find_unused_parameters=False, bucket_cap_mb=10,
        gradient_as_bucket_view=True,
    )
    optimizer = tt.get_optimizer(model)
    from utils.schedulers import fetch_scheduler
    sched = fetch_scheduler(args.lr_scheduler, optimizer, args.train_iters)

    model.train()
    sampler.set_epoch(0)
    it = iter(train_loader)
    warmup = int(os.environ.get("PROBE_WARMUP", "6"))
    measure = int(os.environ.get("PROBE_MEASURE", "12"))

    n_samples = 0
    t0 = None
    for step in range(warmup + measure):
        sample = next(it)
        if step == warmup:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
        tt.train_one_step(model, optimizer, sched, sample, step_id=step)
        if step >= warmup:
            n_samples += sample["action"].shape[0]
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    peak = torch.cuda.max_memory_allocated() / 1e9
    reserved = torch.cuda.max_memory_reserved() / 1e9
    if dist.get_rank() == 0:
        per_gpu = sample["action"].shape[0]
        print(
            f"PROBE_RESULT bs_global={args.batch_size} bs_per_gpu={per_gpu} "
            f"lv2={args.lv2_batch_size} world={dist.get_world_size()} "
            f"steps={measure} step_ms={dt / measure * 1000:.1f} "
            f"samples_per_s={n_samples * dist.get_world_size() / dt:.2f} "
            f"peak_alloc_gb={peak:.2f} peak_reserved_gb={reserved:.2f}",
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
