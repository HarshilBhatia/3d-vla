"""A minimal trainer that exercises the resilience layer.

Deliberately model-agnostic and tiny: its only job is to be a realistic
consumer of ``Checkpointer`` + ``ResumableDistributedSampler`` so that
``test_kill_resume.py`` has something whose losses it can compare. It consumes
randomness (dropout) and cares about sample order, which is what makes the
comparison meaningful.

Runs on CPU/gloo by default so the correctness test needs no GPU; pass
--device cuda to exercise the same paths under NCCL.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from resilience import (  # noqa: E402
    Checkpointer,
    PreemptionGuard,
    ResumableDistributedSampler,
    StatefulLoader,
    seed_everything,
)


class ToyDataset(torch.utils.data.Dataset):
    """Fixed synthetic data. Index is recoverable from the sample so the test
    can assert on data *order*, not just on loss values."""

    def __init__(self, n: int = 512, dim: int = 16) -> None:
        g = torch.Generator().manual_seed(1234)
        self.x = torch.randn(n, dim, generator=g)
        self.y = torch.randn(n, 1, generator=g)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i: int):
        return self.x[i], self.y[i], i


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--kill-at", type=int, default=-1, help="hard-exit before this step")
    # The launcher's contract: these two come from the environment so that
    # train_cmd can be any shell string (a conda wrapper, torchrun, ...) without
    # the launcher having to append argv to it.
    ap.add_argument("--ckpt-dir", default=os.environ.get("RESILIENCE_CKPT_DIR"),
                    help="default: $RESILIENCE_CKPT_DIR")
    ap.add_argument("--log", required=True, help="jsonl of per-step losses")
    ap.add_argument("--batch-size", type=int, default=8, help="per rank")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--ckpt-interval", type=float,
                    default=float(os.environ.get("RESILIENCE_CKPT_INTERVAL", 0.0)),
                    help="seconds between checkpoints; default: $RESILIENCE_CKPT_INTERVAL, "
                         "0 means every step (test mode)")
    ap.add_argument("--step-log", help="jsonl of per-step timings for bench/scaling.py")
    args = ap.parse_args()
    if not args.ckpt_dir:
        ap.error("--ckpt-dir is required (or set RESILIENCE_CKPT_DIR)")

    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    if world > 1:
        backend = "nccl" if args.device == "cuda" else "gloo"
        dist.init_process_group(backend, rank=rank, world_size=world)
    if args.device == "cuda":
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    device = torch.device(args.device)

    seed_everything(args.seed, rank)

    ds = ToyDataset()
    sampler = ResumableDistributedSampler(len(ds), num_replicas=world, rank=rank,
                                          shuffle=True, seed=args.seed, drop_last=True)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, sampler=sampler,
        num_workers=0,  # workers would need their own RNG plumbing; not the point here
        drop_last=True,
    )
    stream = StatefulLoader(loader, sampler, global_batch_size=args.batch_size * world, seed=args.seed)

    # Dropout is load-bearing: it makes the loss depend on RNG state, so a
    # resume that restores data order but not RNG still fails the test.
    model = nn.Sequential(nn.Linear(16, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1)).to(device)
    if world > 1:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[device.index] if args.device == "cuda" else None)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    ckpt = Checkpointer(
        args.ckpt_dir,
        interval_seconds=args.ckpt_interval,
        keep_last=2,
        rank=rank,
        world_size=world,
        require_shared_fs=False,  # the unit test runs in a temp dir
    )

    def build_state() -> dict:
        base = model.module if hasattr(model, "module") else model
        return {"model": base.state_dict(), "optimizer": opt.state_dict(),
                "data": stream.state_dict()}

    def apply_state(p: dict) -> None:
        base = model.module if hasattr(model, "module") else model
        base.load_state_dict(p["model"])
        opt.load_state_dict(p["optimizer"])
        stream.load_state_dict(p["data"])

    start = ckpt.resume(apply_state)
    guard = PreemptionGuard()

    logf = open(args.log, "a" if start else "w", buffering=1) if rank == 0 else None
    stepf = None
    if args.step_log and rank == 0:
        Path(args.step_log).parent.mkdir(parents=True, exist_ok=True)
        stepf = open(args.step_log, "a" if start else "w", buffering=1)
    global_batch = args.batch_size * world
    it = iter(stream)
    step = start
    while step < args.steps:
        if args.kill_at >= 0 and step == args.kill_at:
            # Save first, then die the way a preemption does: no cleanup, no
            # process-group teardown. Resuming from this is the whole point.
            ckpt.maybe_save(step, build_state, force=True)
            if rank == 0:
                print(f"[toy] hard exit at step {step}", flush=True)
            logf and logf.close()
            os._exit(17)

        t_step = time.perf_counter()
        x, y, idx = next(it)
        x, y = x.to(device), y.to(device)
        loss = nn.functional.mse_loss(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if args.device == "cuda":
            torch.cuda.synchronize()  # else step_time measures queue depth, not work
        dt = time.perf_counter() - t_step
        step += 1

        if logf is not None:
            logf.write(json.dumps({"step": step, "loss": round(float(loss.item()), 10),
                                   "idx": idx.tolist()}) + "\n")
        if stepf is not None:
            stepf.write(json.dumps({"step": step, "step_time": dt,
                                    "global_batch": global_batch}) + "\n")

        if guard.check():
            ckpt.maybe_save(step, build_state, force=True)
            if rank == 0:
                # Confirm only after the write returned, so the batch script
                # requeues on evidence rather than on a timeout.
                guard.confirm()
                print("[toy] checkpointed after preemption signal; exiting for requeue", flush=True)
            break
        ckpt.maybe_save(step, build_state)

    ckpt.maybe_save(step, build_state, force=True)
    if logf is not None:
        logf.close()
    if stepf is not None:
        stepf.close()
    if world > 1:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
