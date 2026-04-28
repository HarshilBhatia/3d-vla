"""
Mock skeleton for PerAct2 depth-aware training.

The script shows how depth data flows through the existing pipeline:
    zarr dataset  →  RLBenchDataset.__getitem__
                  →  depth (B, n_cam, H, W) float16
                  →  RLBenchDataPreprocessor.process_obs()
                  →  point-cloud tensors (B, n_cam, 3, H, W)
                  →  [TODO] your model

Run this first to confirm the depth pipeline works end-to-end before
writing the full training script.

Prerequisites
-------------
    conda activate 3dfa
    cd /home/lzaceria/mscv/3dvla/3d-vla

    # Zarr datasets (must exist):
    ls Peract2_zarr/<task>/train.zarr
    ls Peract2_zarr/<task>/val.zarr

Usage
-----
    # Single task, 1 batch
    python scripts/rlbench/mock_train_peract2_depth.py \\
        --train-data-dir Peract2_zarr/bimanual_lift_tray/train.zarr \\
        --task bimanual_lift_tray \\
        --num-batches 2

    # All tasks
    python scripts/rlbench/mock_train_peract2_depth.py \\
        --train-data-dir Peract2_zarr/all/train.zarr \\
        --task all \\
        --num-batches 2

    # Submit via sbatch:
    sbatch scripts/rlbench/mock_train_peract2_depth.sbatch
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Make sure we can import project modules from anywhere inside 3d-vla
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[2]   # 3d-vla/
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-data-dir", default="Peract2_zarr/bimanual_lift_tray/train.zarr",
                   help="Path to train.zarr (relative to 3d-vla/ or absolute).")
    p.add_argument("--instructions",   default="instructions/peract2/instructions.json",
                   help="Path to task instructions JSON.")
    p.add_argument("--task",           default="bimanual_lift_tray",
                   help="Task name to filter on, or 'all' for every task.")
    p.add_argument("--dataset",        default="Peract2_3dfront_3dwrist",
                   help="Dataset name used to select depth2cloud and preprocessor.")
    p.add_argument("--batch-size",     type=int, default=2)
    p.add_argument("--chunk-size",     type=int, default=1,
                   help="Temporal chunk size per sample.")
    p.add_argument("--num-workers",    type=int, default=0)
    p.add_argument("--num-batches",    type=int, default=2,
                   help="How many batches to iterate (for a quick smoke test).")
    p.add_argument("--mem-limit",      type=int, default=8,
                   help="CPU RAM limit per worker (GB) for zarr cache.")
    return p.parse_args()


def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # 1. Dataset
    # ------------------------------------------------------------------
    from datasets.rlbench import RLBenchDataset

    filter_tasks = None if args.task == "all" else [args.task]

    print(f"[DATA] Loading dataset from: {args.train_data_dir}")
    dataset = RLBenchDataset(
        root=args.train_data_dir,
        instructions=args.instructions,
        relative_action=False,
        mem_limit=args.mem_limit,
        chunk_size=args.chunk_size,
        filter_tasks=filter_tasks,
    )
    print(f"[DATA] Samples: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )

    # ------------------------------------------------------------------
    # 2. Depth → point-cloud preprocessor
    # ------------------------------------------------------------------
    from utils.depth2cloud import fetch_depth2cloud
    from utils.data_preprocessors import fetch_data_preprocessor

    depth2cloud = fetch_depth2cloud(args.dataset)
    preprocessor_cls = fetch_data_preprocessor(args.dataset)
    preprocessor = preprocessor_cls(
        keypose_only=True,
        num_history=1,
        depth2cloud=depth2cloud,
    )

    # ------------------------------------------------------------------
    # 3. Iterate batches — verify depth and point-cloud shapes
    # ------------------------------------------------------------------
    print(f"\n[MOCK] Iterating {args.num_batches} batch(es) — depth shape check")
    print("-" * 60)

    for step, batch in enumerate(loader):
        if step >= args.num_batches:
            break

        # Raw tensors from the zarr dataset
        rgb        = batch["rgb"]          # (B, n_cam, 3, H, W)  uint8
        depth      = batch["depth"]        # (B, n_cam, H, W)     float16
        extrinsics = batch["extrinsics"]   # (B, n_cam, 4, 4)     float
        intrinsics = batch["intrinsics"]   # (B, n_cam, 3, 3)     float
        action     = batch["action"]       # (B, T, 8)            float
        instr      = batch["instr"]        # list[str]

        print(f"  step {step}")
        print(f"    rgb        : {rgb.shape}  dtype={rgb.dtype}")
        print(f"    depth      : {depth.shape}  dtype={depth.dtype}  "
              f"range=[{depth.min():.3f}, {depth.max():.3f}]")
        print(f"    extrinsics : {extrinsics.shape}")
        print(f"    intrinsics : {intrinsics.shape}")
        print(f"    action     : {action.shape}")
        print(f"    instr[0]   : {instr[0]}")

        # depth → point cloud via RLBenchDataPreprocessor
        rgb_proc, pcd = preprocessor.process_obs(
            rgbs=rgb,
            rgb2d=None,
            depth=depth,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            augment=False,
        )
        print(f"    rgb_proc   : {rgb_proc.shape}  (preprocessed, float, 0–1)")
        print(f"    pcd        : {pcd.shape}       (point cloud, world coords)")
        print()

    print("-" * 60)
    print("[MOCK] Depth pipeline verified.  Both rgb_proc and pcd are on GPU.")

    # ======================================================================
    # TODO: plug in your model and training loop here.
    #
    # You have:
    #   rgb_proc  : (B, n_cam, 3, H, W)  float32   preprocessed RGB  [0,1]
    #   pcd       : (B, n_cam, 3, H, W)  float32   point cloud (world coords)
    #   action    : (B, T, 8)            float32   ground-truth action
    #   instr     : list[str]                       language instructions
    #
    # Example training loop structure:
    #
    #   model  = YourModel(...).cuda()
    #   optim  = torch.optim.AdamW(model.parameters(), lr=1e-4)
    #
    #   for step, batch in enumerate(loader):
    #       rgb_proc, pcd = preprocessor.process_obs(
    #           batch["rgb"], None, batch["depth"],
    #           batch["extrinsics"], batch["intrinsics"]
    #       )
    #       pred   = model(rgb_proc, pcd, batch["instr"])
    #       loss   = criterion(pred, batch["action"].cuda())
    #       loss.backward()
    #       optim.step()
    #       optim.zero_grad()
    # ======================================================================

    print("\n[TODO] Replace this stub with your model + optimizer + training loop.")


if __name__ == "__main__":
    main()
