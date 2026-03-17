#!/usr/bin/env python3
"""
Evaluate both RAIL checkpoints on a few DROID trajectories.

Usage:
  python experiments/eval_rail_checkpoints.py
  python experiments/eval_rail_checkpoints.py --traj_ids 0 1 2 3 4 --steps 100
"""

import dataclasses
import logging
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import tyro

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

CHECKPOINTS = [
    "/work/hdd/bgkz/hbhatia1/outputs_rail/rail-cached-v1/checkpoint-72000",
    "/work/hdd/bgkz/hbhatia1/outputs_rail/rail-cached-v1/checkpoint-73000",
]
DATASET_PATH = "/work/nvme/bgkz/droid_raw_large_superset"
EMBODIMENT_TAG = EmbodimentTag.OXE_DROID
ACTION_HORIZON = 16


@dataclasses.dataclass
class Args:
    traj_ids: list[int] = dataclasses.field(default_factory=lambda: [0, 1, 2])
    """Trajectory IDs to evaluate."""

    steps: int = 100
    """Max steps per trajectory."""

    denoising_steps: int = 4
    """Number of DiT denoising steps."""


def run_inference(policy: Gr00tPolicy, loader: LeRobotEpisodeLoader, traj_id: int, steps: int):
    """Run inference on one trajectory, return (pred_actions, gt_actions) arrays."""
    traj = loader[traj_id]
    actual_steps = min(steps, len(traj))

    modality_configs = policy.get_modality_config()
    obs_configs = deepcopy(modality_configs)
    obs_configs.pop("action", None)
    action_keys = modality_configs["action"].modality_keys

    pred_actions, gt_actions = [], []

    for step in range(0, actual_steps, ACTION_HORIZON):
        data_point = extract_step_data(traj, step, obs_configs, EMBODIMENT_TAG)

        # Build observation in {modality: {key: array}} format with batch dim
        obs = {"video": {}, "state": {}, "language": {}}
        for k, v in data_point.states.items():
            obs["state"][k] = v[None]               # (1, T, D)
        for k, v in data_point.images.items():
            obs["video"][k] = np.array(v)[None]     # (1, T, H, W, C)
        for lang_key in modality_configs["language"].modality_keys:
            obs["language"][lang_key] = [[data_point.text]]

        action_chunk, _ = policy.get_action(obs)  # {key: (1, horizon, dim)}

        for j in range(ACTION_HORIZON):
            t = step + j
            if t >= actual_steps:
                break
            pred = np.concatenate(
                [np.atleast_1d(action_chunk[k][0][j]) for k in action_keys], axis=0
            )
            gt = np.concatenate(
                [np.atleast_1d(np.array(traj[f"action.{k}"].iloc[t])) for k in action_keys], axis=0
            )
            pred_actions.append(pred)
            gt_actions.append(gt)

    return np.stack(pred_actions), np.stack(gt_actions)


def evaluate_checkpoint(ckpt_path: str, args: Args):
    log.info(f"\n{'='*60}")
    log.info(f"Checkpoint: {Path(ckpt_path).name}")
    log.info(f"{'='*60}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = Gr00tPolicy(
        embodiment_tag=EMBODIMENT_TAG,
        model_path=ckpt_path,
        device=device,
    )

    loader = LeRobotEpisodeLoader(
        dataset_path=DATASET_PATH,
        modality_configs=policy.get_modality_config(),
        video_backend="torchcodec",
    )
    log.info(f"Dataset: {len(loader)} episodes")

    all_mse, all_mae = [], []

    for traj_id in args.traj_ids:
        if traj_id >= len(loader):
            log.warning(f"traj_id {traj_id} out of range, skipping")
            continue

        log.info(f"  Trajectory {traj_id} ...")
        pred, gt = run_inference(policy, loader, traj_id, args.steps)

        mse = float(np.mean((pred - gt) ** 2))
        mae = float(np.mean(np.abs(pred - gt)))
        all_mse.append(mse)
        all_mae.append(mae)
        log.info(f"    MSE={mse:.5f}  MAE={mae:.5f}  shape={pred.shape}")

    avg_mse = float(np.mean(all_mse))
    avg_mae = float(np.mean(all_mae))
    log.info(f"\n  >> {Path(ckpt_path).name}  avg MSE={avg_mse:.5f}  avg MAE={avg_mae:.5f}")
    return avg_mse, avg_mae


def main(args: Args):
    results = {}
    for ckpt in CHECKPOINTS:
        avg_mse, avg_mae = evaluate_checkpoint(ckpt, args)
        results[Path(ckpt).name] = {"mse": avg_mse, "mae": avg_mae}

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, metrics in results.items():
        print(f"  {name}:  MSE={metrics['mse']:.5f}  MAE={metrics['mae']:.5f}")


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
