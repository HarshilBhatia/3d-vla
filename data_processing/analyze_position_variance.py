"""
Step 6: Analyze cross-sample EEF and token position variance.

Hypothesis: EEF / token positions have near-zero variance across the training set
→ RoPE encodes the same thing for every sample → no gradient signal.

Pass: EEF position std > 0.2m, token mean position std > 0.1m across samples
Fail: EEF std < 0.05m → robot barely moves, positional signal is useless

Usage:
    micromamba run -n gr00t python data_processing/analyze_position_variance.py \
        --depth-cache-dir /work/nvme/bgkz/droid_multilab_depth_cam2cam_ext2 \
        --n-shards 20
"""
import argparse
import random
from pathlib import Path

import torch
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--depth-cache-dir", required=True)
    p.add_argument("--n-shards", type=int, default=20)
    p.add_argument("--n-rows", type=int, default=50, help="Rows per shard")
    args = p.parse_args()

    shard_files = sorted(Path(args.depth_cache_dir).glob("depth_shard_?????.pt"))
    if not shard_files:
        raise ValueError(f"No depth shards in {args.depth_cache_dir}")
    print(f"Found {len(shard_files)} shards, sampling {args.n_shards}")

    sampled = random.sample(shard_files, min(args.n_shards, len(shard_files)))

    all_eef = []          # [N, 3]
    all_token_mean = []   # [N, 3] — mean position of image tokens per sample
    all_token_std = []    # [N, 3] — std of image tokens within sample

    for sf in sampled:
        data = torch.load(sf, weights_only=True, map_location="cpu")
        pos = data["token_positions_3d"]  # [N, seq_len, 3]
        eef = data["eef_position_3d"]     # [N, 3]
        N = pos.shape[0]
        rows = random.sample(range(N), min(args.n_rows, N))

        for r in rows:
            p_row = pos[r]
            e_row = eef[r]

            nonzero = p_row.abs().sum(dim=1) > 0
            img_pos = p_row[nonzero]
            if img_pos.shape[0] < 10:
                continue

            all_eef.append(e_row.numpy())
            all_token_mean.append(img_pos.mean(dim=0).numpy())
            all_token_std.append(img_pos.std(dim=0).numpy())

    all_eef = np.array(all_eef)           # [N, 3]
    all_token_mean = np.array(all_token_mean)  # [N, 3]
    all_token_std = np.array(all_token_std)    # [N, 3]

    print(f"\nAnalyzed {len(all_eef)} samples")

    print(f"\n{'='*60}")
    print("EEF POSITION VARIANCE ACROSS SAMPLES (does the robot move?)")
    print(f"{'='*60}")
    eef_std = all_eef.std(axis=0)
    for i, axis in enumerate(['X', 'Y', 'Z']):
        print(f"  {axis}: mean={all_eef[:,i].mean():.3f}m  std={eef_std[i]:.3f}m  "
              f"range=[{all_eef[:,i].min():.2f}, {all_eef[:,i].max():.2f}]m")

    print(f"\n{'='*60}")
    print("TOKEN MEAN POSITION VARIANCE ACROSS SAMPLES (does the scene move?)")
    print(f"{'='*60}")
    for i, axis in enumerate(['X', 'Y', 'Z']):
        vals = all_token_mean[:, i]
        print(f"  {axis}: mean={vals.mean():.3f}m  std={vals.std():.3f}m  "
              f"range=[{vals.min():.2f}, {vals.max():.2f}]m")

    # Correlation between EEF and token mean positions
    print(f"\n{'='*60}")
    print("CORRELATION: EEF vs TOKEN MEAN POSITION (does scene shift with EEF?)")
    print(f"{'='*60}")
    for i, axis in enumerate(['X', 'Y', 'Z']):
        corr = np.corrcoef(all_eef[:, i], all_token_mean[:, i])[0, 1]
        print(f"  {axis}: r = {corr:.3f}")

    print(f"\n{'='*60}")
    print("VERDICT")
    print(f"{'='*60}")
    eef_total_std = eef_std.mean()
    token_var = all_token_mean.std(axis=0).mean()

    if eef_total_std > 0.2 and token_var > 0.1:
        print(f"  PASS: EEF std={eef_total_std:.3f}m, token mean std={token_var:.3f}m → good positional variance")
    elif eef_total_std < 0.05:
        print(f"  FAIL: EEF std={eef_total_std:.3f}m < 0.05m → robot barely moves, no positional signal")
    else:
        print(f"  MARGINAL: EEF std={eef_total_std:.3f}m, token mean std={token_var:.3f}m")


if __name__ == "__main__":
    main()
