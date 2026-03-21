"""
Step 1: Analyze rotation angle magnitude and intra-sample position spread.

Hypothesis: token positions within a sample have too little spread → R(p_k1) ≈ R(p_k2)
→ RoPE cannot discriminate between tokens.

Pass: mean intra-sample XYZ std > 0.3m in ≥2 axes
Fail: std < 0.1m → positions too clustered for RoPE to matter

Usage:
    micromamba run -n gr00t python data_processing/analyze_rope_angles.py \
        --depth-cache-dir /work/nvme/bgkz/droid_multilab_depth_cam2cam_ext2 \
        --n-shards 10
"""
import argparse
import random
from pathlib import Path

import torch
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--depth-cache-dir", required=True)
    p.add_argument("--n-shards", type=int, default=10, help="Number of shards to sample")
    p.add_argument("--n-rows", type=int, default=100, help="Rows per shard to sample")
    p.add_argument("--rope-base-freq", type=float, default=100.0)
    args = p.parse_args()

    shard_files = sorted(Path(args.depth_cache_dir).glob("depth_shard_?????.pt"))
    if not shard_files:
        raise ValueError(f"No depth shards found in {args.depth_cache_dir}")
    print(f"Found {len(shard_files)} depth shards")

    sampled = random.sample(shard_files, min(args.n_shards, len(shard_files)))

    all_intra_std = []   # per-sample std across image tokens [N, 3]
    all_max_angles = []  # per-sample max angle per axis [N, 3]
    all_mean_pos = []    # per-sample mean position [N, 3]
    n_zero_samples = 0

    for sf in sampled:
        data = torch.load(sf, weights_only=True, map_location="cpu")
        pos = data["token_positions_3d"]  # [N, seq_len, 3]
        N = pos.shape[0]
        rows = random.sample(range(N), min(args.n_rows, N))

        for r in rows:
            p_row = pos[r]  # [seq_len, 3]
            # Only consider non-zero tokens (image tokens)
            nonzero_mask = p_row.abs().sum(dim=1) > 0
            img_pos = p_row[nonzero_mask]  # [n_img, 3]

            if img_pos.shape[0] < 10:
                n_zero_samples += 1
                continue

            std_xyz = img_pos.std(dim=0)  # [3] — spread across image tokens
            mean_xyz = img_pos.mean(dim=0)

            # Max rotation angle = max(|position|) * max_freq
            # max_freq = 1 / rope_base_freq^0 = 1.0 rad/m (i=0 in frequency table)
            max_freq = 1.0  # always 1.0 regardless of base (freq[0] = base^0 = 1)
            max_angle_rad = img_pos.abs().max(dim=0).values * max_freq  # [3]

            all_intra_std.append(std_xyz.numpy())
            all_max_angles.append(max_angle_rad.numpy())
            all_mean_pos.append(mean_xyz.numpy())

    all_intra_std = np.array(all_intra_std)   # [N, 3]
    all_max_angles = np.array(all_max_angles)  # [N, 3]
    all_mean_pos = np.array(all_mean_pos)      # [N, 3]

    print(f"\nAnalyzed {len(all_intra_std)} samples ({n_zero_samples} skipped with <10 tokens)")
    print(f"\n{'='*60}")
    print("INTRA-SAMPLE POSITION STD (spread of image tokens within one sample)")
    print(f"{'='*60}")
    for i, axis in enumerate(['X', 'Y', 'Z']):
        vals = all_intra_std[:, i]
        print(f"  {axis}: mean={vals.mean():.3f}m  median={np.median(vals):.3f}m  "
              f"p10={np.percentile(vals,10):.3f}m  p90={np.percentile(vals,90):.3f}m")

    print(f"\n{'='*60}")
    print("MAX ROTATION ANGLE (|position| * max_freq=1.0 rad/m) [degrees]")
    print(f"{'='*60}")
    for i, axis in enumerate(['X', 'Y', 'Z']):
        vals = np.degrees(all_max_angles[:, i])
        print(f"  {axis}: mean={vals.mean():.1f}°  median={np.median(vals):.1f}°  "
              f"p10={np.percentile(vals,10):.1f}°  p90={np.percentile(vals,90):.1f}°")

    print(f"\n{'='*60}")
    print("MEAN TOKEN POSITION (scene center in robot base frame)")
    print(f"{'='*60}")
    for i, axis in enumerate(['X', 'Y', 'Z']):
        vals = all_mean_pos[:, i]
        print(f"  {axis}: mean={vals.mean():.3f}m  std={vals.std():.3f}m")

    print(f"\n{'='*60}")
    print("VERDICT")
    print(f"{'='*60}")
    axes_pass = sum(all_intra_std[:, i].mean() > 0.3 for i in range(3))
    if axes_pass >= 2:
        print(f"  PASS: intra-sample spread > 0.3m in {axes_pass}/3 axes → RoPE can discriminate tokens")
    elif all_intra_std.mean() > 0.1:
        print(f"  MARGINAL: some spread but < 0.3m threshold — RoPE signal may be weak")
    else:
        print(f"  FAIL: intra-sample spread < 0.1m — tokens nearly co-located, RoPE sees near-identity rotations")


if __name__ == "__main__":
    main()
