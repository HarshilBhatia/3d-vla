"""Visualize sampled token correspondences between two orbital camera viewpoints.

Uses raw pretrained CLIP/SigLIP2/DINOv2 backbone features (no checkpoint needed)
to compute subsampled tokens, then matches them across viewpoints by cosine
similarity and shows the result in Rerun.

Sampling modes:
  density       — density-based sampling in feature space (original behaviour)
  fps3d         — farthest-point sampling in 3D position space (XYZ)
  uniform_image — uniform stride-based grid sampling in 2D image/patch space

Usage:
    python scripts/helpers/visualize_fps_correspondences.py \\
        --idx1 0 --idx2 120 --top_k 30 --backbone clip --sampler fps3d
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import einops
import numpy as np
import rerun as rr
import torch
import torch.nn.functional as F

from datasets import fetch_dataset_class
from modeling.encoder.multimodal.base_encoder import density_based_sampler
from modeling.encoder.multimodal.encoder3d import Encoder
from utils.depth2cloud import fetch_depth2cloud


def uniform_image_sampler(ncam, fh, fw, fps_factor):
    """Uniform stride-based grid sampling in 2D image/patch space.

    Selects every `stride` rows and columns from each camera's (fh, fw) feature
    grid, giving ~fps_factor downsampling.  Returns flat indices into the
    (ncam * fh * fw) token sequence.

    Args:
        ncam: number of cameras
        fh, fw: feature map height and width
        fps_factor: target downsampling factor (stride = round(sqrt(fps_factor)))

    Returns:
        inds: (M,) long tensor — flat indices into (ncam*fh*fw)
    """
    import math
    stride = max(1, round(math.sqrt(fps_factor)))
    row_inds = torch.arange(0, fh, stride)
    col_inds = torch.arange(0, fw, stride)
    grid_h, grid_w = torch.meshgrid(row_inds, col_inds, indexing='ij')
    local_inds = (grid_h * fw + grid_w).flatten()           # (fh//stride * fw//stride,)
    cam_offsets = torch.arange(ncam) * (fh * fw)            # (ncam,)
    all_inds = (cam_offsets.unsqueeze(1) + local_inds.unsqueeze(0)).flatten()  # (ncam*M_per_cam,)
    return all_inds


@torch.no_grad()
def farthest_point_sampler_3d(pos, n_samples):
    """Farthest-point sampling on 3D XYZ positions.

    Args:
        pos: (B, N, 3) world-frame positions
        n_samples: number of points to select

    Returns:
        inds: (B, n_samples) long tensor of selected indices
    """
    B, N, _ = pos.shape
    device = pos.device
    inds = torch.zeros(B, n_samples, dtype=torch.long, device=device)
    # Start from a random point per batch element
    inds[:, 0] = torch.randint(0, N, (B,), device=device)
    # dist to nearest already-selected point — initialise as infinity
    min_dists = torch.full((B, N), float('inf'), device=device)

    for i in range(1, n_samples):
        prev = pos[torch.arange(B, device=device), inds[:, i - 1]]  # (B, 3)
        d = ((pos - prev.unsqueeze(1)) ** 2).sum(-1)                # (B, N)
        min_dists = torch.minimum(min_dists, d)
        inds[:, i] = min_dists.argmax(dim=1)

    return inds


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_sample(dataset, depth2cloud, idx):
    """Load a single episode frame and convert depth to world-frame pcd."""
    sample = dataset[idx]
    rgb = sample["rgb"].float() / 255.0        # (1, ncam, 3, H, W) float32 [0,1]
    depth = sample["depth"].float()             # (1, ncam, H, W)
    extrinsics = sample["extrinsics"].float()   # (1, ncam, 4, 4)
    intrinsics = sample["intrinsics"].float()   # (1, ncam, 3, 3)
    with torch.no_grad():
        pcd = depth2cloud(
            depth.cuda(), extrinsics.cuda(), intrinsics.cuda()
        ).cpu()  # (1, ncam, 3, H, W)
    return rgb, pcd, sample


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_fps_tokens(encoder, rgb, pcd, fps_factor, backbone_name, sampler='fps3d'):
    """Extract sampled token positions and raw backbone features.

    Bypasses the trained FPN / vis_lang_attention — uses only the frozen
    pretrained backbone, so no checkpoint is required.

    Args:
        sampler: 'density' (feature-space density), 'fps3d' (3D position FPS),
                 or 'uniform_image' (regular stride grid in 2D patch space)

    Returns:
        fps_feats: (M, C) float32 cpu — raw backbone features at sampled points
        fps_pos:   (M, 3) float32 cpu — world-frame XYZ of those points
    """
    B, ncam = rgb.shape[:2]
    rgb_flat = einops.rearrange(rgb.cuda(), 'b ncam c h w -> (b ncam) c h w')
    rgb_norm = encoder.normalize(rgb_flat)

    raw = encoder.backbone(rgb_norm)
    if backbone_name == 'clip':
        feat_map = raw['res4']   # (b*ncam, 1024, fh, fw) — most semantic ResNet level
    else:
        feat_map = raw           # (b*ncam, hidden_size, fh, fw) for siglip2/dino

    fh, fw = feat_map.shape[-2:]
    feat_dim = feat_map.shape[1]

    feats = einops.rearrange(feat_map, '(b ncam) c fh fw -> b (ncam fh fw) c', b=B, ncam=ncam)

    pcd_flat = einops.rearrange(pcd.cuda(), 'b ncam c h w -> (b ncam) c h w')
    pcd_interp = F.interpolate(pcd_flat, (fh, fw), mode='bilinear', align_corners=False)
    pos = einops.rearrange(pcd_interp, '(b ncam) c fh fw -> b (ncam fh fw) c', b=B, ncam=ncam)

    # Filter invalid depth before sampling so FPS doesn't anchor to origin
    valid_mask = pos[0].norm(dim=-1) > 0.01  # (N,) for batch element 0
    N = feats.shape[1]
    M = N // fps_factor

    # Both samplers operate on valid points only (no missing-depth pixels)
    valid_idx = valid_mask.nonzero(as_tuple=False).squeeze(1)  # (Nv,)
    pos_valid = pos[:, valid_idx]      # (B, Nv, 3)
    feats_valid = feats[:, valid_idx]  # (B, Nv, C)
    M = min(M, len(valid_idx))

    if sampler == 'uniform_image':
        # Sample uniformly in 2D image/patch space on the FULL (ncam*fh*fw) grid,
        # then drop tokens with missing depth.
        grid_inds = uniform_image_sampler(ncam, fh, fw, fps_factor).to(pos.device)  # (M_grid,)
        full_valid = pos[0].norm(dim=-1) > 0.01   # (ncam*fh*fw,) — before valid_idx filter
        grid_inds = grid_inds[full_valid[grid_inds]]           # keep only valid-depth grid pts
        fps_feats = feats[0, grid_inds].cpu()
        fps_pos = pos[0, grid_inds].cpu()
        return fps_feats, fps_pos

    if sampler == 'fps3d':
        inds = farthest_point_sampler_3d(pos_valid, M)         # (B, M)
    else:  # density in 3D position space
        inds = density_based_sampler(pos_valid, fps_factor)    # (B, M)

    fps_feats = feats_valid.gather(1, inds.unsqueeze(-1).expand(-1, -1, feat_dim))
    fps_pos = pos_valid.gather(1, inds.unsqueeze(-1).expand(-1, -1, 3))

    fps_feats = fps_feats[0].cpu()
    fps_pos = fps_pos[0].cpu()

    valid = fps_pos.norm(dim=-1) > 0.01
    return fps_feats[valid], fps_pos[valid]


# ---------------------------------------------------------------------------
# Correspondence matching
# ---------------------------------------------------------------------------

def compute_correspondences(feats1, feats2, top_k):
    """For each FPS token in scene 1, find its nearest neighbour in scene 2.

    Returns the top_k most confident (src_idx, dst_idx, score) triples.
    """
    f1 = F.normalize(feats1, dim=-1)   # (M1, C)
    f2 = F.normalize(feats2, dim=-1)   # (M2, C)
    sim = f1 @ f2.T                     # (M1, M2)

    best_score, best_idx = sim.max(dim=1)   # (M1,)
    k = min(top_k, len(feats1))
    topk = best_score.topk(k)
    src_inds = topk.indices.numpy()
    dst_inds = best_idx[topk.indices].numpy()
    scores = topk.values.numpy()
    return src_inds, dst_inds, scores


def scores_to_colors(scores):
    """Map similarity scores → green (high) to yellow (low) RGB uint8."""
    t = (scores - scores.min()) / (scores.max() - scores.min() + 1e-6)
    r = (255 * (1.0 - t)).astype(np.uint8)
    g = np.full(len(scores), 200, dtype=np.uint8)
    b = np.zeros(len(scores), dtype=np.uint8)
    return np.stack([r, g, b], axis=1)


# ---------------------------------------------------------------------------
# Rerun
# ---------------------------------------------------------------------------

def log_scene(prefix, cameras, sample, pcd, offset):
    """Log full RGB point cloud and camera poses for one scene."""
    rgb_uint8 = sample["rgb"][0]         # (ncam, 3, H, W) uint8
    extrinsics = sample["extrinsics"][0] # (ncam, 4, 4)
    pcd_np = pcd[0].numpy()              # (ncam, 3, H, W)

    for cam_idx, cam_name in enumerate(cameras):
        rgb_img = rgb_uint8[cam_idx].permute(1, 2, 0).numpy()  # (H, W, 3) uint8
        ext = extrinsics[cam_idx].numpy()

        points = pcd_np[cam_idx].reshape(3, -1).T   # (N, 3)
        colors = rgb_img.reshape(-1, 3)              # (N, 3) uint8
        valid = (np.linalg.norm(points, axis=1) > 0.01) & \
                (np.linalg.norm(points, axis=1) < 5.0)
        points = points[valid][::2] + offset
        colors = colors[valid][::2]

        rr.log(f"{prefix}/pcd/{cam_name}", rr.Points3D(points, colors=colors, radii=0.004))

        cam_t = ext[:3, 3] + offset
        cam_r = ext[:3, :3]
        rr.log(f"{prefix}/camera/{cam_name}", rr.Transform3D(translation=cam_t, mat3x3=cam_r))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path,
                        default=Path("/grogu/user/harshilb/open_drawer_multi_cam/train.zarr"))
    parser.add_argument("--instructions", type=Path,
                        default=Path("instructions/peract/instructions.json"))
    parser.add_argument("--dataset", type=str, default="OrbitalWrist")
    parser.add_argument("--idx1", type=int, default=0)
    parser.add_argument("--idx2", type=int, default=10)
    parser.add_argument("--backbone", type=str, default="clip",
                        choices=["clip", "siglip2", "dino"])
    parser.add_argument("--sampler", type=str, default="fps3d",
                        choices=["density", "fps3d", "uniform_image"],
                        help="density=feature-space density sampling, fps3d=3D position FPS, "
                             "uniform_image=uniform stride grid in 2D patch space")
    parser.add_argument("--fps_factor", type=int, default=5)
    parser.add_argument("--top_k", type=int, default=30)
    parser.add_argument("--scene2_offset", type=float, default=1.5,
                        help="X-axis separation between the two scenes (metres)")
    parser.add_argument("--save", type=Path, default=Path("fps_correspondences.rrd"))
    args = parser.parse_args()

    # Dataset
    dataset_class = fetch_dataset_class(args.dataset)
    dataset = dataset_class(
        root=str(args.data_dir),
        instructions=str(args.instructions),
        chunk_size=1,
        copies=1,
        num_history=1,
    )
    depth2cloud = fetch_depth2cloud(args.dataset)
    cameras = dataset_class.cameras

    # Encoder — backbone weights load from HuggingFace automatically; no checkpoint needed
    print(f"Initialising {args.backbone} encoder...")
    encoder = Encoder(backbone=args.backbone, embedding_dim=120).eval().cuda()

    # Load episodes
    print(f"Loading episodes {args.idx1} and {args.idx2}...")
    rgb1, pcd1, sample1 = load_sample(dataset, depth2cloud, args.idx1)
    rgb2, pcd2, sample2 = load_sample(dataset, depth2cloud, args.idx2)
    print(f"  Scene 1 task: {sample1['task'][0]}")
    print(f"  Scene 2 task: {sample2['task'][0]}")

    # Sampled tokens
    print(f"Extracting tokens (sampler={args.sampler})...")
    fps_feats1, fps_pos1 = extract_fps_tokens(encoder, rgb1, pcd1, args.fps_factor, args.backbone, args.sampler)
    fps_feats2, fps_pos2 = extract_fps_tokens(encoder, rgb2, pcd2, args.fps_factor, args.backbone, args.sampler)
    print(f"  Tokens: scene1={len(fps_pos1)}, scene2={len(fps_pos2)}")

    # Correspondences
    print("Computing correspondences...")
    src_inds, dst_inds, scores = compute_correspondences(fps_feats1, fps_feats2, args.top_k)
    print(f"  Top-{len(scores)} similarity — "
          f"mean={scores.mean():.3f}, min={scores.min():.3f}, max={scores.max():.3f}")

    # Rerun
    rr.init("fps_correspondences")
    rr.save(str(args.save))
    rr.set_time("frame", sequence=0)

    offset = np.array([args.scene2_offset, 0.0, 0.0])

    print("Logging to Rerun...")
    log_scene("world/scene1", cameras, sample1, pcd1, np.zeros(3))
    log_scene("world/scene2", cameras, sample2, pcd2, offset)

    fps_pos1_np = fps_pos1.numpy()
    fps_pos2_off = fps_pos2.numpy() + offset

    rr.log("world/scene1/fps_tokens",
           rr.Points3D(fps_pos1_np, colors=[0, 100, 255], radii=0.015))
    rr.log("world/scene2/fps_tokens",
           rr.Points3D(fps_pos2_off, colors=[255, 100, 0], radii=0.015))

    lines = [[fps_pos1_np[s].tolist(), fps_pos2_off[d].tolist()]
             for s, d in zip(src_inds, dst_inds)]
    line_colors = scores_to_colors(scores)
    rr.log("world/correspondences",
           rr.LineStrips3D(lines, colors=line_colors, radii=0.003))

    rr.log("world/origin", rr.Arrows3D(
        origins=[[0, 0, 0]] * 3, vectors=np.eye(3) * 0.15,
        colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]]
    ))

    print(f"Saved to {args.save}  —  open with: rerun {args.save}")


if __name__ == "__main__":
    main()
