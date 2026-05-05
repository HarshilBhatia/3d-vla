"""Visualize FPS token correspondences between two orbital camera viewpoints.

Uses raw pretrained CLIP/SigLIP2 backbone features (no checkpoint needed) to
compute density-based FPS tokens, then matches them across viewpoints by
cosine similarity and shows the result in Rerun.

Usage:
    python scripts/helpers/visualize_fps_correspondences.py \\
        --idx1 0 --idx2 10 --top_k 30 --backbone clip
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
    return rgb, pcd, extrinsics, sample


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_fps_tokens(encoder, rgb, pcd, fps_factor, backbone_name):
    """Extract FPS token positions and raw backbone features.

    Bypasses the trained FPN / vis_lang_attention — uses only the frozen
    pretrained backbone, so no checkpoint is required.

    Returns:
        fps_feats: (M, C) float32 cpu — raw backbone features at FPS points
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

    inds = density_based_sampler(feats, fps_factor)                              # (B, M)
    fps_feats = feats.gather(1, inds.unsqueeze(-1).expand(-1, -1, feat_dim))    # (B, M, C)
    fps_pos = pos.gather(1, inds.unsqueeze(-1).expand(-1, -1, 3))               # (B, M, 3)

    fps_feats = fps_feats[0].cpu()
    fps_pos = fps_pos[0].cpu()

    # Drop points with missing depth (at or near world origin)
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


def scores_to_colors_float(scores):
    """Map similarity scores → green (high) to yellow (low) RGB float [0,1]."""
    t = (scores - scores.min()) / (scores.max() - scores.min() + 1e-6)
    return np.stack([1.0 - t, np.full_like(t, 0.8), np.zeros_like(t)], axis=1)


def get_pcd_points(pcd, sample, stride=4):
    """Return (N,3) world points and (N,3) RGB float colors from all cameras."""
    rgb_uint8 = sample["rgb"][0]    # (ncam, 3, H, W) uint8
    pcd_np = pcd[0].numpy()         # (ncam, 3, H, W)
    all_pts, all_col = [], []
    for cam_idx in range(pcd_np.shape[0]):
        pts = pcd_np[cam_idx].reshape(3, -1).T
        col = rgb_uint8[cam_idx].permute(1, 2, 0).numpy().reshape(-1, 3) / 255.0
        valid = (np.linalg.norm(pts, axis=1) > 0.01) & (np.linalg.norm(pts, axis=1) < 5.0)
        all_pts.append(pts[valid][::stride])
        all_col.append(col[valid][::stride])
    return np.concatenate(all_pts), np.concatenate(all_col)


# ---------------------------------------------------------------------------
# Rerun logging helpers
# ---------------------------------------------------------------------------

def log_scene(prefix, cameras, sample, pcd, offset):
    """Log full RGB point cloud and camera poses for one scene."""
    rgb_uint8 = sample["rgb"][0]             # (ncam, 3, H, W) uint8
    extrinsics = sample["extrinsics"][0]     # (ncam, 4, 4)
    pcd_np = pcd[0].numpy()                  # (ncam, 3, H, W)

    for cam_idx, cam_name in enumerate(cameras):
        rgb_img = rgb_uint8[cam_idx].permute(1, 2, 0).numpy()  # (H, W, 3) uint8
        ext = extrinsics[cam_idx].numpy()

        points = pcd_np[cam_idx].reshape(3, -1).T
        colors = rgb_img.reshape(-1, 3)
        valid = (np.linalg.norm(points, axis=1) > 0.01) & \
                (np.linalg.norm(points, axis=1) < 5.0)
        points = points[valid][::4] + offset
        colors = colors[valid][::4]

        rr.log(f"{prefix}/pcd/{cam_name}", rr.Points3D(points, colors=colors, radii=0.005))

        cam_t = ext[:3, 3] + offset
        cam_r = ext[:3, :3]
        rr.log(f"{prefix}/camera/{cam_name}", rr.Transform3D(translation=cam_t, mat3x3=cam_r))


# ---------------------------------------------------------------------------
# Matplotlib visualization
# ---------------------------------------------------------------------------

def visualize_matplotlib(fps_pos1, fps_pos2, src_inds, dst_inds, scores,
                          pcd1, sample1, pcd2, sample2, offset, save_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(14, 7))
    ax = fig.add_subplot(111, projection='3d')

    fps_pos1_np = fps_pos1.numpy()
    fps_pos2_off = fps_pos2.numpy() + offset

    # Background point clouds (heavily subsampled, faint)
    pts1, col1 = get_pcd_points(pcd1, sample1, stride=16)
    pts2, col2 = get_pcd_points(pcd2, sample2, stride=16)
    ax.scatter(*pts1.T, c=col1, s=0.5, alpha=0.15, linewidths=0)
    ax.scatter(*(pts2 + offset).T, c=col2, s=0.5, alpha=0.15, linewidths=0)

    # FPS tokens
    ax.scatter(*fps_pos1_np.T, c=[[0, 0.4, 1.0]], s=18, zorder=3, label='scene 1 FPS')
    ax.scatter(*fps_pos2_off.T, c=[[1.0, 0.4, 0]], s=18, zorder=3, label='scene 2 FPS')

    # Correspondence lines colored by score
    line_colors = scores_to_colors_float(scores)
    for i, (s, d) in enumerate(zip(src_inds, dst_inds)):
        p1, p2 = fps_pos1_np[s], fps_pos2_off[d]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                color=line_colors[i], linewidth=0.8, alpha=0.8)

    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.legend(loc="upper left", markerscale=2)
    ax.set_title(f"FPS token correspondences  (top-{len(scores)}, green=high sim)")
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150)
    print(f"Saved to {save_path}")


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
    parser.add_argument("--idx1", type=int, default=0,
                        help="First episode index")
    parser.add_argument("--idx2", type=int, default=10,
                        help="Second episode index (different camera group)")
    parser.add_argument("--backbone", type=str, default="clip",
                        choices=["clip", "siglip2", "dino"])
    parser.add_argument("--fps_factor", type=int, default=5,
                        help="Density sampler subsampling factor")
    parser.add_argument("--top_k", type=int, default=30,
                        help="Number of correspondences to show")
    parser.add_argument("--scene2_offset", type=float, default=1.5,
                        help="X-axis separation between the two scenes in Rerun (metres)")
    parser.add_argument("--vis", type=str, default="matplotlib",
                        choices=["matplotlib", "rerun"],
                        help="Visualization backend")
    parser.add_argument("--save", type=Path, default=None,
                        help="Output file (.png for matplotlib, .rrd for rerun)")
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
    rgb1, pcd1, _, sample1 = load_sample(dataset, depth2cloud, args.idx1)
    rgb2, pcd2, _, sample2 = load_sample(dataset, depth2cloud, args.idx2)
    print(f"  Scene 1 task: {sample1['task'][0]}")
    print(f"  Scene 2 task: {sample2['task'][0]}")

    # FPS tokens
    print("Extracting FPS tokens...")
    fps_feats1, fps_pos1 = extract_fps_tokens(encoder, rgb1, pcd1, args.fps_factor, args.backbone)
    fps_feats2, fps_pos2 = extract_fps_tokens(encoder, rgb2, pcd2, args.fps_factor, args.backbone)
    print(f"  FPS tokens: scene1={len(fps_pos1)}, scene2={len(fps_pos2)}")

    # Correspondences
    print("Computing correspondences...")
    src_inds, dst_inds, scores = compute_correspondences(fps_feats1, fps_feats2, args.top_k)
    print(f"  Top-{len(scores)} similarity — "
          f"mean={scores.mean():.3f}, min={scores.min():.3f}, max={scores.max():.3f}")

    offset = np.array([args.scene2_offset, 0.0, 0.0])

    if args.vis == "matplotlib":
        save_path = args.save or Path("fps_correspondences.png")
        visualize_matplotlib(fps_pos1, fps_pos2, src_inds, dst_inds, scores,
                             pcd1, sample1, pcd2, sample2, offset, save_path)

    else:  # rerun
        save_path = args.save or Path("fps_correspondences.rrd")
        rr.init("fps_correspondences")
        rr.save(str(save_path))
        rr.set_time("frame", sequence=0)

        offset1 = np.array([0.0, 0.0, 0.0])
        print("Logging to Rerun...")
        log_scene("world/scene1", cameras, sample1, pcd1, offset1)
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
        print(f"Saved to {save_path}  —  open with: rerun {save_path}")


if __name__ == "__main__":
    main()
