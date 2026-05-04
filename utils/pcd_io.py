import os

import einops
import numpy as np
import torch


def save_ply(path, xyz, rgb):
    """Save a colored point cloud as a binary PLY file.

    Args:
        path: output .ply path
        xyz: (N, 3) float — XYZ positions
        rgb: (N, 3) float in [0, 1] or uint8 in [0, 255]
    """
    if hasattr(xyz, 'detach'):
        xyz = xyz.detach().cpu().numpy()
    if hasattr(rgb, 'detach'):
        rgb = rgb.detach().cpu().numpy()

    xyz = np.asarray(xyz, dtype=np.float32)
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8:
        rgb = (rgb.clip(0.0, 1.0) * 255).astype(np.uint8)

    N = xyz.shape[0]
    header = (
        f"ply\n"
        f"format binary_little_endian 1.0\n"
        f"element vertex {N}\n"
        f"property float x\n"
        f"property float y\n"
        f"property float z\n"
        f"property uchar red\n"
        f"property uchar green\n"
        f"property uchar blue\n"
        f"end_header\n"
    )
    dt = np.dtype([
        ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
    ])
    data = np.empty(N, dtype=dt)
    data['x'], data['y'], data['z'] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    data['red'], data['green'], data['blue'] = rgb[:, 0], rgb[:, 1], rgb[:, 2]

    with open(path, 'wb') as f:
        f.write(header.encode())
        f.write(data.tobytes())


def save_encoder_debug_pcd(debug_dir, step, pcd_curr, fps_scene_pos, rgb3d):
    """Save full and FPS-subsampled colored point clouds for one encoder step.

    Interpolates rgb3d to match the backbone feature resolution of pcd_curr,
    then saves both .ply (for visualization) and .npz (for programmatic access).

    Args:
        debug_dir:      output directory
        step:           integer step index used in filenames
        pcd_curr:       (B, Np, 3) backbone-interpolated point positions
        fps_scene_pos:  (B, M, 3) density-subsampled point positions
        rgb3d:          (B, ncam, 3, H, W) or (B, nhist, ncam, 3, H, W) raw RGB input
    """
    from torch.nn.functional import interpolate as F_interp

    os.makedirs(debug_dir, exist_ok=True)

    ncam = rgb3d.shape[-4]
    feat_h = feat_w = int((pcd_curr.shape[1] // ncam) ** 0.5)

    # Current frame raw RGB before backbone normalization
    rgb_raw = rgb3d[:, -1] if rgb3d.ndim == 6 else rgb3d  # (B, ncam, 3, H, W)

    # Downsample to backbone feature resolution so each point has a matching color
    rgb_interp = F_interp(
        einops.rearrange(rgb_raw, 'b ncam c h w -> (b ncam) c h w').float(),
        size=(feat_h, feat_w), mode='bilinear', align_corners=False,
    )
    rgb_interp = einops.rearrange(
        rgb_interp, '(b ncam) c h w -> b (ncam h w) c', ncam=ncam,
    )  # (B, Np, 3) in [0, 1]

    # fps_scene_pos is gathered from pcd_curr rows, so nearest-neighbor gives exact matches
    dists = torch.cdist(fps_scene_pos[0].float(), pcd_curr[0].float())
    fps_rgb = rgb_interp[0][dists.argmin(dim=1)]  # (M, 3)

    save_ply(os.path.join(debug_dir, f"pcd_{step:04d}.ply"), pcd_curr[0], rgb_interp[0])
    save_ply(os.path.join(debug_dir, f"fps_pcd_{step:04d}.ply"), fps_scene_pos[0], fps_rgb)
    np.savez(
        os.path.join(debug_dir, f"pcd_{step:04d}.npz"),
        pcd=pcd_curr.detach().cpu().numpy(),
        pcd_rgb=rgb_interp.detach().cpu().numpy(),
        fps_pcd=fps_scene_pos.detach().cpu().numpy(),
        fps_rgb=fps_rgb.detach().cpu().numpy(),
    )
