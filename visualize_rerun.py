"""Rerun-based visualizer for multi-camera 3D data.

Usage:
    python visualize_rerun.py --data_dir /path/to/data --instructions /path/to/instructions.json --dataset Peract

This visualizer shows:
- RGB images from each camera
- 3D point clouds in world coordinates (transformed using extrinsics/intrinsics)
- Camera poses in the world frame
- Robot end-effector pose
- Action trajectories
"""

import argparse
from pathlib import Path

import numpy as np
import rerun as rr
import torch

from datasets import fetch_dataset_class
from utils.depth2cloud import fetch_depth2cloud


def quaternion_to_rotation_matrix(q):
    """Convert quaternion (x, y, z, w) to 3x3 rotation matrix."""
    x, y, z, w = q
    
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])


def visualize_sample(sample, dataset_class, depth2cloud=None, frame_idx=0, show_pcds=True, show_frustums=True):
    """Visualize a single sample using Rerun."""
    
    # Set the time for this frame
    rr.set_time("frame", sequence=frame_idx)
    
    rgb = sample["rgb"]
    action = sample["action"]
    proprio = sample["proprioception"]


    if "depth" in sample and depth2cloud is not None:
        depth = sample["depth"]  # (chunk_size, n_cam, H, W)
        extrinsics = sample["extrinsics"]  # (chunk_size, n_cam, 4, 4)
        intrinsics = sample["intrinsics"]  # (chunk_size, n_cam, 3, 3)
        
        # Convert depth to point cloud in world coordinates
        with torch.no_grad():
            pcds = depth2cloud(
                depth.cuda().to(torch.float32),
                extrinsics.cuda().to(torch.float32),
                intrinsics.cuda().to(torch.float32)
            )  # (chunk_size, n_cam, 3, H, W)
            pcds = pcds.cpu().numpy()
        
        # Visualize cameras and point clouds
        n_cam = rgb.shape[1]
        cameras = getattr(dataset_class, 'cameras', [f'cam_{i}' for i in range(n_cam)])
        
        
        for cam_idx in range(n_cam):
            cam_name = cameras[cam_idx] if cam_idx < len(cameras) else f'cam_{cam_idx}'
            
            rgb_img = rgb[0, cam_idx].permute(1, 2, 0).numpy().astype(np.uint8)
            ext = extrinsics[0, cam_idx].numpy()
            cam_translation = ext[:3, 3]
            cam_rotation = ext[:3, :3]
            
            intr = intrinsics[0, cam_idx].numpy()
            fx, fy = intr[0, 0], intr[1, 1]
            cx, cy = intr[0, 2], intr[1, 2]
            h, w = depth.shape[-2:]
            
            
            # Camera pose and pinhole
            rr.log(f"world/cameras/{cam_name}", rr.Transform3D(translation=cam_translation, mat3x3=cam_rotation))
            rr.log(f"world/cameras/{cam_name}", rr.Pinhole(resolution=[w, h], focal_length=[fx, fy], principal_point=[cx, cy]))
            rr.log(f"world/cameras/{cam_name}", rr.Image(rgb_img))
            
            # Add visible marker
            rr.log(f"world/cameras/{cam_name}/marker", rr.Points3D([[0, 0, 0]], colors=[255, 255, 0], radii=0.03))
            
            if show_pcds:
                pcd = pcds[0, cam_idx]
                points = pcd.reshape(3, -1).T
                colors = rgb_img.reshape(-1, 3)
                valid_mask = (np.linalg.norm(points, axis=1) > 0.01) & (np.linalg.norm(points, axis=1) < 5.0)
                points = points[valid_mask][::4]
                colors = colors[valid_mask][::4]
                
                rr.log(f"world/point_clouds/{cam_name}", rr.Points3D(points, colors=colors, radii=0.006))
    
    elif "pcd" in sample and show_pcds:
        pcd = sample["pcd"]
        n_cam = rgb.shape[1]
        cameras = getattr(dataset_class, 'cameras', [f'cam_{i}' for i in range(n_cam)])
        
        for cam_idx in range(n_cam):
            cam_name = cameras[cam_idx] if cam_idx < len(cameras) else f'cam_{cam_idx}'
            rgb_img = rgb[0, cam_idx].permute(1, 2, 0).numpy().astype(np.uint8)
            points = pcd[0, cam_idx].reshape(3, -1).T.numpy()
            colors = rgb_img.reshape(-1, 3)
            valid_mask = (np.linalg.norm(points, axis=1) > 0.01) & (np.linalg.norm(points, axis=1) < 5.0)
            points = points[valid_mask][::4]
            colors = colors[valid_mask][::4]
            rr.log(f"world/point_clouds/{cam_name}", rr.Points3D(points, colors=colors, radii=0.006))
    
    # Visualize proprioception history as PAST TRAJECTORY (RED)
    n_hands = proprio.shape[2] if len(proprio.shape) >= 3 else 1
    for hand_idx in range(n_hands):
        hand_name = "left" if hand_idx == 0 else "right" if n_hands == 2 else "hand"
        
        # Get all history states for this hand: (nhist, 8)
        if len(proprio.shape) == 4:
            hand_history = proprio[0, :, hand_idx].numpy()  # (nhist, 8)
        else:
            hand_history = proprio[0].numpy()  # (nhist, 8)
        
        # Extract positions from history
        history_pos = hand_history[:, :3]  # (nhist, 3)
        
        
        # Current state (most recent)
        current_pos = history_pos[0]  # First in history is current
        current_quat = hand_history[0, 3:7]
        current_rot = quaternion_to_rotation_matrix(current_quat)
        
        # Plot PAST trajectory in RED - directly in world frame
        past_trajectory = history_pos[::-1]
        if len(np.unique(past_trajectory, axis=0)) > 1:
            rr.log(f"world/trajectory/{hand_name}/past",
                   rr.LineStrips3D([past_trajectory], colors=[255, 0, 0], radii=0.015))
        
        for t_idx, pos in enumerate(past_trajectory):
            alpha = int(100 + (t_idx / len(past_trajectory)) * 155)
            rr.log(f"world/trajectory/{hand_name}/past_point_{t_idx}",
                   rr.Points3D([pos], colors=[[alpha, 0, 0]], radii=0.04))
        
        # Current robot gripper - sphere directly in world coordinates
        rr.log(f"world/robot/{hand_name}_sphere", 
               rr.Points3D([current_pos], colors=[[255, 0, 0]], radii=0.05))
        
        # Current pose with coordinate frame (axes in local coordinates)
        rr.log(f"world/robot/{hand_name}_frame", rr.Transform3D(translation=current_pos, mat3x3=current_rot))
        rr.log(f"world/robot/{hand_name}_frame/axes", 
               rr.Arrows3D(origins=[[0, 0, 0]] * 3, vectors=np.eye(3) * 0.08,
                          colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]]))
    
    # Visualize FUTURE ACTION targets (BLUE/CYAN)
    n_hands_action = action.shape[2] if len(action.shape) >= 3 else 1
    for hand_idx in range(n_hands_action):
        hand_name = "left" if hand_idx == 0 else "right" if n_hands_action == 2 else "hand"
        hand_actions = action[0, :, hand_idx].numpy() if len(action.shape) == 4 else action[0].numpy()
        action_pos = hand_actions[:, :3]
        action_quat = hand_actions[:, 3:7]
        
        
        # Plot action target(s) - sphere in world frame, axes in local frame
        for t_idx, (pos, quat) in enumerate(zip(action_pos, action_quat)):
            # Target sphere directly in world coordinates
            rr.log(f"world/action/{hand_name}_target_{t_idx}_sphere",
                   rr.Points3D([pos], colors=[[0, 200, 255]], radii=0.06))
            
            # Coordinate frame at target (axes in local coordinates)
            rr.log(f"world/action/{hand_name}_target_{t_idx}_frame", 
                   rr.Transform3D(translation=pos, mat3x3=quaternion_to_rotation_matrix(quat)))
            rr.log(f"world/action/{hand_name}_target_{t_idx}_frame/axes",
                   rr.Arrows3D(origins=[[0, 0, 0]] * 3, vectors=np.eye(3) * 0.1,
                              colors=[[100, 200, 255], [100, 255, 200], [200, 200, 255]]))
    
    rr.log("world/origin", rr.Arrows3D(origins=[[0, 0, 0]] * 3, vectors=np.eye(3) * 0.15,
                                        colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]]))




def main():
    parser = argparse.ArgumentParser(description="Visualize dataset with Rerun")
    parser.add_argument("--data_dir", type=Path, required=True, help="Path to dataset directory")
    parser.add_argument("--instructions", type=Path, required=True, help="Path to instructions JSON")
    parser.add_argument("--dataset", type=str, default="Peract", help="Dataset class name")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to visualize")
    parser.add_argument("--chunk_size", type=int, default=1, help="Chunk size for dataset")
    parser.add_argument("--save", type=Path, default=None, help="Save recording to RRD file")
    parser.add_argument("--no_point_clouds", action="store_true", help="Skip point cloud visualization")
    parser.add_argument("--no_frustums", action="store_true", help="Skip camera frustum visualization")
    args = parser.parse_args()
    
    # Initialize Rerun
    rr.init("3d_flowmatch_visualizer")
    # rr.serve(open_browser=False)
    server_uri = rr.serve_grpc()

    # Connect the web viewer to the gRPC server and open it in the browser
    rr.serve_web_viewer(connect_to=server_uri)
    
    # Load dataset
    dataset_class = fetch_dataset_class(args.dataset)
    dataset = dataset_class(
        root=args.data_dir,
        instructions=args.instructions,
        chunk_size=args.chunk_size,
        copies=1,
        mem_limit=8
    )
    
    depth2cloud = fetch_depth2cloud(args.dataset)
    
    for i in range(min(args.num_samples, len(dataset))):
        sample = dataset[i]
        for key in sample:
            if not isinstance(sample[key], torch.Tensor):
                if isinstance(sample[key], list) and not isinstance(sample[key][0], str):
                    sample[key] = torch.tensor(sample[key])
        
        visualize_sample(sample, dataset_class, depth2cloud, frame_idx=i, 
                        show_pcds=not args.no_point_clouds,
                        show_frustums=not args.no_frustums)
    
    input("\nPress Enter to exit...")


if __name__ == "__main__":
    main()
