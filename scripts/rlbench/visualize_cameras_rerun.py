"""
Launch RLBench / CoppeliaSim, reset a task, and visualise in Rerun:
  - merged point cloud from the scene's existing cameras
  - original camera frustums (with their captured RGB)
  - 10 candidate external cameras as frustums (with their captured RGB)

Requires COPPELIASIM_ROOT, LD_LIBRARY_PATH, QT_QPA_PLATFORM_PLUGIN_PATH to be
set (see scripts/rlbench/README_RLBench.md) and a virtual display:

    xvfb-run -a python scripts/rlbench/visualize_cameras_rerun.py \\
        --task close_jar \\
        --out  camera_viz.rrd

    # Bimanual (PerAct2):
    xvfb-run -a python scripts/rlbench/visualize_cameras_rerun.py \\
        --task bimanual_lift_tray --bimanual \\
        --out  camera_viz.rrd

Then copy camera_viz.rrd to your machine and open with:
    rerun camera_viz.rrd
"""

import argparse
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation as ScipyR
import rerun as rr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))


# ── Geometry helpers ─────────────────────────────────────────────────────────

def _normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def look_at(eye, target, world_up=np.array([0., 0., 1.])):
    """Camera-to-world rotation (3×3). Z=forward, X=right, Y=down-in-image."""
    z = _normalize(np.asarray(target, float) - np.asarray(eye, float))
    if abs(float(np.dot(z, world_up))) > 0.99:
        world_up = np.array([0., 1., 0.])
    x = _normalize(np.cross(world_up, z))
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def R_to_pyrep_quat(R):
    """3×3 cam-to-world rotation → [qx, qy, qz, qw] for PyRep set_pose."""
    return ScipyR.from_matrix(R).as_quat()   # scipy returns xyzw


# ── Candidate camera positions ───────────────────────────────────────────────

def structured_cameras(center, radius, z_table):
    """10 pre-defined positions that give good workspace coverage."""
    cx, cy, cz = center
    r = radius
    configs = [
        ("front",          [cx - r*1.8, cy,          cz + r*0.2]),
        ("front_high",     [cx - r*1.4, cy,          cz + r*0.9]),
        ("left_shoulder",  [cx - r*0.2, cy + r*1.5,  cz + r*1.3]),
        ("right_shoulder", [cx - r*0.2, cy - r*1.5,  cz + r*1.3]),
        ("overhead",       [cx,          cy,          cz + r*2.2]),
        ("left_side",      [cx + r*0.4,  cy + r*1.9,  cz + r*0.4]),
        ("right_side",     [cx + r*0.4,  cy - r*1.9,  cz + r*0.4]),
        ("back_high",      [cx + r*1.6,  cy,          cz + r*1.0]),
        ("front_low",      [cx - r*2.0, cy,          cz - r*0.2]),
        ("wrist_approx",   [cx + r*0.1, cy + r*0.3, cz + r*0.15]),
    ]
    cameras = []
    for name, pos_list in configs:
        pos = np.array(pos_list, dtype=float)
        pos[2] = max(pos[2], z_table + 0.05)
        cameras.append({"name": name, "pos": pos, "R": look_at(pos, center)})
    return cameras


# ── Point cloud helpers ───────────────────────────────────────────────────────

def sensor_to_pointcloud(sensor, max_pts=8000, rng=None):
    """
    Capture RGB + depth from a VisionSensor and return
    (pts_world, colors_uint8): (N,3) float32, (N,3) uint8.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    rgb   = sensor.capture_rgb()                         # (H, W, 3) float [0,1]
    depth = sensor.capture_depth(in_meters=True)         # (H, W) float metres
    pcd   = sensor.capture_pointcloud()                  # (H, W, 3) world XYZ

    H, W = depth.shape
    pts  = pcd.reshape(-1, 3).astype(np.float32)
    cols = (rgb.reshape(-1, 3) * 255).clip(0, 255).astype(np.uint8)

    valid = (depth.reshape(-1) > 0.01) & np.isfinite(pts).all(1)
    pts, cols = pts[valid], cols[valid]

    if len(pts) > max_pts:
        idx  = rng.choice(len(pts), max_pts, replace=False)
        pts, cols = pts[idx], cols[idx]

    return pts, cols


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task",        default="close_jar",
                   help="RLBench task name (default: close_jar)")
    p.add_argument("--variation",   type=int, default=0)
    p.add_argument("--bimanual",    action="store_true",
                   help="Use bimanual (PerAct2) scene")
    p.add_argument("--image_size",  type=int, default=256)
    p.add_argument("--fov_deg",     type=float, default=60.0,
                   help="FOV for candidate cameras (default: 60)")
    p.add_argument("--out",         default="camera_viz.rrd",
                   help="Output .rrd file (default: camera_viz.rrd)")
    p.add_argument("--spawn",       action="store_true",
                   help="Spawn Rerun viewer directly (needs X display)")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── RLBench imports ───────────────────────────────────────────────────────
    try:
        from pyrep.const import RenderMode
        from rlbench.observation_config import ObservationConfig, CameraConfig
        from rlbench.environment import Environment
        from rlbench.action_modes.action_mode import MoveArmThenGripper
        from rlbench.action_modes.arm_action_modes import JointVelocity
        from rlbench.action_modes.gripper_action_modes import Discrete
        from pyrep.objects.vision_sensor import VisionSensor
    except ImportError as e:
        sys.exit("[ERROR] RLBench/PyRep import failed: {}\n"
                 "Set COPPELIASIM_ROOT, LD_LIBRARY_PATH, "
                 "QT_QPA_PLATFORM_PLUGIN_PATH first.".format(e))

    # ── Scene / action mode ───────────────────────────────────────────────────
    if args.bimanual:
        from rlbench.action_modes.action_mode import BimanualMoveArmThenGripper
        from rlbench.action_modes.arm_action_modes import BimanualEndEffectorPoseViaPlanning
        from rlbench.action_modes.gripper_action_modes import BimanualDiscrete
        action_mode    = BimanualMoveArmThenGripper(
            BimanualEndEffectorPoseViaPlanning(), BimanualDiscrete())
        robot_setup    = "dual_panda"
        scene_cam_names = ["front", "wrist_left", "wrist_right"]
    else:
        action_mode    = MoveArmThenGripper(JointVelocity(), Discrete())
        robot_setup    = "panda"
        scene_cam_names = ["front", "over_shoulder_left", "over_shoulder_right"]

    # ── ObservationConfig: enable RGB + depth + point_cloud for scene cams ───
    sz  = (args.image_size, args.image_size)
    on  = CameraConfig(rgb=True, depth=True, point_cloud=True, mask=False,
                       image_size=sz, render_mode=RenderMode.OPENGL3)
    off = CameraConfig(); off.set_all(False)

    all_known = ["front", "over_shoulder_left", "over_shoulder_right",
                 "wrist", "wrist_left", "wrist_right"]
    cam_cfgs  = {n: (on if n in scene_cam_names else off) for n in all_known}

    obs_config = ObservationConfig(
        camera_configs=cam_cfgs,
        joint_velocities=False, joint_positions=False, joint_forces=False,
        gripper_open=False, gripper_pose=False, task_low_dim_state=False,
    )

    # ── Launch env ────────────────────────────────────────────────────────────
    print("[INFO] Launching RLBench ({} / {})…".format(
          args.task, "bimanual" if args.bimanual else "unimanual"))
    env = Environment(action_mode, "", obs_config,
                      headless=True, robot_setup=robot_setup)
    env.launch()

    # ── Load task ─────────────────────────────────────────────────────────────
    if args.bimanual:
        import importlib
        mod        = importlib.import_module("rlbench.bimanual_tasks.{}".format(args.task))
        class_name = "".join(w.capitalize() for w in args.task.split("_"))
        task_class = getattr(mod, class_name)
    else:
        from rlbench.backend.utils import task_file_to_task_class
        task_class = task_file_to_task_class(args.task)

    task_env = env.get_task(task_class)
    task_env.set_variation(args.variation)
    _, _ = task_env.reset()
    print("[INFO] Task reset done.")

    # ── Workspace geometry from scene ─────────────────────────────────────────
    scene     = env._scene
    ws_pos    = np.array(scene._workspace.get_position())
    ws_bbox   = scene._workspace.get_bounding_box()
    ws_radius = max(abs(ws_bbox[1] - ws_bbox[0]),
                    abs(ws_bbox[3] - ws_bbox[2])) / 2.0 + 0.2
    z_table   = ws_pos[2]
    center    = np.array([ws_pos[0], ws_pos[1], ws_pos[2] + 0.2])
    print("[INFO] Workspace: center={}  radius={:.3f}m  table_z={:.3f}m".format(
          center.round(3), ws_radius, z_table))

    # ── Capture point cloud from existing scene cameras ───────────────────────
    rng      = np.random.default_rng(0)
    all_pts, all_cols = [], []
    scene_cam_data    = []   # (name, E, K, rgb_img) for Rerun logging

    for name in scene_cam_names:
        sensor = scene.camera_sensors[name]
        sensor.set_explicit_handling(1)
        sensor.handle_explicitly()

        pts, cols = sensor_to_pointcloud(sensor, max_pts=8000, rng=rng)
        all_pts.append(pts)
        all_cols.append(cols)

        E   = sensor.get_matrix().astype(np.float64)          # (4,4) cam-to-world
        K   = sensor.get_intrinsic_matrix().astype(np.float64) # (3,3)
        rgb = (sensor.capture_rgb() * 255).clip(0, 255).astype(np.uint8)

        scene_cam_data.append({"name": name, "E": E, "K": K, "rgb": rgb})
        print("[INFO]  scene cam '{}': {:,} pts".format(name, len(pts)))

    merged_pts  = np.concatenate(all_pts,  axis=0)
    merged_cols = np.concatenate(all_cols, axis=0)
    print("[INFO] Total: {:,} world points".format(len(merged_pts)))

    # ── Create 10 candidate VisionSensors ─────────────────────────────────────
    candidate_cams = structured_cameras(center, ws_radius, z_table)
    candidate_data = []   # (name, E, K, rgb_img)

    for cam in candidate_cams:
        pos  = cam["pos"]
        R    = cam["R"]
        quat = R_to_pyrep_quat(R)            # [qx, qy, qz, qw]
        pose = pos.tolist() + quat.tolist()  # 7-element PyRep pose

        sensor = VisionSensor.create(
            resolution=[args.image_size, args.image_size],
            explicit_handling=True,
            view_angle=args.fov_deg,
            near_clipping_plane=0.01,
            far_clipping_plane=10.0,
            render_mode=RenderMode.OPENGL3,
        )
        sensor.set_pose(pose)
        sensor.handle_explicitly()

        E   = sensor.get_matrix().astype(np.float64)
        K   = sensor.get_intrinsic_matrix().astype(np.float64)
        rgb = (sensor.capture_rgb() * 255).clip(0, 255).astype(np.uint8)

        candidate_data.append({"name": cam["name"], "E": E, "K": K, "rgb": rgb})
        print("[INFO]  candidate cam '{}'".format(cam["name"]))

    env.shutdown()

    # ── Log to Rerun ──────────────────────────────────────────────────────────
    rr.init("rlbench_camera_viz", spawn=args.spawn)

    # Point cloud
    rr.log("world/pcd", rr.Points3D(merged_pts, colors=merged_cols, radii=0.003))

    # Scene cameras (orange)
    for cam in scene_cam_data:
        R = cam["E"][:3, :3]
        t = cam["E"][:3, 3]
        path = "world/scene_cams/{}".format(cam["name"])
        rr.log(path, rr.Transform3D(translation=t, mat3x3=R))
        rr.log(path, rr.Pinhole(image_from_camera=cam["K"],
                                width=args.image_size, height=args.image_size,
                                image_plane_distance=0.15))
        rr.log("{}/rgb".format(path), rr.Image(cam["rgb"]))

    rr.log(
        "world/scene_cams/_labels",
        rr.Points3D(
            np.array([c["E"][:3, 3] for c in scene_cam_data], dtype=np.float32),
            labels=[c["name"] for c in scene_cam_data],
            radii=0.02,
            colors=[[255, 160, 0]] * len(scene_cam_data),
        ),
    )

    # Candidate cameras (cyan)
    for cam in candidate_data:
        R = cam["E"][:3, :3]
        t = cam["E"][:3, 3]
        path = "world/candidate_cams/{}".format(cam["name"])
        rr.log(path, rr.Transform3D(translation=t, mat3x3=R))
        rr.log(path, rr.Pinhole(image_from_camera=cam["K"],
                                width=args.image_size, height=args.image_size,
                                image_plane_distance=0.25))
        rr.log("{}/rgb".format(path), rr.Image(cam["rgb"]))

    rr.log(
        "world/candidate_cams/_labels",
        rr.Points3D(
            np.array([c["E"][:3, 3] for c in candidate_data], dtype=np.float32),
            labels=[c["name"] for c in candidate_data],
            radii=0.02,
            colors=[[100, 200, 255]] * len(candidate_data),
        ),
    )

    if not args.spawn:
        rr.save(args.out)
        print("\n[DONE] Saved: {}".format(os.path.abspath(args.out)))
        print("       rerun {}".format(args.out))


if __name__ == "__main__":
    main()
