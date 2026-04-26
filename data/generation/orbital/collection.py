"""
Orbital episode collection utilities.

Covers everything needed to run a single collection session:
  - Camera pose loading from JSON
  - VisionSensor creation and extrinsics capture
  - RLBench ObservationConfig factory
  - Raw episode saving (RGB/depth PNGs + pkl)
  - Single-episode debug video and zarr saving
  - collect_one_episode() — the main per-episode driver
"""

import json
import os
import pickle
import time

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as ScipyR

from data.generation.orbital.constants import NCAM, NHAND, num2id


# ---------------------------------------------------------------------------
# Camera pose helpers
# ---------------------------------------------------------------------------

def R_to_pyrep_quat(R_mat):
    """3×3 cam-to-world rotation → [qx, qy, qz, qw] for PyRep set_pose."""
    return ScipyR.from_matrix(R_mat).as_quat()  # scipy returns xyzw


def load_group_cameras(cameras_file, group):
    """Return (left_cam, right_cam) dicts with keys 'pos' and 'R' for `group`."""
    with open(cameras_file) as f:
        groups = json.load(f)
    for entry in groups:
        if entry["group"] == group:
            return (
                {"pos": np.array(entry["left"]["pos"]),
                 "R":   np.array(entry["left"]["R"])},
                {"pos": np.array(entry["right"]["pos"]),
                 "R":   np.array(entry["right"]["R"])},
            )
    raise ValueError("Group {} not found in {}".format(group, cameras_file))


# ---------------------------------------------------------------------------
# Sensor creation
# ---------------------------------------------------------------------------

def create_orbital_sensor(pos, R_mat, image_size, fov_deg):
    """Spawn a VisionSensor at the given pose and return it."""
    from pyrep.const import RenderMode
    from pyrep.objects.vision_sensor import VisionSensor

    pose   = pos.tolist() + R_to_pyrep_quat(R_mat).tolist()
    sensor = VisionSensor.create(
        resolution=[image_size, image_size],
        explicit_handling=True,
        view_angle=fov_deg,
        near_clipping_plane=0.01,
        far_clipping_plane=10.0,
        render_mode=RenderMode.OPENGL3,
    )
    sensor.set_pose(pose)
    return sensor


def capture_orbital_extrinsics(left_sensor, right_sensor):
    """Return 4×4 extrinsics and 3×3 intrinsics for both sensors.
    Must be called while sensors are still alive."""
    return {
        "left_extrinsics":  left_sensor.get_matrix().astype(np.float32),
        "right_extrinsics": right_sensor.get_matrix().astype(np.float32),
        "left_intrinsics":  left_sensor.get_intrinsic_matrix().astype(np.float32),
        "right_intrinsics": right_sensor.get_intrinsic_matrix().astype(np.float32),
    }


# ---------------------------------------------------------------------------
# ObservationConfig factory
# ---------------------------------------------------------------------------

def make_obs_config(image_size):
    """ObservationConfig enabling wrist camera + extrinsics/intrinsics."""
    from pyrep.const import RenderMode
    from rlbench.observation_config import ObservationConfig, CameraConfig

    sz = (image_size, image_size)
    on = CameraConfig(
        rgb=True, depth=True, point_cloud=False, mask=False,
        image_size=sz, render_mode=RenderMode.OPENGL3,
        depth_in_meters=False,
    )
    # Don't disable unused cameras — _set_camera_properties() calls .remove()
    # on any camera with all channels off, permanently deleting it from the scene.
    obs_config = ObservationConfig(
        wrist_camera=on,
        joint_velocities=True,
        joint_positions=True,
        joint_forces=False,
        gripper_open=True,
        gripper_pose=True,
        gripper_joint_positions=False,
        task_low_dim_state=False,
    )
    return obs_config


# ---------------------------------------------------------------------------
# Observation accessors (obs attrs for orbital; perception_data for standard)
# ---------------------------------------------------------------------------

def _get_rgb(obs, key):
    img = getattr(obs, key, None)
    if img is None:
        img = obs.perception_data.get(key)
    if img is None:
        raise ValueError("RGB key '{}' not found in obs".format(key))
    return img


def _get_depth(obs, key):
    d = getattr(obs, key, None)
    if d is None:
        d = obs.perception_data.get(key)
    if d is None:
        raise ValueError("Depth key '{}' not found in obs".format(key))
    return d


# ---------------------------------------------------------------------------
# Raw episode saving
# ---------------------------------------------------------------------------

_UNUSED_OBS_ATTRS = [
    "left_shoulder_rgb",   "left_shoulder_depth",   "left_shoulder_mask",   "left_shoulder_point_cloud",
    "right_shoulder_rgb",  "right_shoulder_depth",  "right_shoulder_mask",  "right_shoulder_point_cloud",
    "overhead_rgb",        "overhead_depth",         "overhead_mask",        "overhead_point_cloud",
    "front_rgb",           "front_depth",            "front_mask",           "front_point_cloud",
    "wrist_mask",          "wrist_point_cloud",
    "joint_forces",        "gripper_matrix",         "gripper_joint_positions",
    "gripper_touch_forces","task_low_dim_state",      "ignore_collisions",
    "mesh_points",
]

_WRIST_MISC_KEYS = {
    "wrist_camera_near", "wrist_camera_far",
    "wrist_camera_extrinsics", "wrist_camera_intrinsics",
}


def _strip_obs(obs):
    """Null out obs fields not needed by the zarr converter, in-place."""
    for attr in _UNUSED_OBS_ATTRS:
        if hasattr(obs, attr):
            setattr(obs, attr, None)
    if hasattr(obs, "misc") and isinstance(obs.misc, dict):
        obs.misc = {k: v for k, v in obs.misc.items() if k in _WRIST_MISC_KEYS}


def save_orbital_episode(demo, ep_path, group, orbital_extrinsics):
    """
    Save a single orbital demo to ep_path/:
      orbital_left_rgb/      orbital_left_depth/
      orbital_right_rgb/     orbital_right_depth/
      wrist_rgb/             wrist_depth/
      low_dim_obs.pkl
      camera_group.txt
      orbital_extrinsics.pkl
    """
    os.makedirs(ep_path, exist_ok=True)

    cam_attrs = [
        ("orbital_left_rgb",  "orbital_left_rgb"),
        ("orbital_right_rgb", "orbital_right_rgb"),
        ("wrist_rgb",         "wrist_rgb"),
    ]

    for attr, folder_name in cam_attrs:
        folder = os.path.join(ep_path, folder_name)
        os.makedirs(folder, exist_ok=True)
        for i, obs in enumerate(demo):
            data = getattr(obs, attr, None)
            if data is None:
                data = obs.perception_data.get(attr)
            if data is None:
                continue
            fname = os.path.join(folder, "{}.png".format(num2id(i)))
            Image.fromarray(data).save(fname)
            if hasattr(obs, attr):
                setattr(obs, attr, None)

    with open(os.path.join(ep_path, "camera_group.txt"), "w") as f:
        f.write(group + "\n")

    with open(os.path.join(ep_path, "orbital_extrinsics.pkl"), "wb") as f:
        pickle.dump(orbital_extrinsics, f)

    for obs in demo:
        _strip_obs(obs)
    with open(os.path.join(ep_path, "low_dim_obs.pkl"), "wb") as f:
        pickle.dump(demo, f)


# ---------------------------------------------------------------------------
# Debug video / zarr (--video-only mode)
# ---------------------------------------------------------------------------

def save_debug_video(demo, video_out, image_size):
    """Render all frames as a 3-panel side-by-side MP4."""
    try:
        import imageio
    except ImportError:
        print("[WARN] imageio not found; skipping video save.")
        return

    frames = []
    for obs in demo:
        panels = [
            _get_rgb(obs, "orbital_left_rgb"),
            _get_rgb(obs, "orbital_right_rgb"),
            _get_rgb(obs, "wrist_rgb"),
        ]
        frames.append(np.concatenate(panels, axis=1))

    out_dir = os.path.dirname(os.path.abspath(video_out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    imageio.mimwrite(video_out, frames, fps=10)
    print("[VIDEO] Saved {} frames to {}".format(len(frames), video_out))

# ---------------------------------------------------------------------------
# Per-episode driver
# ---------------------------------------------------------------------------

def collect_one_episode(task_env, scene, cam_left, cam_right, image_size, fov_deg):
    """
    Reset task, place orbital sensors, collect one demo, remove sensors.
    Returns (demo, orbital_extrinsics, timing_dict) or (None, None, None) on failure.
    """
    print("[STEP] Resetting task environment...")
    t0 = time.perf_counter()
    task_env.reset()
    t_reset = time.perf_counter() - t0
    print("[STEP] Task reset done ({:.2f}s)".format(t_reset))

    print("[STEP] Spawning orbital sensors (image_size={}, fov={:.1f}deg)...".format(
        image_size, fov_deg))
    t1 = time.perf_counter()
    left_sensor  = create_orbital_sensor(cam_left["pos"],  cam_left["R"],  image_size, fov_deg)
    right_sensor = create_orbital_sensor(cam_right["pos"], cam_right["R"], image_size, fov_deg)
    scene.set_orbital_sensors(left_sensor, right_sensor)
    orbital_extrinsics = capture_orbital_extrinsics(left_sensor, right_sensor)
    t_sensors = time.perf_counter() - t1
    print("[STEP] Sensors ready ({:.2f}s)".format(t_sensors))

    demo       = None
    t_demos    = 0.0
    step_timers = {}
    for attempt in range(5):
        try:
            print("[STEP] Running demo (attempt {}/5)...".format(attempt + 1))
            scene.reset_step_timers()
            t2 = time.perf_counter()
            demo, = task_env.get_demos(amount=1, live_demos=True)
            t_demos = time.perf_counter() - t2
            step_timers = scene.get_step_timers()
            n = step_timers["n_steps"]
            t_traj = max(0.0, t_demos
                         - step_timers["obs_wrist"]
                         - step_timers["obs_orbital"]
                         - step_timers["physics"])
            print("[STEP] Demo collected: {} steps in {:.2f}s ({:.2f}s/step)".format(
                len(demo), t_demos, t_demos / max(len(demo), 1)))
            print("[STEP]   per-step breakdown (avg over {} sim steps):".format(n))
            print("[STEP]     traj={:.3f}s  physics={:.3f}s  "
                  "obs_wrist={:.3f}s  obs_orbital={:.3f}s".format(
                t_traj / max(n, 1),
                step_timers["physics"]     / max(n, 1),
                step_timers["obs_wrist"]   / max(n, 1),
                step_timers["obs_orbital"] / max(n, 1),
            ))
            break
        except Exception as e:
            print("[WARN] Attempt {}/5 failed: {}".format(attempt + 1, e))

    print("[STEP] Cleaning up sensors...")
    t3 = time.perf_counter()
    scene.clear_orbital_sensors()
    left_sensor.remove()
    right_sensor.remove()
    t_cleanup = time.perf_counter() - t3
    print("[STEP] Cleanup done ({:.2f}s)".format(t_cleanup))

    if demo is None:
        print("[ERROR] All attempts failed; skipping episode.")
        return None, None, None

    timing = dict(reset=t_reset, sensors=t_sensors, demos=t_demos,
                  cleanup=t_cleanup, **step_timers)
    return demo, orbital_extrinsics, timing
