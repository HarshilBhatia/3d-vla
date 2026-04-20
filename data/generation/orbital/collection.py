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

from data.generation.orbital.constants import DEPTH_SCALE, NCAM, NHAND, num2id


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
    """
    Return a dict with 4×4 extrinsics and 3×3 intrinsics for both sensors.
    Must be called while sensors are still alive.
    """
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
        depth_in_meters=True,
    )
    # Don't disable unused cameras — _set_camera_properties() calls .remove()
    # on any camera with all channels off, permanently deleting it from the scene.
    obs_config = ObservationConfig(
        camera_configs={"wrist": on},
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
# Depth encoding
# ---------------------------------------------------------------------------

def float_array_to_rgb_image(float_array, scale_factor):
    """Encode a float depth map as a 3-channel RGB PNG (RLBench convention)."""
    scaled = np.round(float_array * scale_factor).astype(np.uint32)
    r = (scaled >> 16) & 0xFF
    g = (scaled >> 8)  & 0xFF
    b =  scaled        & 0xFF
    return Image.fromarray(np.stack([r, g, b], axis=-1).astype(np.uint8))


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
        ("orbital_left_rgb",   "orbital_left_rgb",   "rgb"),
        ("orbital_left_depth", "orbital_left_depth", "depth"),
        ("orbital_right_rgb",  "orbital_right_rgb",  "rgb"),
        ("orbital_right_depth","orbital_right_depth","depth"),
        ("wrist_rgb",          "wrist_rgb",           "rgb"),
        ("wrist_depth",        "wrist_depth",         "depth"),
    ]

    for attr, folder_name, kind in cam_attrs:
        folder = os.path.join(ep_path, folder_name)
        os.makedirs(folder, exist_ok=True)
        for i, obs in enumerate(demo):
            data = getattr(obs, attr, None)
            if data is None:
                data = obs.perception_data.get(attr)
            if data is None:
                continue
            fname = os.path.join(folder, "{}.png".format(num2id(i)))
            if kind == "rgb":
                Image.fromarray(data).save(fname)
            else:
                float_array_to_rgb_image(data, DEPTH_SCALE).save(fname)
            if hasattr(obs, attr):
                setattr(obs, attr, None)

    with open(os.path.join(ep_path, "camera_group.txt"), "w") as f:
        f.write(group + "\n")

    with open(os.path.join(ep_path, "orbital_extrinsics.pkl"), "wb") as f:
        pickle.dump(orbital_extrinsics, f)

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


def save_debug_zarr(demo, zarr_path, group, orbital_extrinsics, image_size=256):
    """
    Save a single-episode zarr using the same schema as orbital/to_zarr.py.
    orbital_extrinsics must be captured while sensors are still alive.
    """
    try:
        import zarr
        from numcodecs import Blosc
        from data.processing.rlbench_utils import keypoint_discovery
    except ImportError as e:
        print("[WARN] Could not save debug zarr: {}".format(e))
        return

    compressor = Blosc(cname="lz4", clevel=1, shuffle=Blosc.SHUFFLE)

    key_frames = keypoint_discovery(demo, bimanual=False)
    key_frames.insert(0, 0)
    if len(key_frames) < 2:
        print("[WARN] Not enough keyframes; skipping debug zarr.")
        return

    E_left  = orbital_extrinsics["left_extrinsics"]
    E_right = orbital_extrinsics["right_extrinsics"]
    K_left  = orbital_extrinsics["left_intrinsics"]
    K_right = orbital_extrinsics["right_intrinsics"]
    group_id = int(group[1])  # "G3" → 3

    with zarr.open_group(zarr_path, mode="w") as zf:

        def _create(name, shape, dtype):
            zf.create_dataset(
                name, shape=(0,) + shape,
                chunks=(1,) + shape,
                compressor=compressor, dtype=dtype,
            )

        _create("rgb",                   (NCAM, 3, image_size, image_size), "uint8")
        _create("depth",                 (NCAM, image_size, image_size),    "float16")
        _create("extrinsics",            (NCAM, 4, 4),                      "float16")
        _create("intrinsics",            (NCAM, 3, 3),                      "float16")
        _create("proprioception",        (3, NHAND, 8),                     "float32")
        _create("action",                (1, NHAND, 8),                     "float32")
        _create("proprioception_joints", (1, NHAND, 8),                     "float32")
        _create("action_joints",         (1, NHAND, 8),                     "float32")
        _create("task_id",               (),                                 "uint8")
        _create("variation",             (),                                 "uint8")
        _create("camera_group",          (),                                 "uint8")

        for idx, k in enumerate(key_frames[:-1]):
            obs      = demo[k]
            obs_next = demo[key_frames[idx + 1]]

            rgb = np.stack([
                _get_rgb(obs, "orbital_left_rgb").transpose(2, 0, 1),
                _get_rgb(obs, "orbital_right_rgb").transpose(2, 0, 1),
                _get_rgb(obs, "wrist_rgb").transpose(2, 0, 1),
            ])[np.newaxis]

            depth = np.stack([
                _get_depth(obs, "orbital_left_depth"),
                _get_depth(obs, "orbital_right_depth"),
                _get_depth(obs, "wrist_depth"),
            ]).astype(np.float16)[np.newaxis]

            E_wrist = np.array(obs.misc.get("wrist_camera_extrinsics", np.eye(4)), dtype=np.float32)
            K_wrist = np.array(obs.misc.get("wrist_camera_intrinsics", np.eye(3)), dtype=np.float32)
            extr = np.stack([E_left, E_right, E_wrist]).astype(np.float16)[np.newaxis]
            intr = np.stack([K_left, K_right, K_wrist]).astype(np.float16)[np.newaxis]

            def _eef(o):
                return np.concatenate([o.gripper_pose, [o.gripper_open]]).astype(np.float32)

            s0   = _eef(demo[key_frames[max(0, idx - 2)]])
            s1   = _eef(demo[key_frames[max(0, idx - 1)]])
            s2   = _eef(obs)
            prop = np.stack([s0, s1, s2]).reshape(3, NHAND, 8)[np.newaxis]

            action = _eef(obs_next).reshape(1, NHAND, 8)[np.newaxis]

            def _joints(o):
                return np.concatenate([o.joint_positions, [o.gripper_open]]).astype(np.float32)

            prop_j = _joints(obs).reshape(1, NHAND, 8)[np.newaxis]
            act_j  = _joints(obs_next).reshape(1, NHAND, 8)[np.newaxis]

            zf["rgb"].append(rgb)
            zf["depth"].append(depth)
            zf["extrinsics"].append(extr)
            zf["intrinsics"].append(intr)
            zf["proprioception"].append(prop)
            zf["action"].append(action)
            zf["proprioception_joints"].append(prop_j)
            zf["action_joints"].append(act_j)
            zf["task_id"].append(np.array([0], dtype=np.uint8))
            zf["variation"].append(np.array([0], dtype=np.uint8))
            zf["camera_group"].append(np.array([group_id], dtype=np.uint8))

    print("[ZARR] Saved debug zarr to {}".format(zarr_path))


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
