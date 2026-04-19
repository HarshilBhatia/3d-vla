"""
Collect RLBench PerAct rollouts using OOD cameras from ood_camera.json.

The two OOD cameras replace the orbital left/right pair:
  slot 0 → ood_az30_el40   (left)
  slot 1 → ood_az330_el55  (right)
  slot 2 → wrist            (unchanged)

Normal mode  → saves raw episodes to --save-path/{task}/OOD/episode_{N}/
Video mode   → saves a single episode as MP4 + single-episode zarr
               for pipeline validation.

Example (headless):
    xvfb-run -a python scripts/rlbench/collect_ood_rollouts.py \\
        --task close_jar --n-episodes 5 \\
        --save-path data/ood_rollouts \\
        --ood-file ood_camera.json

    # Debug / video mode:
    xvfb-run -a python scripts/rlbench/collect_ood_rollouts.py \\
        --task close_jar --video-only \\
        --video-dir debug_videos \\
        --ood-file ood_camera.json
"""

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as ScipyR

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

DEPTH_SCALE = 2 ** 24 - 1
GROUP_NAME = "OOD"


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def R_to_pyrep_quat(R_mat):
    """3×3 cam-to-world rotation → [qx, qy, qz, qw] for PyRep set_pose."""
    return ScipyR.from_matrix(R_mat).as_quat()  # scipy returns xyzw


def load_ood_cameras(ood_file):
    """
    Return (left_cam, right_cam) dicts from ood_camera.json.
    cams[0] = ood_az30_el40  → left slot
    cams[1] = ood_az330_el55 → right slot
    """
    with open(ood_file) as f:
        cams = json.load(f)
    if len(cams) < 2:
        raise ValueError("ood_camera.json must have at least 2 entries")
    left  = {"name": cams[0]["name"], "pos": np.array(cams[0]["pos"]), "R": np.array(cams[0]["R"])}
    right = {"name": cams[1]["name"], "pos": np.array(cams[1]["pos"]), "R": np.array(cams[1]["R"])}
    return left, right


# ---------------------------------------------------------------------------
# Saving helpers
# ---------------------------------------------------------------------------

def _num2id(i):
    return str(i).zfill(4)


def float_array_to_rgb_image(float_array, scale_factor):
    """Encode a float depth map as a 3-channel RGB PNG (RLBench convention)."""
    scaled = np.round(float_array * scale_factor).astype(np.uint32)
    r = (scaled >> 16) & 0xFF
    g = (scaled >> 8)  & 0xFF
    b = scaled & 0xFF
    return Image.fromarray(
        np.stack([r, g, b], axis=-1).astype(np.uint8)
    )


def save_ood_episode(demo, ep_path, orbital_extrinsics, variation=0):
    """
    Save a single OOD demo to ep_path/:
      ood_left_rgb/      ood_left_depth/
      ood_right_rgb/     ood_right_depth/
      wrist_rgb/         wrist_depth/
      low_dim_obs.pkl
      camera_group.txt   (writes "OOD")
      variation.txt
      ood_extrinsics.pkl (E and K for both OOD cameras)
    """
    os.makedirs(ep_path, exist_ok=True)

    cam_attrs = [
        ("orbital_left_rgb",    "ood_left_rgb",    "rgb"),
        ("orbital_left_depth",  "ood_left_depth",  "depth"),
        ("orbital_right_rgb",   "ood_right_rgb",   "rgb"),
        ("orbital_right_depth", "ood_right_depth", "depth"),
        ("wrist_rgb",           "wrist_rgb",        "rgb"),
        ("wrist_depth",         "wrist_depth",      "depth"),
    ]

    for attr, folder_name, kind in cam_attrs:
        folder = os.path.join(ep_path, folder_name)
        os.makedirs(folder, exist_ok=True)
        for i, obs in enumerate(demo):
            data = getattr(obs, attr, None)
            if data is None:
                data = obs.perception_data.get(attr) if hasattr(obs, "perception_data") else None
            if data is None:
                continue
            fname = os.path.join(folder, "{}.png".format(_num2id(i)))
            if kind == "rgb":
                Image.fromarray(data).save(fname)
            else:
                float_array_to_rgb_image(data, DEPTH_SCALE).save(fname)
            if hasattr(obs, attr):
                setattr(obs, attr, None)

    # Group tag
    with open(os.path.join(ep_path, "camera_group.txt"), "w") as f:
        f.write(GROUP_NAME + "\n")

    # Variation
    with open(os.path.join(ep_path, "variation.txt"), "w") as f:
        f.write(str(variation) + "\n")

    # OOD camera extrinsics / intrinsics
    with open(os.path.join(ep_path, "ood_extrinsics.pkl"), "wb") as f:
        pickle.dump(orbital_extrinsics, f)

    # Low-dim pickle
    with open(os.path.join(ep_path, "low_dim_obs.pkl"), "wb") as f:
        pickle.dump(demo, f)


# ---------------------------------------------------------------------------
# Video / zarr helpers (used in --video-only mode)
# ---------------------------------------------------------------------------

def _get_rgb(obs, key):
    img = getattr(obs, key, None)
    if img is None and hasattr(obs, "perception_data"):
        img = obs.perception_data.get(key)
    if img is None:
        raise ValueError("img is None for key {}".format(key))
    return img


def _get_depth(obs, key):
    d = getattr(obs, key, None)
    if d is None and hasattr(obs, "perception_data"):
        d = obs.perception_data.get(key)
    if d is None:
        raise ValueError("depth is None for key {}".format(key))
    return d


def save_debug_video(demo, video_out, image_size):
    """Render all frames as a side-by-side 3-camera MP4."""
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


def save_debug_zarr(demo, zarr_path, orbital_extrinsics, image_size=256):
    """
    Save a single-episode zarr using the same schema as ood_to_zarr.py.
    Used to validate the zarr pipeline before large-scale collection.
    orbital_extrinsics must be captured while sensors are still alive.
    """
    try:
        import zarr
        from numcodecs import Blosc
        from data_processing.rlbench_utils import keypoint_discovery
    except ImportError as e:
        print("[WARN] Could not save debug zarr: {}".format(e))
        return

    NCAM = 3
    NHAND = 1

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

            rgb_list = [
                _get_rgb(obs, "orbital_left_rgb").transpose(2, 0, 1),
                _get_rgb(obs, "orbital_right_rgb").transpose(2, 0, 1),
                _get_rgb(obs, "wrist_rgb").transpose(2, 0, 1),
            ]
            rgb = np.stack(rgb_list)[np.newaxis]

            depth_list = [
                _get_depth(obs, "orbital_left_depth"),
                _get_depth(obs, "orbital_right_depth"),
                _get_depth(obs, "wrist_depth"),
            ]
            depth = np.stack(depth_list).astype(np.float16)[np.newaxis]

            E_wrist = np.array(obs.misc.get("wrist_camera_extrinsics", np.eye(4)), dtype=np.float32)
            K_wrist = np.array(obs.misc.get("wrist_camera_intrinsics", np.eye(3)), dtype=np.float32)
            extr = np.stack([E_left, E_right, E_wrist]).astype(np.float16)[np.newaxis]
            intr = np.stack([K_left, K_right, K_wrist]).astype(np.float16)[np.newaxis]

            def _eef(o):
                return np.concatenate([o.gripper_pose, [o.gripper_open]]).astype(np.float32)

            s0 = _eef(demo[key_frames[max(0, idx - 2)]])
            s1 = _eef(demo[key_frames[max(0, idx - 1)]])
            s2 = _eef(obs)
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
            zf["camera_group"].append(np.array([0], dtype=np.uint8))  # OOD → 0

    print("[ZARR] Saved debug zarr to {}".format(zarr_path))


# ---------------------------------------------------------------------------
# Sensor creation (mirrors collect_orbital_rollouts.py)
# ---------------------------------------------------------------------------

def create_orbital_sensor(pos, R_mat, image_size, fov_deg):
    """Spawn a VisionSensor at the given pose and return it."""
    from pyrep.const import RenderMode
    from pyrep.objects.vision_sensor import VisionSensor

    quat = R_to_pyrep_quat(R_mat)
    pose = pos.tolist() + quat.tolist()

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
# Observation config
# ---------------------------------------------------------------------------

def make_obs_config(image_size):
    """ObservationConfig enabling wrist camera + gripper state."""
    from pyrep.const import RenderMode
    from rlbench.observation_config import ObservationConfig, CameraConfig

    sz = (image_size, image_size)
    on = CameraConfig(
        rgb=True, depth=True, point_cloud=False, mask=False,
        image_size=sz, render_mode=RenderMode.OPENGL3,
        depth_in_meters=True,
    )

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
# Episode collection
# ---------------------------------------------------------------------------

def collect_one_episode(task_env, scene, cam_left, cam_right, image_size, fov_deg):
    """
    Reset task, place OOD sensors, collect one demo, remove sensors.
    Returns (demo, ood_extrinsics, timing_dict) or (None, None, None) on failure.
    """
    print("[STEP] Resetting task environment...")
    t0 = time.perf_counter()
    task_env.reset()
    t_reset = time.perf_counter() - t0
    print("[STEP] Task reset done ({:.2f}s)".format(t_reset))

    print("[STEP] Spawning OOD sensors (image_size={}, fov={:.1f}deg)...".format(image_size, fov_deg))
    t1 = time.perf_counter()
    left_sensor  = create_orbital_sensor(cam_left["pos"],  cam_left["R"],  image_size, fov_deg)
    right_sensor = create_orbital_sensor(cam_right["pos"], cam_right["R"], image_size, fov_deg)
    scene.set_orbital_sensors(left_sensor, right_sensor)
    ood_extrinsics = capture_orbital_extrinsics(left_sensor, right_sensor)
    t_sensors = time.perf_counter() - t1
    print("[STEP] Sensors ready ({:.2f}s)".format(t_sensors))

    demo = None
    t_demos = 0.0
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
            print("[STEP] Demo collected: {} steps in {:.2f}s ({:.2f}s/step)".format(
                len(demo), t_demos, t_demos / max(len(demo), 1)))
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

    timing = dict(reset=t_reset, sensors=t_sensors, demos=t_demos, cleanup=t_cleanup,
                  **step_timers)
    return demo, ood_extrinsics, timing


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Collect RLBench OOD rollouts using cameras from ood_camera.json."
    )
    p.add_argument("--task",        required=True,
                   help="RLBench task name (e.g. close_jar)")
    p.add_argument("--n-episodes",  type=int, default=5,
                   help="Number of episodes to collect (default: 5)")
    p.add_argument("--save-path",   default="data/ood_rollouts",
                   help="Root directory for raw episode output")
    p.add_argument("--ood-file",    default="ood_camera.json",
                   help="Path to ood_camera.json")
    p.add_argument("--image-size",  type=int, default=256)
    p.add_argument("--fov-deg",     type=float, default=60.0,
                   help="FOV for OOD cameras (degrees)")
    p.add_argument("--video-only",  action="store_true",
                   help="Debug: collect 1 episode, save MP4 + zarr")
    p.add_argument("--video-dir",   default="debug_videos",
                   help="Output directory for debug MP4s (used with --video-only)")
    p.add_argument("--variation",   type=int, default=None,
                   help="Fix a specific variation index. Default: sample randomly.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    try:
        from rlbench.action_modes.action_mode import MoveArmThenGripper
        from rlbench.action_modes.arm_action_modes import JointVelocity
        from rlbench.action_modes.gripper_action_modes import Discrete
        from rlbench.backend.utils import task_file_to_task_class
    except ImportError as e:
        sys.exit("[ERROR] RLBench import failed: {}\n"
                 "Set COPPELIASIM_ROOT etc. first.".format(e))

    from data_generation.orbital_rlbench import OrbitalEnvironment

    cam_left, cam_right = load_ood_cameras(args.ood_file)
    print("[INFO] OOD cameras loaded:")
    print("  left : {} @ {}".format(cam_left["name"],  cam_left["pos"]))
    print("  right: {} @ {}".format(cam_right["name"], cam_right["pos"]))

    obs_config  = make_obs_config(args.image_size)
    action_mode = MoveArmThenGripper(JointVelocity(), Discrete())

    env = OrbitalEnvironment(
        action_mode=action_mode,
        obs_config=obs_config,
        headless=True,
        robot_setup="panda",
    )
    env.launch()

    task_class = task_file_to_task_class(args.task)
    task_env   = env.get_task(task_class)
    scene      = env._scene  # OrbitalScene instance

    print("[INFO] task={} mode={}".format(
        args.task, "video" if args.video_only else "collect"))

    if args.video_only:
        video_out = os.path.join(
            args.video_dir, "{}_OOD.mp4".format(args.task))
        zarr_path = video_out + ".zarr"

        demo, ood_extrinsics, timing = collect_one_episode(
            task_env, scene, cam_left, cam_right,
            args.image_size, args.fov_deg,
        )
        if demo is not None:
            print("[TIME] reset={:.2f}s sensors={:.2f}s demos={:.2f}s cleanup={:.2f}s steps={}".format(
                timing["reset"], timing["sensors"], timing["demos"], timing["cleanup"], len(demo)))
            save_debug_video(demo, video_out, args.image_size)
            save_debug_zarr(demo, zarr_path,
                            orbital_extrinsics=ood_extrinsics,
                            image_size=args.image_size)
            print("[DONE] video={} zarr={}".format(video_out, zarr_path))
    else:
        base_path = os.path.join(args.save_path, args.task, GROUP_NAME)

        # Resume from existing episodes
        ep_start = 0
        if os.path.exists(base_path):
            existing = [
                d for d in os.listdir(base_path)
                if d.startswith("episode_") and
                   os.path.isdir(os.path.join(base_path, d))
            ]
            ep_start = len(existing)
            if ep_start >= args.n_episodes:
                print("[SKIP] {}/{} already has {} episodes.".format(
                    args.task, GROUP_NAME, ep_start))
                env.shutdown()
                return
            if ep_start > 0:
                print("[RESUME] {}/{} from episode {}.".format(
                    args.task, GROUP_NAME, ep_start))

        ep_times = []
        for ep_idx in range(ep_start, args.n_episodes):
            print("[INFO] {}/{} episode {}/{}".format(
                args.task, GROUP_NAME, ep_idx + 1, args.n_episodes))
            if args.variation is None:
                variation = task_env.sample_variation()
            else:
                task_env.set_variation(args.variation)
                variation = args.variation
            print("[INFO] variation={}".format(variation))

            demo, ood_extrinsics, timing = collect_one_episode(
                task_env, scene, cam_left, cam_right,
                args.image_size, args.fov_deg,
            )
            if demo is None:
                continue

            t_collect = timing["reset"] + timing["sensors"] + timing["demos"] + timing["cleanup"]
            ep_path = os.path.join(base_path, "episode_{}".format(ep_idx))
            print("[STEP] Saving episode to {}...".format(ep_path))
            t1 = time.perf_counter()
            save_ood_episode(demo, ep_path, ood_extrinsics, variation=variation)
            t_save = time.perf_counter() - t1
            t_total = t_collect + t_save
            ep_times.append(t_total)
            avg = sum(ep_times) / len(ep_times)
            remaining = (args.n_episodes - ep_idx - 1) * avg
            print("[SAVED] {} | collect={:.1f}s save={:.1f}s total={:.1f}s avg={:.1f}s eta={:.1f}s".format(
                ep_path, t_collect, t_save, t_total, avg, remaining))

    env.shutdown()
    print("[DONE]")


if __name__ == "__main__":
    main()
