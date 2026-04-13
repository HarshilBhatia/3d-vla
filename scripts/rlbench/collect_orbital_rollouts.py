"""
Collect RLBench PerAct rollouts using orbital camera groups.

Each episode records 4 cameras:
  [orbital_left, orbital_right, over_shoulder_left, over_shoulder_right]

Normal mode  → saves raw episodes to --save-path/{task}/{group}/episode_{N}/
Video mode   → saves a single episode as MP4 and a single-episode zarr
               for pipeline validation.

Example (headless):
    xvfb-run -a python scripts/rlbench/collect_orbital_rollouts.py \\
        --task close_jar --group G1 --n-episodes 30 \\
        --save-path data/orbital_rollouts \\
        --cameras-file orbital_cameras_grouped.json

    # Debug / video mode:
    xvfb-run -a python scripts/rlbench/collect_orbital_rollouts.py \\
        --task close_jar --group G1 --video-only \\
        --video-out debug_videos/close_jar_G1.mp4 \\
        --cameras-file orbital_cameras_grouped.json
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


# ---------------------------------------------------------------------------
# Geometry helpers (mirrors visualize_cameras_rerun.py)
# ---------------------------------------------------------------------------

def R_to_pyrep_quat(R_mat):
    """3×3 cam-to-world rotation → [qx, qy, qz, qw] for PyRep set_pose."""
    return ScipyR.from_matrix(R_mat).as_quat()  # scipy returns xyzw


def load_group_cameras(cameras_file, group):
    """Return the left/right camera dicts for `group` (e.g. 'G1')."""
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


def save_orbital_episode(demo, ep_path, group, orbital_extrinsics):
    """
    Save a single orbital demo to ep_path/:
      orbital_left_rgb/      orbital_left_depth/
      orbital_right_rgb/     orbital_right_depth/
      over_shoulder_left_rgb/  over_shoulder_left_depth/
      over_shoulder_right_rgb/ over_shoulder_right_depth/
      low_dim_obs.pkl
      camera_group.txt
      orbital_extrinsics.pkl   (E and K for both orbital cameras)
    """
    os.makedirs(ep_path, exist_ok=True)

    cam_attrs = [
        ("orbital_left_rgb",   "orbital_left_rgb",   "rgb"),
        ("orbital_left_depth", "orbital_left_depth", "depth"),
        ("orbital_right_rgb",  "orbital_right_rgb",  "rgb"),
        ("orbital_right_depth","orbital_right_depth","depth"),
        ("wrist_rgb",          "wrist_rgb",          "rgb"),
        ("wrist_depth",        "wrist_depth",        "depth"),
    ]

    for attr, folder_name, kind in cam_attrs:
        folder = os.path.join(ep_path, folder_name)
        os.makedirs(folder, exist_ok=True)
        for i, obs in enumerate(demo):
            # Orbital data is a direct attribute; shoulder data is in perception_data
            data = getattr(obs, attr, None)
            if data is None:
                data = obs.perception_data.get(attr)
            if data is None:
                continue
            fname = os.path.join(folder, "{}.png".format(_num2id(i)))
            if kind == "rgb":
                Image.fromarray(data).save(fname)
            else:
                float_array_to_rgb_image(data, DEPTH_SCALE).save(fname)
            # Clear to free memory (only for direct attrs; perception_data cleared separately)
            if hasattr(obs, attr):
                setattr(obs, attr, None)

    # Save group tag
    with open(os.path.join(ep_path, "camera_group.txt"), "w") as f:
        f.write(group + "\n")

    # Save orbital camera extrinsics / intrinsics (captured before sensor removal)
    with open(os.path.join(ep_path, "orbital_extrinsics.pkl"), "wb") as f:
        pickle.dump(orbital_extrinsics, f)

    # Save low-dim pickle (no large arrays remain after above)
    with open(os.path.join(ep_path, "low_dim_obs.pkl"), "wb") as f:
        pickle.dump(demo, f)


# ---------------------------------------------------------------------------
# Video / zarr helpers (used in --video-only mode)
# ---------------------------------------------------------------------------

def _get_rgb(obs, key, image_size):
    """Get RGB from obs: orbital cameras are direct attrs; standard cams are in perception_data."""
    img = getattr(obs, key, None)
    if img is None:
        img = obs.perception_data.get(key)
    if img is None:
        raise ValueError('img is None')
    return img


def _get_depth(obs, key, image_size):
    """Get depth from obs: orbital cameras are direct attrs; standard cams are in perception_data."""
    d = getattr(obs, key, None)
    if d is None:
        d = obs.perception_data.get(key)
    if d is None:
        raise ValueError('img is None')

    return d


def save_debug_video(demo, video_out, image_size):
    """Render all frames as a side-by-side 4-camera MP4."""
    try:
        import imageio
    except ImportError:
        print("[WARN] imageio not found; skipping video save.")
        return

    frames = []
    for obs in demo:
        panels = [
            _get_rgb(obs, "orbital_left_rgb",  image_size),
            _get_rgb(obs, "orbital_right_rgb",  image_size),
            _get_rgb(obs, "wrist_rgb",          image_size),
        ]
        frames.append(np.concatenate(panels, axis=1))  # (H, 3*W, 3)

    out_dir = os.path.dirname(os.path.abspath(video_out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    imageio.mimwrite(video_out, frames, fps=10)
    print("[VIDEO] Saved {} frames to {}".format(len(frames), video_out))


def save_debug_zarr(demo, zarr_path, group, orbital_extrinsics, image_size=256):
    """
    Save a single-episode zarr using the same schema as orbital_to_zarr.py.
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

            # RGB — orbital attrs; wrist from perception_data
            rgb_list = [
                _get_rgb(obs, "orbital_left_rgb",  image_size).transpose(2, 0, 1),
                _get_rgb(obs, "orbital_right_rgb",  image_size).transpose(2, 0, 1),
                _get_rgb(obs, "wrist_rgb",          image_size).transpose(2, 0, 1),
            ]
            rgb = np.stack(rgb_list)[np.newaxis]

            # Depth
            depth_list = [
                _get_depth(obs, "orbital_left_depth",  image_size),
                _get_depth(obs, "orbital_right_depth",  image_size),
                _get_depth(obs, "wrist_depth",          image_size),
            ]
            depth = np.stack(depth_list).astype(np.float16)[np.newaxis]

            # Extrinsics / intrinsics
            E_wrist = np.array(obs.misc.get("wrist_camera_extrinsics", np.eye(4)), dtype=np.float32)
            K_wrist = np.array(obs.misc.get("wrist_camera_intrinsics", np.eye(3)), dtype=np.float32)
            extr = np.stack([E_left, E_right, E_wrist]).astype(np.float16)[np.newaxis]
            intr = np.stack([K_left, K_right, K_wrist]).astype(np.float16)[np.newaxis]

            # Proprioception
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
            zf["camera_group"].append(np.array([group_id], dtype=np.uint8))

    print("[ZARR] Saved debug zarr to {}".format(zarr_path))


# ---------------------------------------------------------------------------
# Sensor creation
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
    """ObservationConfig enabling wrist camera + extrinsics/intrinsics."""
    from pyrep.const import RenderMode
    from rlbench.observation_config import ObservationConfig, CameraConfig

    sz = (image_size, image_size)
    on = CameraConfig(
        rgb=True, depth=True, point_cloud=False, mask=False,
        image_size=sz, render_mode=RenderMode.OPENGL3,
        depth_in_meters=True,
    )

    # Don't pass rgb=False/depth=False/point_cloud=False for unused cameras —
    # _set_camera_properties() calls .remove() on them, permanently deleting them
    # from the scene. OrbitalScene.__init__ then can't find them on the second
    # Scene.__init__ call. Just leave unused cameras at their defaults.
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
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Collect RLBench orbital rollouts."
    )
    p.add_argument("--task",         required=True,
                   help="RLBench task name (e.g. close_jar)")
    p.add_argument("--groups",       nargs="+", required=True,
                   help="One or more camera groups (e.g. G1 G2 G3). "
                        "All groups are collected in a single CoppeliaSim session.")
    p.add_argument("--n-episodes",   type=int, default=30,
                   help="Number of episodes to collect per group (default: 30)")
    p.add_argument("--save-path",    default="data/orbital_rollouts",
                   help="Root directory for raw episode output")
    p.add_argument("--image-size",   type=int, default=256)
    p.add_argument("--fov-deg",      type=float, default=60.0,
                   help="FOV for orbital cameras (degrees)")
    p.add_argument("--cameras-file", default="orbital_cameras_grouped.json",
                   help="Path to orbital_cameras_grouped.json")
    p.add_argument("--video-only",   action="store_true",
                   help="Debug: collect 1 episode per group, save MP4 + zarr")
    p.add_argument("--video-dir",    default="debug_videos",
                   help="Output directory for debug MP4s (used with --video-only)")
    p.add_argument("--variation",    type=int, default=0,
                   help="Task variation index (default: 0)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
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

    print("[STEP] Spawning orbital sensors (image_size={}, fov={:.1f}deg)...".format(image_size, fov_deg))
    t1 = time.perf_counter()
    left_sensor  = create_orbital_sensor(cam_left["pos"],  cam_left["R"],  image_size, fov_deg)
    right_sensor = create_orbital_sensor(cam_right["pos"], cam_right["R"], image_size, fov_deg)
    scene.set_orbital_sensors(left_sensor, right_sensor)
    orbital_extrinsics = capture_orbital_extrinsics(left_sensor, right_sensor)
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
            t_traj = max(0.0, t_demos - step_timers["obs_wrist"]
                                       - step_timers["obs_orbital"]
                                       - step_timers["physics"])
            print("[STEP] Demo collected: {} steps in {:.2f}s ({:.2f}s/step)".format(
                len(demo), t_demos, t_demos / max(len(demo), 1)))
            print("[STEP]   per-step breakdown (avg over {} sim steps):".format(n))
            print("[STEP]     traj={:.3f}s  physics={:.3f}s  obs_wrist={:.3f}s  obs_orbital={:.3f}s".format(
                t_traj   / max(n, 1),
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

    timing = dict(reset=t_reset, sensors=t_sensors, demos=t_demos, cleanup=t_cleanup,
                  **step_timers)
    return demo, orbital_extrinsics, timing


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
    task_env.set_variation(args.variation)
    scene      = env._scene  # OrbitalScene instance

    print("[INFO] task={} groups={} mode={}".format(
        args.task, args.groups, "video" if args.video_only else "collect"))

    # ── Loop over groups (all within one CoppeliaSim session) ─────────────────
    for group in args.groups:
        cam_left, cam_right = load_group_cameras(args.cameras_file, group)

        if args.video_only:
            # One episode per group → MP4 + zarr
            video_out = os.path.join(
                args.video_dir, "{}_{}.mp4".format(args.task, group))
            zarr_path = video_out + ".zarr"
            if os.path.exists(zarr_path):
                print("[SKIP] {}/{} already exists.".format(args.task, group))
                continue

            demo, orbital_extrinsics, timing = collect_one_episode(
                task_env, scene, cam_left, cam_right,
                args.image_size, args.fov_deg,
            )
            if demo is None:
                continue
            print("[TIME] reset={:.2f}s sensors={:.2f}s demos={:.2f}s cleanup={:.2f}s steps={}".format(
                timing["reset"], timing["sensors"], timing["demos"], timing["cleanup"], len(demo)))
            t_vid0 = time.perf_counter()
            save_debug_video(demo, video_out, args.image_size)
            t_video = time.perf_counter() - t_vid0
            print("[TIME] video={:.2f}s ({} frames → {})".format(t_video, len(demo), video_out))
            t_zarr0 = time.perf_counter()
            save_debug_zarr(demo, zarr_path, group,
                            orbital_extrinsics=orbital_extrinsics,
                            image_size=args.image_size)
            t_zarr = time.perf_counter() - t_zarr0
            print("[TIME] zarr={:.2f}s → {}".format(t_zarr, zarr_path))
            t_total = timing["reset"] + timing["sensors"] + timing["demos"] + timing["cleanup"] + t_video + t_zarr
            print("[TIME] total={:.2f}s (collect={:.2f}s save={:.2f}s)".format(
                t_total,
                timing["reset"] + timing["sensors"] + timing["demos"] + timing["cleanup"],
                t_video + t_zarr))

        else:
            # Normal collection: n_episodes per group
            base_path = os.path.join(args.save_path, args.task, group)

            # Resume: count already-saved episodes
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
                        args.task, group, ep_start))
                    continue
                if ep_start > 0:
                    print("[RESUME] {}/{} from episode {}.".format(
                        args.task, group, ep_start))

            ep_times = []
            for ep_idx in range(ep_start, args.n_episodes):
                print("[INFO] {}/{} episode {}/{}".format(
                    args.task, group, ep_idx + 1, args.n_episodes))
                demo, orbital_extrinsics, timing = collect_one_episode(
                    task_env, scene, cam_left, cam_right,
                    args.image_size, args.fov_deg,
                )
                if demo is None:
                    continue
                t_collect = timing["reset"] + timing["sensors"] + timing["demos"] + timing["cleanup"]
                print("[TIME]  reset={:.2f}s sensors={:.2f}s demos={:.2f}s cleanup={:.2f}s steps={}".format(
                    timing["reset"], timing["sensors"], timing["demos"], timing["cleanup"], len(demo)))
                ep_path = os.path.join(base_path, "episode_{}".format(ep_idx))
                print("[STEP] Saving episode to {}...".format(ep_path))
                t1 = time.perf_counter()
                save_orbital_episode(demo, ep_path, group, orbital_extrinsics)
                t_save = time.perf_counter() - t1
                print("[TIME]  save={:.2f}s → {}".format(t_save, ep_path))
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
