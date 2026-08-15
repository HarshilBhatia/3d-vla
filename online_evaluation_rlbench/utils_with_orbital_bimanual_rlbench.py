"""Online evaluation utilities for the bimanual orbital camera setup.

Cameras: orbital_left, orbital_right, wrist_left, wrist_right  (NCAM=4, dual_panda)

This is the eval counterpart of the OrbitalPeract2 training dataset. It combines
the two halves that previously lived apart:

  * the bimanual rollout loop (dual-arm Mover, (2, 8) actions, BimanualObservation
    proprio, stored PerAct2 test seeds) from utils_with_bimanual_rlbench, and
  * dynamically spawned orbital VisionSensors plus depth->PCD unprojection from
    utils_with_orbital_rlbench.

Camera geometry is chosen by `spawn_camera_group` (e.g. "G4"): the scene state
comes from the stored test demo, and the two orbital cameras are re-spawned at
that group's poses. Scene state and camera pose are independent, so one stored
test set serves every camera group.

Camera order matches data/processing/orbital_utils.process_episode with
PERACT2_PROFILE: the two orbital cameras first, then the wrists.
"""

import numpy as np
import torch

from data.generation.orbital.collection import (
    load_group_cameras,
    create_orbital_sensor,
    capture_orbital_extrinsics,
)
from data.generation.orbital.scene import OrbitalEnvironment
from utils.depth2cloud.rlbench import RLBenchDepth2Cloud

from rlbench.observation_config import ObservationConfig, CameraConfig
from pyrep.const import RenderMode

from online_evaluation_rlbench.utils_with_bimanual_rlbench import (  # noqa: F401
    Actioner,
    Mover,
    RLBenchEnv as BimanualRLBenchEnv,
    task_file_to_task_class,
)

ORBITAL_CAMERAS = ("orbital_left", "orbital_right")
WRIST_CAMERAS = ("wrist_left", "wrist_right")


class RLBenchEnv(BimanualRLBenchEnv):
    """Bimanual RLBench env whose two orbital cameras are spawned per camera group."""

    def __init__(
        self,
        data_path,
        task_str=None,
        image_size=(256, 256),
        apply_rgb=True,
        apply_depth=True,
        apply_pc=False,
        headless=False,
        apply_cameras=ORBITAL_CAMERAS + WRIST_CAMERAS,
        collision_checking=False,
        cameras_file=None,
        spawn_camera_group=None,
        fov_deg=60.0,
    ):
        if cameras_file is None:
            raise ValueError("cameras_file must be provided for orbital eval")
        if spawn_camera_group is None:
            raise ValueError(
                "spawn_camera_group must be provided for bimanual orbital eval "
                "(the stored test demos carry no camera group of their own)"
            )
        self._cameras_file = cameras_file
        self._spawn_camera_group = spawn_camera_group
        self._fov_deg = float(fov_deg)

        # Camera order must match training: orbital pair first, then wrists.
        if tuple(apply_cameras) != ORBITAL_CAMERAS + WRIST_CAMERAS:
            raise ValueError(
                f"bimanual orbital eval expects cameras "
                f"{ORBITAL_CAMERAS + WRIST_CAMERAS}, got {tuple(apply_cameras)}"
            )
        self._wrist_cameras = WRIST_CAMERAS

        h, w = image_size if isinstance(image_size, (tuple, list)) else (image_size, image_size)
        self._image_h = h

        super().__init__(
            data_path=data_path,
            task_str=task_str,
            image_size=(h, w),
            apply_rgb=True,
            apply_depth=True,
            apply_pc=False,
            headless=headless,
            apply_cameras=apply_cameras,
            collision_checking=collision_checking,
            use_depth2cloud=False,
        )

        # Orbital sensors are raw PyRep VisionSensors, so their depth never passes
        # through RLBench's ObservationConfig; unproject it here, as training does.
        self._depth2cloud = RLBenchDepth2Cloud((h, w))
        self._left_sensor = None
        self._right_sensor = None
        self._orbital_extrinsics = None

    def create_obs_config(self, image_size, apply_rgb, apply_depth, apply_pc, apply_cameras, **kwargs):
        """ObservationConfig for the wrist cameras only.

        The orbital cameras are spawned as VisionSensors after launch and are not
        RLBench cameras, so listing them here would fail sensor lookup.
        """
        used = CameraConfig(
            rgb=True, depth=True, point_cloud=False, mask=False,
            image_size=image_size, render_mode=RenderMode.OPENGL3,
            depth_in_meters=False,
        )
        return ObservationConfig(
            camera_configs={cam: used for cam in self._wrist_cameras},
            joint_forces=False,
            joint_positions=True,
            joint_velocities=True,
            task_low_dim_state=False,
            gripper_touch_forces=False,
            gripper_pose=True,
            gripper_open=True,
            gripper_matrix=False,
            gripper_joint_positions=False,
        )

    def _make_env(self, data_path, headless):
        """OrbitalEnvironment swaps in OrbitalScene, which captures the sensors."""
        return OrbitalEnvironment(
            action_mode=self.action_mode,
            obs_config=self.obs_config,
            headless=headless,
            dataset_root=str(data_path),
            robot_setup="dual_panda",
        )

    def _launch_env(self):
        super()._launch_env()
        self._spawn_sensors(self._spawn_camera_group)

    def _spawn_sensors(self, group):
        """Spawn the group's orbital VisionSensors and capture their calibration."""
        cam_left, cam_right = load_group_cameras(self._cameras_file, group)
        self._left_sensor = create_orbital_sensor(
            cam_left["pos"], cam_left["R"], self._image_h, self._fov_deg
        )
        self._right_sensor = create_orbital_sensor(
            cam_right["pos"], cam_right["R"], self._image_h, self._fov_deg
        )
        self.env._scene.set_orbital_sensors(self._left_sensor, self._right_sensor)
        self._orbital_extrinsics = capture_orbital_extrinsics(
            self._left_sensor, self._right_sensor
        )
        print(f"[orbital bimanual eval] spawned camera group {group}", flush=True)

    def get_rgb_pcd_gripper_from_obs(self, obs):
        """Build (rgb, pcd, gripper) for the 4-camera bimanual orbital setup.

        Returns:
            rgb:     (1, 4, 3, H, W) float32 in [0, 1]
            pcd:     (1, 4, 3, H, W) float32 in world coordinates
            gripper: (1, 16)         float32, (left, right) pose + open
        """
        # --- RGB: orbital sensors write obs attributes, wrists use perception_data ---
        rgbs = [
            torch.tensor(obs.orbital_left_rgb, dtype=torch.float32).permute(2, 0, 1) / 255.0,
            torch.tensor(obs.orbital_right_rgb, dtype=torch.float32).permute(2, 0, 1) / 255.0,
        ] + [
            torch.tensor(obs.perception_data[f"{cam}_rgb"], dtype=torch.float32).permute(2, 0, 1) / 255.0
            for cam in self._wrist_cameras
        ]
        rgb = torch.stack(rgbs).unsqueeze(0)  # (1, 4, 3, H, W)

        # --- Depth: orbital already in metres; wrists are [0, 1] over near..far ---
        depths = [
            torch.tensor(obs.orbital_left_depth, dtype=torch.float32),
            torch.tensor(obs.orbital_right_depth, dtype=torch.float32),
        ]
        for cam in self._wrist_cameras:
            near = obs.misc[f"{cam}_camera_near"]
            far = obs.misc[f"{cam}_camera_far"]
            raw = obs.perception_data[f"{cam}_depth"]
            depths.append(torch.tensor(near + raw * (far - near), dtype=torch.float32))
        depth = torch.stack(depths).unsqueeze(0)  # (1, 4, H, W)

        # --- Calibration ---
        exts = [
            torch.tensor(self._orbital_extrinsics["left_extrinsics"], dtype=torch.float32),
            torch.tensor(self._orbital_extrinsics["right_extrinsics"], dtype=torch.float32),
        ] + [
            torch.tensor(obs.misc[f"{cam}_camera_extrinsics"], dtype=torch.float32)
            for cam in self._wrist_cameras
        ]
        ints = [
            torch.tensor(self._orbital_extrinsics["left_intrinsics"], dtype=torch.float32),
            torch.tensor(self._orbital_extrinsics["right_intrinsics"], dtype=torch.float32),
        ] + [
            torch.tensor(obs.misc[f"{cam}_camera_intrinsics"], dtype=torch.float32)
            for cam in self._wrist_cameras
        ]
        extrinsics = torch.stack(exts).unsqueeze(0)   # (1, 4, 4, 4)
        intrinsics = torch.stack(ints).unsqueeze(0)   # (1, 4, 3, 3)

        pcd = self._depth2cloud(
            depth.cuda(non_blocking=True).to(torch.bfloat16),
            extrinsics.cuda(non_blocking=True).to(torch.bfloat16),
            intrinsics.cuda(non_blocking=True).to(torch.bfloat16),
        ).float().cpu()  # (1, 4, 3, H, W)

        gripper = torch.from_numpy(np.concatenate([
            obs.left.gripper_pose, [obs.left.gripper_open],
            obs.right.gripper_pose, [obs.right.gripper_open],
        ])).float().unsqueeze(0)  # (1, 16)

        return rgb, pcd, gripper

    def _extract_video_frame(self, obs):
        frames = [obs.orbital_left_rgb, obs.orbital_right_rgb] + [
            obs.perception_data[f"{cam}_rgb"] for cam in self._wrist_cameras
        ]
        return np.concatenate(frames, axis=1)
