"""
RLBenchEnv variant for OOD camera evaluation.

Replaces the standard orbital left/right cameras with the two OOD cameras
defined in ood_camera.json, while keeping the wrist camera unchanged.

Camera slots during live eval:
  0 → ood_az30_el40   (obs.orbital_left_rgb / obs.orbital_left_depth)
  1 → ood_az330_el55  (obs.orbital_right_rgb / obs.orbital_right_depth)
  2 → wrist            (obs.wrist_rgb / obs.wrist_depth)

Usage (via evaluate_policy.py):
    python evaluate_policy.py --dataset OrbitalOOD --task close_jar \\
        --checkpoint <ckpt> --headless True
"""

import json
import os
import glob
import random

import open3d  # DON'T DELETE THIS!
import numpy as np
import torch
import torch.nn.functional as F
import einops
from scipy.spatial.transform import Rotation as ScipyR
from tqdm import tqdm

from pyrep.const import RenderMode
from pyrep.objects.vision_sensor import VisionSensor
from rlbench.observation_config import ObservationConfig, CameraConfig
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
from rlbench.backend.exceptions import InvalidActionError
from pyrep.errors import IKError, ConfigurationPathError

from modeling.encoder.text import fetch_tokenizers
from online_evaluation_rlbench.get_stored_demos import get_stored_demos


# Default path relative to repo root — override via --ood-file arg or env var
_DEFAULT_OOD_FILE = os.path.join(
    os.path.dirname(__file__), "..", "ood_camera.json"
)


def task_file_to_task_class(task_file):
    import importlib
    name = task_file.replace(".py", "")
    class_name = "".join([w[0].upper() + w[1:] for w in name.split("_")])
    mod = importlib.import_module("rlbench.tasks.%s" % name)
    mod = importlib.reload(mod)
    return getattr(mod, class_name)


def _R_to_pyrep_quat(R_mat):
    """3×3 cam-to-world rotation → [qx, qy, qz, qw] for PyRep set_pose."""
    return ScipyR.from_matrix(R_mat).as_quat()


def _load_ood_cameras(ood_file):
    """
    Return (left_cam, right_cam) dicts from ood_camera.json.
    cams[0] = ood_az30_el40  → left slot
    cams[1] = ood_az330_el55 → right slot
    """
    with open(ood_file) as f:
        cams = json.load(f)
    left  = {"name": cams[0]["name"], "pos": np.array(cams[0]["pos"]), "R": np.array(cams[0]["R"])}
    right = {"name": cams[1]["name"], "pos": np.array(cams[1]["pos"]), "R": np.array(cams[1]["R"])}
    return left, right


def _create_ood_sensor(pos, R_mat, image_size, fov_deg=60.0):
    """Spawn a VisionSensor at the given OOD pose."""
    quat = _R_to_pyrep_quat(R_mat)
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


class Mover:

    def __init__(self, task, max_tries=1):
        self._task = task
        self._last_action = None
        self._max_tries = max_tries

    def __call__(self, action, collision_checking=False):
        obs = None
        terminate = None
        reward = 0

        target = action.copy()
        if self._last_action is not None:
            action[7] = self._last_action[7].copy()
        for _ in range(self._max_tries):
            action_collision = np.ones(action.shape[0] + 1)
            action_collision[:-1] = action
            if collision_checking:
                action_collision[-1] = 0
            obs, reward, terminate = self._task.step(action_collision)

            pos = obs.gripper_pose[:3]
            dist_pos = np.sqrt(np.square(target[:3] - pos).sum())
            if dist_pos < 5e-3 or reward == 1:
                break

        action = target
        if (
            not reward == 1.0
            and self._last_action is not None
            and action[7] != self._last_action[7]
        ):
            action_collision = np.ones(action.shape[0] + 1)
            action_collision[:-1] = action
            if collision_checking:
                action_collision[-1] = 0
            obs, reward, terminate = self._task.step(action_collision)

        self._last_action = action.copy()
        return obs, reward, terminate


class Actioner:

    def __init__(self, policy=None, backbone='clip'):
        self._policy = policy.cuda()
        self._policy.eval()
        self._instr = None
        self.tokenizer = fetch_tokenizers(backbone)

    def load_episode(self, descriptions):
        instr = [random.choice(descriptions)]
        self._instr = self.tokenizer(instr).cuda(non_blocking=True)

    def predict(self, rgbs, pcds, gripper, prediction_len=1):
        return self._policy(
            None,
            torch.full([1, prediction_len, 1], False).cuda(non_blocking=True),
            rgbs,
            None,
            pcds,
            self._instr,
            gripper[:, :, None, :7],
            run_inference=True
        ).view(1, prediction_len, 8)


class RLBenchEnv:

    def __init__(
        self,
        data_path,
        task_str=None,
        image_size=(256, 256),
        apply_rgb=False,
        apply_depth=False,
        apply_pc=False,
        headless=False,
        apply_cameras=("orbital_left", "orbital_right", "wrist"),
        collision_checking=False,
        ood_file=None,
        fov_deg=60.0,
    ):
        self.data_path = data_path
        self.apply_rgb = apply_rgb
        self.apply_depth = apply_depth
        self.apply_pc = apply_pc
        self.apply_cameras = apply_cameras  # kept for interface compatibility
        self.image_size = image_size
        self._fov_deg = fov_deg
        self._ood_file = ood_file or os.environ.get("OOD_FILE", _DEFAULT_OOD_FILE)

        # Load OOD camera poses
        self._cam_left, self._cam_right = _load_ood_cameras(self._ood_file)
        print("[OOD] left : {} @ {}".format(self._cam_left["name"],  self._cam_left["pos"]))
        print("[OOD] right: {} @ {}".format(self._cam_right["name"], self._cam_right["pos"]))

        # Only enable wrist in ObservationConfig — OOD cameras captured via OrbitalScene
        self.obs_config = self._create_obs_config(image_size, apply_rgb, apply_depth, apply_pc)

        self.action_mode = MoveArmThenGripper(
            arm_action_mode=EndEffectorPoseViaPlanning(collision_checking=collision_checking),
            gripper_action_mode=Discrete()
        )

        from data_generation.orbital_rlbench import OrbitalEnvironment
        self.env = OrbitalEnvironment(
            self.action_mode, str(data_path), self.obs_config,
            headless=headless
        )

        # Sensors are created after env.launch()
        self._ood_left_sensor  = None
        self._ood_right_sensor = None

    def launch(self):
        """Launch the environment and spawn OOD VisionSensors."""
        self.env.launch()
        scene = self.env._scene

        im = self.image_size[0] if isinstance(self.image_size, (list, tuple)) else self.image_size
        self._ood_left_sensor  = _create_ood_sensor(
            self._cam_left["pos"],  self._cam_left["R"],  im, self._fov_deg)
        self._ood_right_sensor = _create_ood_sensor(
            self._cam_right["pos"], self._cam_right["R"], im, self._fov_deg)

        scene.set_orbital_sensors(self._ood_left_sensor, self._ood_right_sensor)
        print("[OOD] VisionSensors spawned and registered with OrbitalScene.")

    def shutdown(self):
        """Remove OOD sensors and shut down the environment."""
        scene = self.env._scene
        scene.clear_orbital_sensors()
        if self._ood_left_sensor is not None:
            self._ood_left_sensor.remove()
        if self._ood_right_sensor is not None:
            self._ood_right_sensor.remove()
        self.env.shutdown()

    def get_obs_action(self, obs):
        """
        Extract RGB, depth, and point cloud from an OOD observation.
        OOD cameras are attached to obs by OrbitalScene.get_observation():
          obs.orbital_left_rgb  / obs.orbital_left_depth
          obs.orbital_right_rgb / obs.orbital_right_depth
        Wrist comes from the standard RLBench obs:
          obs.wrist_rgb / obs.wrist_depth / obs.wrist_point_cloud
        """
        state_dict = {"rgb": [], "depth": [], "pc": []}

        # Slot 0: OOD left
        if self.apply_rgb:
            state_dict["rgb"].append(obs.orbital_left_rgb)
        if self.apply_depth:
            state_dict["depth"].append(obs.orbital_left_depth)
        if self.apply_pc:
            # OOD cameras don't generate point clouds natively; use zeros as placeholder
            h, w = obs.orbital_left_rgb.shape[:2]
            state_dict["pc"].append(np.zeros((h, w, 3), dtype=np.float32))

        # Slot 1: OOD right
        if self.apply_rgb:
            state_dict["rgb"].append(obs.orbital_right_rgb)
        if self.apply_depth:
            state_dict["depth"].append(obs.orbital_right_depth)
        if self.apply_pc:
            h, w = obs.orbital_right_rgb.shape[:2]
            state_dict["pc"].append(np.zeros((h, w, 3), dtype=np.float32))

        # Slot 2: wrist (standard RLBench camera)
        if self.apply_rgb:
            state_dict["rgb"].append(obs.wrist_rgb)
        if self.apply_depth:
            state_dict["depth"].append(obs.wrist_depth)
        if self.apply_pc:
            state_dict["pc"].append(obs.wrist_point_cloud)

        action = np.concatenate([obs.gripper_pose, [obs.gripper_open]])
        return state_dict, torch.from_numpy(action).float()

    def get_rgb_pcd_gripper_from_obs(self, obs):
        state_dict, gripper = self.get_obs_action(obs)
        obs_rgb = [
            torch.tensor(state_dict["rgb"][i]).float().permute(2, 0, 1) / 255.0
            for i in range(len(state_dict["rgb"]))
        ]
        obs_pc = [
            torch.tensor(state_dict["pc"][i]).float().permute(2, 0, 1)
            if len(state_dict["pc"]) > 0 else None
            for i in range(len(state_dict["rgb"]))
        ]
        state = torch.cat(obs_rgb + obs_pc, dim=0)
        ncam = len(self.apply_cameras)
        state = einops.rearrange(
            state,
            "(m n ch) h w -> n m ch h w",
            ch=3,
            n=ncam,
            m=2
        )
        rgb = state[:, 0].unsqueeze(0)   # (1, ncam, 3, H, W)
        pcd = state[:, 1].unsqueeze(0)   # (1, ncam, 3, H, W)
        gripper = gripper.unsqueeze(0)   # (1, 8)
        return rgb, pcd, gripper

    def evaluate_task_on_multiple_variations(
        self,
        task_str,
        max_steps,
        actioner,
        max_tries=1,
        prediction_len=1,
        num_history=1
    ):
        self.launch()
        task_type = task_file_to_task_class(task_str)
        task = self.env.get_task(task_type)
        task_variations = glob.glob(
            os.path.join(self.data_path, task_str, "variation*")
        )
        task_variations = [
            int(n.split('/')[-1].replace('variation', ''))
            for n in task_variations
        ]

        var_success_rates = {}
        var_num_valid_demos = {}

        for variation in tqdm(task_variations):
            task.set_variation(variation)
            success_rate, valid, num_valid_demos = self._evaluate_task_on_one_variation(
                task_str=task_str,
                task=task,
                max_steps=max_steps,
                variation=variation,
                actioner=actioner,
                max_tries=max_tries,
                prediction_len=prediction_len,
                num_history=num_history
            )
            if valid:
                var_success_rates[variation] = success_rate
                var_num_valid_demos[variation] = num_valid_demos

        self.shutdown()

        var_success_rates["mean"] = (
            sum(var_success_rates.values()) /
            sum(var_num_valid_demos.values())
        ) if var_num_valid_demos else 0.0

        return var_success_rates

    @torch.no_grad()
    def _evaluate_task_on_one_variation(
        self,
        task_str,
        task,
        max_steps,
        variation,
        actioner,
        max_tries=1,
        prediction_len=1,
        num_history=1
    ):
        success_rate = 0
        total_reward = 0
        var_demos = get_stored_demos(
            amount=-1,
            dataset_root=self.data_path,
            variation_number=variation,
            task_name=task_str,
            random_selection=False,
            from_episode_number=0
        )

        for demo_id, demo in enumerate(var_demos):
            grippers = torch.Tensor([]).cuda(non_blocking=True)
            descriptions, obs = task.reset_to_demo(demo)
            actioner.load_episode(descriptions)

            move = Mover(task, max_tries=max_tries)
            max_reward = 0.0

            for step_id in range(max_steps):
                rgb, pcd, gripper = self.get_rgb_pcd_gripper_from_obs(obs)
                rgbs_input = rgb.cuda(non_blocking=True)
                pcds_input = pcd.cuda(non_blocking=True)
                gripper = gripper.cuda(non_blocking=True)
                grippers = torch.cat([grippers, gripper.unsqueeze(1)], 1)

                gripper_input = grippers[:, -num_history:]
                npad = num_history - gripper_input.shape[1]
                gripper_input = F.pad(
                    gripper_input, (0, 0, npad, 0), mode='replicate'
                )

                output = actioner.predict(
                    rgbs_input,
                    pcds_input,
                    gripper_input,
                    prediction_len=prediction_len
                )

                try:
                    actions = output[-1].cpu().numpy()
                    actions[:, -1] = actions[:, -1].round()
                    for action in actions:
                        obs, reward, _ = move(action, collision_checking=False)
                    max_reward = max(max_reward, reward)
                    if reward == 1:
                        success_rate += 1
                        break
                except (IKError, ConfigurationPathError, InvalidActionError) as e:
                    print(task_str, demo, step_id, success_rate, e)
                    reward = 0

            total_reward += max_reward
            print(
                task_str, "Variation", variation, "Demo", demo_id,
                "Reward", f"{reward:.2f}", "max_reward", f"{max_reward:.2f}",
                f"SR: {success_rate}/{demo_id + 1}",
                f"SR: {total_reward:.2f}/{demo_id + 1}",
                "# valid demos", demo_id + 1,
            )

        valid = len(var_demos) > 0
        return success_rate, valid, len(var_demos)

    def _create_obs_config(self, image_size, apply_rgb, apply_depth, apply_pc):
        """
        Only enable wrist camera in ObservationConfig.
        OOD cameras are captured via OrbitalScene outside the standard pipeline.
        """
        unused_cams = CameraConfig()
        unused_cams.set_all(False)

        wrist_cam = CameraConfig(
            rgb=apply_rgb,
            point_cloud=apply_pc,
            depth=apply_depth,
            mask=False,
            image_size=image_size,
            render_mode=RenderMode.OPENGL,
        )

        obs_config = ObservationConfig(
            front_camera=unused_cams,
            left_shoulder_camera=unused_cams,
            right_shoulder_camera=unused_cams,
            wrist_camera=wrist_cam,
            overhead_camera=unused_cams,
            joint_forces=False,
            joint_positions=False,
            joint_velocities=True,
            task_low_dim_state=False,
            gripper_touch_forces=False,
            gripper_pose=True,
            gripper_open=True,
            gripper_matrix=True,
            gripper_joint_positions=True
        )
        return obs_config
