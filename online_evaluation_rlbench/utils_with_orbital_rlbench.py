"""Online evaluation utilities for the orbital + wrist camera setup.

Cameras: orbital_left, orbital_right, wrist  (NCAM=3, single-arm Panda)

Each task was trained on 3 camera groups (from task_group_mapping.json).
Eval iterates over all 3 groups, re-spawning VisionSensors per group, and
averages success rates across (group × variation) weighted by demo count.

Orbital cameras are dynamically-spawned PyRep VisionSensors — not native
RLBench cameras — so we compute point clouds from depth ourselves using
RLBenchDepth2Cloud, exactly as in training (RLBenchDataPreprocessor).

Miscalibration is applied by camera index, matching the training convention:
  cam_idx 0 (orbital_left)  ← "front"      noise from miscalibration_noise.json
  cam_idx 1 (orbital_right) ← "wrist_left" noise
  cam_idx 2 (wrist)         ← "wrist_right" noise
"""

import json
import os
import glob
import random

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from data.generation.orbital.collection import (
    load_group_cameras,
    create_orbital_sensor,
    capture_orbital_extrinsics,
    make_obs_config,
)
from data.generation.orbital.scene import OrbitalEnvironment
try:
    from utils.data_preprocessors.rlbench import _load_miscalibration_noise
except ImportError:
    _load_miscalibration_noise = None
from utils.depth2cloud.rlbench import RLBenchDepth2Cloud

from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
from rlbench.backend.exceptions import InvalidActionError
from pyrep.errors import IKError, ConfigurationPathError

from modeling.encoder.text import fetch_tokenizers
from online_evaluation_rlbench.get_stored_demos import get_stored_demos


def task_file_to_task_class(task_file):
    import importlib
    name = task_file.replace(".py", "")
    class_name = "".join([w[0].upper() + w[1:] for w in name.split("_")])
    mod = importlib.import_module("rlbench.tasks.%s" % name)
    mod = importlib.reload(mod)
    return getattr(mod, class_name)


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

    def __init__(self, policy=None, backbone="clip"):
        self._policy = policy.cuda()
        self._policy.eval()
        self._instr = None
        self.tokenizer = fetch_tokenizers(backbone)

    def load_episode(self, descriptions):
        instr = [random.choice(descriptions)]
        self._instr = self.tokenizer(instr).cuda(non_blocking=True)

    def predict(self, rgbs, pcds, gripper, prediction_len=1):
        """
        Args:
            rgbs:    (1, ncam, 3, H, W)
            pcds:    (1, ncam, 3, H, W)
            gripper: (1, nhist, 8)
            prediction_len: int
        Returns:
            (1, prediction_len, 8)
        """
        dtype = next(self._policy.parameters()).dtype
        return self._policy(
            None,
            torch.full([1, prediction_len, 1], False).cuda(non_blocking=True),
            rgbs.to(dtype),
            None,
            pcds.to(dtype),
            self._instr,
            gripper[:, :, None, :7].to(dtype),
            run_inference=True,
        ).view(1, prediction_len, 8)


class RLBenchEnv:

    def __init__(
        self,
        data_path,
        task_str=None,
        image_size=(256, 256),
        apply_rgb=True,
        apply_pc=True,
        headless=False,
        apply_cameras=("orbital_left", "orbital_right", "wrist"),
        collision_checking=False,
        cameras_file=None,
        task_group_mapping_file=None,
        fov_deg=60.0,
        miscalibration_noise_level=None,
        camera_groups=None,
    ):
        self.data_path = data_path
        self._task_str = task_str
        self.apply_cameras = apply_cameras
        self._fov_deg = fov_deg
        self._camera_groups_override = camera_groups

        if cameras_file is None:
            raise ValueError("cameras_file must be provided for orbital eval")
        self._cameras_file = cameras_file

        if task_group_mapping_file is None:
            raise ValueError("task_group_mapping_file must be provided for orbital eval")
        self._task_group_mapping_file = task_group_mapping_file

        # Miscalibration noise (optional)
        self._miscal_cameras = None
        self._miscal_noise = None
        if miscalibration_noise_level is not None:
            if _load_miscalibration_noise is None:
                raise ImportError("_load_miscalibration_noise is not available; cannot use miscalibration_noise_level")
            self._miscal_cameras, self._miscal_noise = _load_miscalibration_noise(
                miscalibration_noise_level
            )
            print(
                f"[orbital eval] Miscalibration noise: level='{miscalibration_noise_level}', "
                f"cameras={self._miscal_cameras}"
            )

        # Depth → point cloud converter (same as training)
        h, w = image_size if isinstance(image_size, (tuple, list)) else (image_size, image_size)
        self._depth2cloud = RLBenchDepth2Cloud((h, w))
        self._image_h = h

        # Build OrbitalEnvironment (wrist camera via ObservationConfig; orbital via VisionSensors)
        obs_config = make_obs_config(h)
        action_mode = MoveArmThenGripper(
            arm_action_mode=EndEffectorPoseViaPlanning(
                collision_checking=collision_checking
            ),
            gripper_action_mode=Discrete(),
        )
        self.env = OrbitalEnvironment(
            action_mode=action_mode,
            obs_config=obs_config,
            headless=headless,
            dataset_root=str(data_path),
            robot_setup="panda",
        )

        self._left_sensor = None
        self._right_sensor = None
        self._orbital_extrinsics = None  # dict from capture_orbital_extrinsics

    # ------------------------------------------------------------------
    # Task-group mapping
    # ------------------------------------------------------------------

    def _get_task_groups(self, task_str):
        """Return the list of camera groups this task was trained on."""
        with open(self._task_group_mapping_file) as f:
            mapping = json.load(f)
        if task_str not in mapping:
            raise ValueError(
                f"Task '{task_str}' not found in {self._task_group_mapping_file}"
            )
        groups = mapping[task_str]
        if self._camera_groups_override is not None:
            groups = self._camera_groups_override
            # groups = [g for g in self._camera_groups_overrideif g in groups]
            # if not groups:
            #     raise ValueError(
            #         f"No camera groups left after applying override={self._camera_groups_override} "
            #         f"for task='{task_str}'. Available groups from mapping: {mapping[task_str]}"
            #     )
        return groups

    # ------------------------------------------------------------------
    # Sensor lifecycle
    # ------------------------------------------------------------------

    def _spawn_sensors(self, group):
        """Spawn orbital VisionSensors for `group` and capture extrinsics/intrinsics."""
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

    def _remove_sensors(self):
        if self._left_sensor is not None:
            self.env._scene.clear_orbital_sensors()
            self._left_sensor.remove()
            self._right_sensor.remove()
            self._left_sensor = None
            self._right_sensor = None

    # ------------------------------------------------------------------
    # Observation processing
    # ------------------------------------------------------------------

    def _apply_miscalibration(self, extrinsics):
        """Perturb extrinsics by index, matching training convention."""
        ext = extrinsics.clone().float()
        for cam_idx, cam_name in enumerate(self._miscal_cameras):
            if cam_name not in self._miscal_noise:
                continue
            R_noise = self._miscal_noise[cam_name]["R_noise"].to(ext.device)
            t_noise = self._miscal_noise[cam_name]["t_noise"].to(ext.device)
            ext[:, cam_idx, :3, :3] = R_noise @ ext[:, cam_idx, :3, :3]
            ext[:, cam_idx, :3, 3] += t_noise
        return ext.to(extrinsics.dtype)

    def get_rgb_pcd_gripper_from_obs(self, obs):
        """
        Build (rgb, pcd, gripper) from an OrbitalScene observation.

        Returns:
            rgb:     (1, 3, 3, H, W)  float32 in [0, 1]
            pcd:     (1, 3, 3, H, W)  float32 in world coords
            gripper: (1, 8)           float32
        """
        # --- RGB ---
        orbital_left_rgb = (
            torch.tensor(obs.orbital_left_rgb, dtype=torch.float32).permute(2, 0, 1) / 255.0
        )
        orbital_right_rgb = (
            torch.tensor(obs.orbital_right_rgb, dtype=torch.float32).permute(2, 0, 1) / 255.0
        )
        wrist_rgb_raw = self._get_wrist_attr(obs, "wrist_rgb")
        wrist_rgb = torch.tensor(wrist_rgb_raw, dtype=torch.float32).permute(2, 0, 1) / 255.0

        rgb = torch.stack([orbital_left_rgb, orbital_right_rgb, wrist_rgb]).unsqueeze(0)
        # (1, 3, 3, H, W)

        # --- Depth ---
        orbital_left_depth = torch.tensor(obs.orbital_left_depth, dtype=torch.float32)
        orbital_right_depth = torch.tensor(obs.orbital_right_depth, dtype=torch.float32)
        wrist_depth_raw = self._get_wrist_attr(obs, "wrist_depth")
        near_wr = obs.misc.get("wrist_camera_near", 0.1)
        far_wr = obs.misc.get("wrist_camera_far", 4.0)
        wrist_depth = torch.tensor(
            near_wr + wrist_depth_raw * (far_wr - near_wr), dtype=torch.float32
        )

        depth = torch.stack(
            [orbital_left_depth, orbital_right_depth, wrist_depth]
        ).unsqueeze(0)  # (1, 3, H, W)

        # --- Extrinsics + intrinsics ---
        E_left = torch.tensor(
            self._orbital_extrinsics["left_extrinsics"], dtype=torch.float32
        )
        E_right = torch.tensor(
            self._orbital_extrinsics["right_extrinsics"], dtype=torch.float32
        )
        E_wrist = torch.tensor(
            obs.misc.get("wrist_camera_extrinsics", np.eye(4)), dtype=torch.float32
        )

        K_left = torch.tensor(
            self._orbital_extrinsics["left_intrinsics"], dtype=torch.float32
        )
        K_right = torch.tensor(
            self._orbital_extrinsics["right_intrinsics"], dtype=torch.float32
        )
        K_wrist = torch.tensor(
            obs.misc.get("wrist_camera_intrinsics", np.eye(3)), dtype=torch.float32
        )

        extrinsics = torch.stack([E_left, E_right, E_wrist]).unsqueeze(0)  # (1, 3, 4, 4)
        intrinsics = torch.stack([K_left, K_right, K_wrist]).unsqueeze(0)  # (1, 3, 3, 3)

        # --- Miscalibration ---
        if self._miscal_noise is not None:
            extrinsics = self._apply_miscalibration(extrinsics)

        # --- Depth → point cloud ---
        pcd = self._depth2cloud(
            depth.cuda(non_blocking=True).to(torch.bfloat16),
            extrinsics.cuda(non_blocking=True).to(torch.bfloat16),
            intrinsics.cuda(non_blocking=True).to(torch.bfloat16),
        ).float().cpu()  # (1, 3, 3, H, W)

        # --- Gripper ---
        gripper = torch.from_numpy(
            np.concatenate([obs.gripper_pose, [obs.gripper_open]])
        ).float().unsqueeze(0)  # (1, 8)

        return rgb, pcd, gripper

    @staticmethod
    def _get_wrist_attr(obs, key):
        """Fetch wrist data from obs attributes or perception_data dict."""
        val = getattr(obs, key, None)
        if val is None:
            val = obs.perception_data.get(key)
        if val is None:
            raise ValueError(f"Could not find '{key}' in obs")
        return val

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Orbital rollout loader
    # ------------------------------------------------------------------

    def _load_orbital_rollout_demos(self, task_str, group):
        """Load demos from {data_path}/{task_str}/{group}/episode_*/low_dim_obs.pkl."""
        import pickle
        group_dir = os.path.join(self.data_path, task_str, group)
        episode_dirs = sorted(
            d for d in os.listdir(group_dir)
            if d.startswith("episode_") and os.path.isdir(os.path.join(group_dir, d))
        )
        demos = []
        for ep in episode_dirs:
            pkl_path = os.path.join(group_dir, ep, "low_dim_obs.pkl")
            if not os.path.exists(pkl_path):
                continue
            try:
                with open(pkl_path, "rb") as f:
                    demo = pickle.load(f)
            except Exception as e:
                print(f"[orbital eval] WARNING: skipping {pkl_path} (failed to load: {e})", flush=True)
                continue
            demo.variation_number = 0
            demos.append(demo)
        print(f"[orbital eval] loaded {len(demos)} demos from {group_dir}", flush=True)
        return demos

    # Evaluation loop
    # ------------------------------------------------------------------

    def evaluate_task_on_multiple_variations(
        self,
        task_str,
        max_steps,
        actioner,
        max_tries=1,
        prediction_len=1,
        num_history=1,
        save_trajectory=False,
        output_file=None,
    ):
        print(f"[orbital eval] launching env...", flush=True)
        self.env.launch()
        print(f"[orbital eval] env launched", flush=True)

        groups = self._get_task_groups(task_str)
        print(f"[orbital eval] task={task_str} groups={groups}", flush=True)

        task_type = task_file_to_task_class(task_str)
        task = self.env.get_task(task_type)

        # Detect layout:
        #   1. orbital rollout format: {data_path}/{task_str}/{group}/episode_*/
        #   2. variation*/ dirs
        #   3. all_variations/ (loaded lazily)
        orbital_rollout_root = os.path.join(self.data_path, task_str)
        use_orbital_rollout = os.path.isdir(
            os.path.join(orbital_rollout_root, groups[0])
        ) if groups else False

        if not use_orbital_rollout:
            variation_dirs = glob.glob(os.path.join(self.data_path, task_str, "variation*"))
            if variation_dirs:
                task_variations = sorted([
                    int(n.split("/")[-1].replace("variation", ""))
                    for n in variation_dirs
                ])
                demos_by_variation = None
            else:
                all_demos = get_stored_demos(
                    amount=-1,
                    dataset_root=self.data_path,
                    variation_number=-1,
                    task_name=task_str,
                    random_selection=False,
                    from_episode_number=0,
                )
                from collections import defaultdict
                demos_by_variation = defaultdict(list)
                for d in all_demos:
                    demos_by_variation[d.variation_number].append(d)
                task_variations = sorted(demos_by_variation.keys())

        # Accumulate across all (group, variation) pairs
        total_success = 0
        total_demos = 0
        per_group_rates = {}  # group -> {variation: rate}

        for group in groups:
            print(f"[orbital eval] === Group {group} ===", flush=True)
            print(f"[orbital eval] spawning sensors for group {group}...", flush=True)
            self._spawn_sensors(group)
            print(f"[orbital eval] sensors spawned", flush=True)

            if use_orbital_rollout:
                try:
                    group_demos = self._load_orbital_rollout_demos(task_str, group)
                except Exception as e:
                    print(f"[orbital eval] WARNING: skipping group {group} for {task_str} (failed to load demos: {e})", flush=True)
                    continue
                if not group_demos:
                    print(f"[orbital eval] WARNING: skipping group {group} for {task_str} (no valid demos loaded)", flush=True)
                    continue
                task_variations = [0]
                demos_by_variation = {0: group_demos}

            group_rates = {}
            for variation in tqdm(task_variations, desc=f"{task_str}/{group}"):
                task.set_variation(variation)
                pre_loaded = demos_by_variation[variation] if demos_by_variation is not None else None
                success_rate, valid, num_valid_demos = self._evaluate_task_on_one_variation(
                    task_str=task_str,
                    task=task,
                    max_steps=max_steps,
                    variation=variation,
                    actioner=actioner,
                    max_tries=max_tries,
                    prediction_len=prediction_len,
                    num_history=num_history,
                    pre_loaded_demos=pre_loaded,
                    save_trajectory=save_trajectory,
                    output_file=output_file,
                    group=group,
                )
                if valid:
                    group_rates[variation] = success_rate / num_valid_demos
                    total_success += success_rate
                    total_demos += num_valid_demos

            per_group_rates[group] = group_rates
            self._remove_sensors()

        self.env.shutdown()

        result = dict(per_group_rates)
        result["mean"] = total_success / total_demos if total_demos > 0 else 0.0
        return result

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
        num_history=1,
        pre_loaded_demos=None,
        save_trajectory=False,
        output_file=None,
        group=None,
    ):
        success_rate = 0
        total_reward = 0
        var_demos = pre_loaded_demos if pre_loaded_demos is not None else get_stored_demos(
            amount=-1,
            dataset_root=self.data_path,
            variation_number=variation,
            task_name=task_str,
            random_selection=False,
            from_episode_number=0,
        )

        print(f"  [var {variation}] {len(var_demos)} demos", flush=True)
        for demo_id, demo in enumerate(var_demos):

            print(f"  [var {variation}] demo {demo_id+1}/{len(var_demos)} — resetting...", flush=True)
            grippers = torch.Tensor([]).cuda(non_blocking=True)
            descriptions, obs = task.reset_to_demo(demo)
            actioner.load_episode(descriptions)
            print(f"  [var {variation}] demo {demo_id+1} — running up to {max_steps} steps", flush=True)

            move = Mover(task, max_tries=max_tries)
            max_reward = 0.0
            trajectory = []  # list of (step_id, action) rows

            for step_id in range(max_steps):

                print(f"    step {step_id+1}/{max_steps} — getting obs...", flush=True)
                rgb, pcd, gripper = self.get_rgb_pcd_gripper_from_obs(obs)
                rgbs_input = rgb.cuda(non_blocking=True)
                pcds_input = pcd.cuda(non_blocking=True)
                gripper = gripper.cuda(non_blocking=True)
                grippers = torch.cat([grippers, gripper.unsqueeze(1)], 1)

                gripper_input = grippers[:, -num_history:]
                npad = num_history - gripper_input.shape[1]
                gripper_input = F.pad(
                    gripper_input, (0, 0, npad, 0), mode="replicate"
                )

                print(f"    step {step_id+1}/{max_steps} — predicting...", flush=True)
                output = actioner.predict(
                    rgbs_input,
                    pcds_input,
                    gripper_input,
                    prediction_len=prediction_len,
                )
                print(f"    step {step_id+1}/{max_steps} — executing action...", flush=True)

                try:
                    actions = output[-1].cpu().numpy()
                    actions[:, -1] = actions[:, -1].round()
                    print(f"    predicted action: xyz={actions[0, :3]} quat={actions[0, 3:7].round(2)} gripper={actions[0, -1]:.2f}", flush=True)

                    if save_trajectory:
                        for sub_id, action in enumerate(actions):
                            trajectory.append((step_id, sub_id, action.tolist()))

                    for action in actions:
                        obs, reward, _ = move(action, collision_checking=False)

                    max_reward = max(max_reward, reward)
                    print(f"    step {step_id+1}/{max_steps} — reward={reward:.2f}", flush=True)

                    if reward == 1:
                        success_rate += 1
                        break

                except (IKError, ConfigurationPathError, InvalidActionError) as e:
                    print(task_str, demo, step_id, success_rate, e)
                    reward = 0

            total_reward += max_reward

            if save_trajectory and output_file is not None:
                traj_dir = os.path.join(os.path.dirname(output_file), "trajectories")
                os.makedirs(traj_dir, exist_ok=True)
                group_tag = group if group is not None else "default"
                traj_path = os.path.join(
                    traj_dir,
                    f"{task_str}_{group_tag}_var{variation}_demo{demo_id}.txt",
                )
                with open(traj_path, "w") as f:
                    f.write("step sub_step x y z qx qy qz qw gripper\n")
                    for step_id, sub_id, action in trajectory:
                        row = f"{step_id} {sub_id} " + " ".join(f"{v:.6f}" for v in action)
                        f.write(row + "\n")
                print(f"  [trajectory] saved to {traj_path}", flush=True)

            print(
                task_str,
                "Variation", variation,
                "Demo", demo_id,
                "Reward", f"{reward:.2f}",
                "max_reward", f"{max_reward:.2f}",
                f"SR: {success_rate}/{demo_id + 1}",
                f"SR: {total_reward:.2f}/{demo_id + 1}",
                "# valid demos", demo_id + 1,
            )

        valid = len(var_demos) > 0
        return success_rate, valid, len(var_demos)
