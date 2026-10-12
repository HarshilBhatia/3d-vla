import json
import os
import glob
import random

from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F


from rlbench.observation_config import ObservationConfig, CameraConfig
from rlbench.environment import Environment
from rlbench.action_modes.action_mode import BimanualMoveArmThenGripper
from rlbench.action_modes.gripper_action_modes import BimanualDiscrete, assert_action_shape
from rlbench.action_modes.arm_action_modes import ArmActionMode, BimanualEndEffectorPoseViaPlanning
from rlbench.backend.exceptions import InvalidActionError
from pyrep.errors import IKError, ConfigurationPathError
from pyrep.const import RenderMode

from modeling.encoder.text import fetch_tokenizers
from ..stored_demos import get_stored_demos

from data.preprocessing.miscalibration import (
    setup_miscalibration,
    per_cam_noise_T,
    build_pcd_from_obs,
)


def task_file_to_task_class(task_file):
    import importlib

    name = task_file.replace(".py", "")
    class_name = "".join([w[0].upper() + w[1:] for w in name.split("_")])
    mod = importlib.import_module("rlbench.bimanual_tasks.%s" % name)
    mod = importlib.reload(mod)
    task_class = getattr(mod, class_name)
    return task_class


class Mover:

    def __init__(self, task, max_tries=1):
        self._task = task
        self._last_action = None
        self._max_tries = max_tries

    def __call__(self, action, collision_checking=False):
        # action is an array (2, 8)
        obs = None
        terminate = None
        reward = 0

        # Try to reach the desired pose without changing the gripper state
        target = action.copy()
        if self._last_action is not None:
            action[:, 7] = self._last_action[:, 7].copy() # copy gripper state
        for _ in range(self._max_tries):
            action_collision = np.ones((action.shape[0], action.shape[1]+1))
            action_collision[:, :-1] = action
            if collision_checking:
                action_collision[:, -1] = 0
            # Peract2 takes (right, left) action, but we predict (left, right)
            action_collision = action_collision[::-1]
            action_collision = action_collision.ravel()
            obs, reward, terminate = self._task.step(action_collision)

            # Check if we reached the desired pose (planner may be inaccurate)
            l_pos = obs.left.gripper_pose[:3]
            r_pos = obs.right.gripper_pose[:3]
            l_dist_pos = np.sqrt(np.square(target[0, :3] - l_pos).sum())
            r_dist_pos = np.sqrt(np.square(target[1, :3] - r_pos).sum())
            criteria = (l_dist_pos < 5e-3, r_dist_pos < 5e-3)

            if all(criteria) or reward == 1:
                break

        # Then execute with gripper action (open/close))
        action = target
        if (
            not reward == 1.0
            and self._last_action is not None
            and (  # if any gripper state has changed, re-execute
                action[0, 7] != self._last_action[0, 7]
                or action[1, 7] != self._last_action[1, 7]
            )
        ):
            action_collision = np.ones((action.shape[0], action.shape[1]+1))
            action_collision[:, :-1] = action
            if collision_checking:
                action_collision[:, -1] = 0
            action_collision = action_collision[::-1]
            action_collision = action_collision.ravel()
            obs, reward, terminate = self._task.step(action_collision)

        # Store the last action action for the gripper state
        self._last_action = action.copy()

        return obs, reward, terminate


class Actioner:

    def __init__(self, policy=None, backbone='clip', cfg_scale=None):
        self._policy = policy.cuda()
        self._policy.eval()
        self._instr = None
        self.tokenizer = fetch_tokenizers(backbone)
        self._cfg_scale = float(cfg_scale) if cfg_scale is not None else None

    def load_episode(self, descriptions):
        instr = [random.choice(descriptions)]
        self._instr = self.tokenizer(instr).cuda(non_blocking=True)

    def predict(self, rgbs, pcds, gripper, prediction_len=1, proprio_pcds=None):
        """
        Args:
            rgbs: (1, nhist, ncam, 3, H, W) when ``nhist > 1``;
                the legacy current-frame layout is retained for ``nhist == 1``.
            pcds: same layout as ``rgbs``.
            gripper: (1, nhist, 2*8)
            prediction_len: int

        Returns:
            trajectory: (1, nhist, nhand=2, 8)
        """
        policy_kwargs = dict(
            run_inference=True,
        )
        # The 2D PerAct actor does not implement classifier-free guidance;
        # passing cfg_scale=None is still an unsupported keyword for it.
        if self._cfg_scale is not None:
            policy_kwargs["cfg_scale"] = self._cfg_scale
        if proprio_pcds is not None:
            policy_kwargs["proprio_pcd"] = proprio_pcds
        return self._policy(
            None,
            torch.full([1, prediction_len, 2], False).cuda(non_blocking=True),
            rgbs,
            None,
            pcds,
            self._instr,
            gripper.unflatten(-1, (2, -1)),  # (1, nhist, nhand=2, 8)
            **policy_kwargs,
        )


def _append_causal_visual_history(history, current, visual_num_history):
    """Append one observation and return a causal, oldest→current visual window.

    Training builds Video-DeltaM windows from adjacent records and repeats the
    first valid frame at an episode boundary.  Online rollouts must use the
    same convention rather than silently feeding the K=5 model a single frame.
    ``current`` has shape ``(B, ncam, C, H, W)`` and the return has shape
    ``(B, visual_num_history, ncam, C, H, W)`` for histories longer than one.
    """
    history = current.unsqueeze(1) if history is None else torch.cat(
        (history, current.unsqueeze(1)), dim=1
    )
    # A bounded ring buffer prevents unbounded rollout memory growth.
    history = history[:, -visual_num_history:]
    missing = visual_num_history - history.shape[1]
    if missing:
        history = torch.cat((history[:, :1].expand(-1, missing, *history.shape[2:]), history), dim=1)
    return history


def _random_visual_history_indices(num_frames, visual_num_history, rng):
    """Mirror of BaseDataset._random_hist_indices: K-1 distinct random past frames, current last."""
    current = num_frames - 1
    picked = rng.sample(range(current), min(current, visual_num_history - 1))
    return [0] * (visual_num_history - 1 - len(picked)) + picked + [current]


def _causal_proprio_window(history, visual_num_history):
    """Reproduce the dataset's Video-DeltaM proprio convention exactly.

    ``datasets/base.py:_get_single_proprio_hist`` returns the state at the
    sampled record first, followed by preceding records (current→past).  At a
    demo boundary it pads the *tail* with the current state.  This is unlike
    visual history, which is oldest→current and left-padded with the earliest
    frame.  Keep the distinction here so online evaluation feeds an existing
    checkpoint precisely the layout it was trained with.
    """
    window = torch.flip(history[:, -visual_num_history:], dims=(1,))
    missing = visual_num_history - window.shape[1]
    if missing:
        window = torch.cat(
            (window, window[:, :1].expand(-1, missing, *window.shape[2:])),
            dim=1,
        )
    return window


def _proprio_window(history, visual_num_history, order):
    """Return either the checkpoint-faithful or causal temporal convention."""
    if order == "current_to_past":
        return _causal_proprio_window(history, visual_num_history)
    if order == "past_to_current":
        window = history[:, -visual_num_history:]
        missing = visual_num_history - window.shape[1]
        if missing:
            window = torch.cat(
                (window[:, :1].expand(-1, missing, *window.shape[2:]), window),
                dim=1,
            )
        return window
    raise ValueError(
        f"Unknown proprio_history_order={order!r}; expected "
        "'current_to_past' or 'past_to_current'"
    )


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
        apply_cameras=("over_shoulder_left", "over_shoulder_right", "wrist_left", "wrist_right", "front"),
        collision_checking=False,
        use_depth2cloud=False,
        controller="planner",
    ):
        if controller not in ("planner", "ik"):
            raise ValueError(f"eval_controller must be planner or ik, got {controller!r}")
        self._controller = controller
        self._use_depth2cloud = use_depth2cloud
        if self._use_depth2cloud:
            apply_depth = True

        # setup required inputs
        self.data_path = data_path
        self.apply_cameras = apply_cameras
        self._task_str = task_str

        # setup RLBench environments
        self.obs_config = self.create_obs_config(
            image_size, apply_rgb, apply_depth, apply_pc, apply_cameras
        )

        self.action_mode = BimanualMoveArmThenGripper(
            arm_action_mode=(BimanualEndEffectorPoseViaIK() if controller == "ik"
                             else BimanualEndEffectorPoseViaPlanning(collision_checking=collision_checking)),
            gripper_action_mode=HandoverDiscrete() if 'handover' in task_str else BimanualDiscrete()
        )
        self.env = self._make_env(data_path, headless)

        # depth2cloud module (optional, enabled via use_depth2cloud)
        ctx = setup_miscalibration(
            level=None,
            image_size=image_size,
            build_depth2cloud=self._use_depth2cloud,
            log_prefix="[bimanual eval]",
        )
        self._depth2cloud = ctx.depth2cloud
        self._miscal_T = None

    def _make_env(self, data_path, headless):
        """Build the RLBench Environment. Subclasses may return a different class."""
        return Environment(
            self.action_mode, str(data_path), self.obs_config,
            headless=headless, robot_setup="dual_panda"
        )

    def _launch_env(self):
        """Launch the CoppeliaSim environment. Subclasses may extend."""
        self.env.launch()

    def _get_pcd_from_depth(self, obs):
        """Compute point clouds from depth + extrinsics, applying miscalibration if configured.

        Returns:
            pcd: (1, ncam, 3, H, W) float32 CPU tensor
        """
        return build_pcd_from_obs(
            obs, self.apply_cameras, self._depth2cloud,
            T_noise=self._miscal_T,
            depth_accessor=lambda o, name: o.perception_data[name],
        )

    def get_rgb_pcd_gripper_from_obs(self, obs):
        """
        Return rgb, pcd, and gripper from a given observation
        :param obs: an Observation from the env
        :return: rgb, pcd, gripper
        """
        rgb = torch.stack([
            torch.tensor(obs.perception_data["{}_rgb".format(cam)]).float().permute(2, 0, 1) / 255.0
            for cam in self.apply_cameras
        ]).unsqueeze(0)  # 1, N, C, H, W

        if self._use_depth2cloud:
            pcd = self._get_pcd_from_depth(obs)
        else:
            pcd = torch.stack([
                torch.tensor(obs.perception_data["{}_point_cloud".format(cam)]).float().permute(2, 0, 1)
                for cam in self.apply_cameras
            ]).unsqueeze(0)  # 1, N, C, H, W

        # action is an array of length 16 = (7+1)*2
        gripper = torch.from_numpy(np.concatenate([
            obs.left.gripper_pose, [obs.left.gripper_open],
            obs.right.gripper_pose, [obs.right.gripper_open]
        ])).float().unsqueeze(0)  # 1, D

        return rgb, pcd, gripper

    def _apply_history_probe(self, probe, rgbs, pcds, step_id, sampling, rng):
        """Replace the K-1 past visual slots (current frame stays last) for an eval-only history probe."""
        if probe == "current_only" or (probe == "other_episode" and getattr(self, "_probe_prev", None) is None):
            if probe == "other_episode":
                print("[history probe] no previous episode yet; this episode uses current_only", flush=True)
            return rgbs[:, -1:].expand_as(rgbs).clone(), pcds[:, -1:].expand_as(pcds).clone()
        if probe != "other_episode":
            raise ValueError(f"eval_history_probe must be none, current_only or other_episode, got {probe!r}")
        prev_rgb, prev_pcd = self._probe_prev
        past, length = rgbs.shape[1] - 1, prev_rgb.shape[1]
        if sampling == "random":
            idx = sorted(rng.sample(range(length), min(length, past)))
            idx = [idx[0]] * (past - len(idx)) + idx
        else:
            # Same relative position in the other episode as a consecutive window here.
            idx = [min(max(step_id - k, 0), length - 1) for k in range(past, 0, -1)]
        rgbs, pcds = rgbs.clone(), pcds.clone()
        rgbs[:, :-1], pcds[:, :-1] = prev_rgb[:, idx], prev_pcd[:, idx]
        return rgbs, pcds

    def _extract_video_frame(self, obs):
        frames = [obs.perception_data[f"{cam}_rgb"] for cam in self.apply_cameras]
        return np.concatenate(frames, axis=1)

    def evaluate_task_on_multiple_variations(
        self,
        task_str,
        max_steps,
        actioner,
        max_tries=1,
        prediction_len=1,
        visual_num_history=1,
        proprio_num_history=None,
        proprio_history_order="past_to_current",
        save_trajectory=False,
        save_video=False,
        output_file=None,
        progress_file=None,
        num_demos=None,
        num_demos_total=None,
        visual_history_sampling="consecutive",
        history_seed=0,
        history_probe="none",
        history_every_step=False,
    ):
        progress = {}
        if progress_file is not None and os.path.exists(progress_file):
            with open(progress_file) as f:
                progress = json.load(f)
            print(f"[resume] loaded {len(progress)} completed demos from {progress_file}", flush=True)

        self._launch_env()
        task_type = task_file_to_task_class(task_str)
        task = self.env.get_task(task_type)
        task_variations = glob.glob(
            os.path.join(self.data_path, task_str, "variation*")
        )
        task_variations = sorted(
            int(n.split('/')[-1].replace('variation', ''))
            for n in task_variations
        )

        # num_demos caps episodes PER VARIATION; num_demos_total is a budget for
        # the whole task, spread as evenly as possible over its variations so a
        # 46-variation task costs the same as a single-variation one.
        per_variation = None
        if num_demos_total is not None:
            n_var = len(task_variations)
            base, extra = divmod(int(num_demos_total), n_var)
            per_variation = {
                v: base + (1 if i < extra else 0)
                for i, v in enumerate(task_variations)
            }
            task_variations = [v for v in task_variations if per_variation[v] > 0]
            print(
                f"[eval] num_demos_total={num_demos_total} over {n_var} variations -> "
                f"{ {v: per_variation[v] for v in task_variations} }",
                flush=True,
            )

        # Video-DeltaM may use a longer visual history than proprio history.
        # Keep older checkpoints backward compatible by using visual K when
        # no independent proprio length is supplied.
        if proprio_num_history is None:
            proprio_num_history = visual_num_history

        var_success_rates = {}
        var_num_valid_demos = {}

        for variation in tqdm(task_variations):
            task.set_variation(variation)
            var_num_demos = (
                per_variation[variation] if per_variation is not None else num_demos
            )
            success_rate, valid, num_valid_demos = (
                self._evaluate_task_on_one_variation(
                    task_str=task_str,
                    task=task,
                    max_steps=max_steps,
                    variation=variation,
                    actioner=actioner,
                    max_tries=max_tries,
                    prediction_len=prediction_len,
                    visual_num_history=visual_num_history,
                    proprio_num_history=proprio_num_history,
                    proprio_history_order=proprio_history_order,
                    save_video=save_video,
                    output_file=output_file,
                    progress=progress,
                    progress_file=progress_file,
                    num_demos=var_num_demos,
                    visual_history_sampling=visual_history_sampling,
                    history_seed=history_seed,
                    history_probe=history_probe,
                    history_every_step=history_every_step,
                )
            )
            if valid:
                var_success_rates[variation] = success_rate
                var_num_valid_demos[variation] = num_valid_demos

        self.env.shutdown()

        var_success_rates["mean"] = (
            sum(var_success_rates.values()) /
            sum(var_num_valid_demos.values())
        )

        return var_success_rates

    @torch.no_grad()
    def _evaluate_task_on_one_variation(
        self,
        task_str,  # this is str
        task,  # this instance of TaskEnvironment
        max_steps,
        variation,
        actioner,
        max_tries=1,
        prediction_len=50,
        visual_num_history=1,
        proprio_num_history=None,
        proprio_history_order="past_to_current",
        save_video=False,
        output_file=None,
        progress=None,
        progress_file=None,
        num_demos=None,
        visual_history_sampling="consecutive",
        history_seed=0,
        history_probe="none",
        history_every_step=False,
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
        if num_demos is not None:
            var_demos = var_demos[:num_demos]
            print(
                f"  capped to {len(var_demos)} demos (num_demos={num_demos})",
                flush=True,
            )

        for demo_id, demo in enumerate(var_demos):

            key = f"var{variation}_demo{demo_id}"
            if progress is not None and key in progress:
                result = progress[key]
                success_rate += result
                total_reward += result
                print(f"  [resume] {key}: {'success' if result else 'fail'}", flush=True)
                continue

            grippers = torch.Tensor([]).cuda(non_blocking=True)
            rgb_history = None
            pcd_history = None
            clean_history = None
            clean_proprio = getattr(self, "_clean_proprio", False)
            if clean_proprio and history_probe != "none":
                raise ValueError("eval_clean_proprio cannot be combined with eval_history_probe")
            if history_every_step and (visual_history_sampling != "random" or clean_proprio or history_probe != "none"):
                raise ValueError("dense visual history needs random sampling and no clean-proprio or history probe")
            # Dense-history models saw every timestep in training, so keep the planner's in-between frames too.
            between = []
            self.action_mode.arm_action_mode.set_callable_each_step(between.append if history_every_step else None)
            descriptions, obs = task.reset_to_demo(demo)
            actioner.load_episode(descriptions)
            history_rng = random.Random(f"{history_seed}-{task_str}-{variation}-{demo_id}")
            episode_rgb, episode_pcd = [], []

            move = IKMover(task) if self._controller == "ik" else Mover(task, max_tries=max_tries)
            max_reward = 0.0
            video_frames = [] if save_video else None

            for step_id in range(max_steps):

                if save_video:
                    video_frames.append(self._extract_video_frame(obs))

                # Fetch the current observation and predict one action
                rgb, pcd, gripper = self.get_rgb_pcd_gripper_from_obs(obs)
                rgb_current = rgb.cuda(non_blocking=True)
                pcd_current = pcd.cuda(non_blocking=True)
                clean_input = None
                if clean_proprio:
                    clean_current = self._last_clean_pcd.cuda(non_blocking=True)
                # Keep visual history at the policy replan cadence.  The
                # returned windows are oldest→current, exactly matching the
                # training loader's `_get_single_frame_hist` convention.
                if history_every_step and between and rgb_history is not None:
                    rgb_between, pcd_between = zip(*[self.get_rgb_pcd_gripper_from_obs(o)[:2] for o in between[:-1]]) if len(between) > 1 else ((), ())
                    between.clear()
                    if rgb_between:
                        rgb_history = torch.cat([rgb_history, torch.stack(rgb_between, 1).cuda(non_blocking=True)], dim=1)
                        pcd_history = torch.cat([pcd_history, torch.stack(pcd_between, 1).cuda(non_blocking=True)], dim=1)
                if visual_history_sampling == "random":
                    # Keep every episode frame and draw an unordered history set, as in training.
                    rgb_history = rgb_current.unsqueeze(1) if rgb_history is None else torch.cat(
                        (rgb_history, rgb_current.unsqueeze(1)), dim=1
                    )
                    pcd_history = pcd_current.unsqueeze(1) if pcd_history is None else torch.cat(
                        (pcd_history, pcd_current.unsqueeze(1)), dim=1
                    )
                    hist_idx = _random_visual_history_indices(rgb_history.shape[1], visual_num_history, history_rng)
                    rgbs_input = rgb_history[:, hist_idx]
                    pcds_input = pcd_history[:, hist_idx]
                    if clean_proprio:
                        # The true-extrinsics cloud follows exactly the same history and indices as the normal one.
                        clean_history = clean_current.unsqueeze(1) if clean_history is None else torch.cat(
                            (clean_history, clean_current.unsqueeze(1)), dim=1
                        )
                        clean_input = clean_history[:, hist_idx]
                else:
                    rgbs_input = _append_causal_visual_history(
                        rgb_history, rgb_current, visual_num_history
                    )
                    pcds_input = _append_causal_visual_history(
                        pcd_history, pcd_current, visual_num_history
                    )
                    rgb_history = rgbs_input
                    pcd_history = pcds_input
                    if clean_proprio:
                        clean_input = _append_causal_visual_history(clean_history, clean_current, visual_num_history)
                        clean_history = clean_input
                if history_probe != "none" and visual_num_history > 1:
                    episode_rgb.append(rgb_current)
                    episode_pcd.append(pcd_current)
                    rgbs_input, pcds_input = self._apply_history_probe(
                        history_probe, rgbs_input, pcds_input, step_id, visual_history_sampling, history_rng
                    )
                # Preserve the pre-Video-DeltaM input layout for old models.
                if visual_num_history == 1:
                    rgbs_input = rgbs_input[:, 0]
                    pcds_input = pcds_input[:, 0]
                    clean_input = None if clean_input is None else clean_input[:, 0]
                gripper = gripper.cuda(non_blocking=True)
                grippers = torch.cat([grippers, gripper.unsqueeze(1)], 1)

                # Proprio's length and temporal order are independent of the
                # visual K.  Do not silently expand a K=3 proprio input to K=5
                # just because Video-DeltaM consumes five visual frames.
                gripper_input = _proprio_window(
                    grippers,
                    visual_num_history if proprio_num_history is None else proprio_num_history,
                    proprio_history_order,
                )

                output = actioner.predict(
                    rgbs_input,
                    pcds_input,
                    gripper_input,
                    prediction_len=prediction_len,
                    **({"proprio_pcds": clean_input} if clean_proprio else {}),
                )

                # Update the observation based on the predicted action
                try:
                    # Execute entire predicted trajectory step by step
                    actions = output[-1].cpu().numpy()
                    actions[..., -1] = actions[..., -1].round()

                    # execute
                    for action in actions:
                        obs, reward, _ = move(action, collision_checking=False)
                        if reward == 1:
                            break

                    max_reward = max(max_reward, reward)

                    if reward == 1:
                        success_rate += 1
                        break

                except (IKError, ConfigurationPathError, InvalidActionError) as e:
                    print(task_str, demo, step_id, success_rate, e)
                    reward = 0

            total_reward += max_reward
            if episode_rgb:
                # The next episode's other-episode probe reads this episode's frames.
                self._probe_prev = (torch.stack(episode_rgb, 1), torch.stack(episode_pcd, 1))

            if progress is not None and progress_file is not None:
                progress[key] = 1 if max_reward == 1 else 0
                with open(progress_file, 'w') as pf:
                    json.dump(progress, pf, indent=2)

            if save_video and video_frames and output_file is not None:
                video_dir = os.path.join(os.path.dirname(output_file), "videos", "success" if max_reward == 1 else "fail")
                os.makedirs(video_dir, exist_ok=True)
                video_path = os.path.join(
                    video_dir, f"{task_str}_var{variation}_demo{demo_id}.gif"
                )
                imgs = [Image.fromarray(f) for f in video_frames]
                imgs[0].save(
                    video_path, save_all=True, append_images=imgs[1:], duration=150, loop=0
                )
                print(f"  [video] saved to {video_path}", flush=True)

            print(
                task_str,
                "Variation",
                variation,
                "Demo",
                demo_id,
                "Reward",
                f"{reward:.2f}",
                "max_reward",
                f"{max_reward:.2f}",
                f"SR: {success_rate}/{demo_id+1}",
                f"SR: {total_reward:.2f}/{demo_id+1}",
                "# valid demos", demo_id + 1
            )

        # Compensate for failed demos
        valid = len(var_demos) > 0

        return success_rate, valid, len(var_demos)

    def create_obs_config(
        self, image_size, apply_rgb, apply_depth, apply_pc, apply_cameras, **kwargs
    ):
        """
        Set up observation config for RLBench environment.
            :param image_size: Image size.
            :param apply_rgb: Applying RGB as inputs.
            :param apply_depth: Applying Depth as inputs.
            :param apply_pc: Applying Point Cloud as inputs.
            :param apply_cameras: Desired cameras.
            :return: observation config
        """
        # Define a config for an unused camera with all applications as False.
        unused_cams = CameraConfig()
        unused_cams.set_all(False)

        # Define a config for a used camera with the given image size and flags
        used_cams = CameraConfig(
            rgb=apply_rgb,
            point_cloud=apply_pc,
            depth=apply_depth,
            mask=False,
            image_size=image_size,
            render_mode=RenderMode.OPENGL3,  # note OPENGL3 for Peract2!
            **kwargs
        )

        # apply_cameras is a tuple with the names(str) of all the cameras
        camera_names = apply_cameras
        cameras = {}
        for name in camera_names:
            cameras[name] = used_cams

        obs_config = ObservationConfig(
            camera_configs=cameras,
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


class BimanualEndEffectorPoseViaIK(ArmActionMode):
    """Both arms to a nearby pose via Jacobian IK, stepping physics until reached; for fixed-time action chunks."""

    def __init__(self, max_substeps=10, stop_when_still=True, planner_fallback=True):
        self._max_substeps = max_substeps
        self._stop_when_still = stop_when_still
        # CoppeliaSim's Jacobian IK can fail on cm-scale steps near grasps; the planner then moves to that one pose.
        self._fallback = BimanualEndEffectorPoseViaPlanning() if planner_fallback else None
        self.fallbacks = 0

    def action(self, scene, action, ignore_collisions=None):
        arms = (scene.robot.right_arm, scene.robot.left_arm)
        targets = []
        for arm, pose in zip(arms, (action[:7], action[7:])):
            try:
                targets.append(arm.solve_ik_via_jacobian(pose[:3], quaternion=pose[3:]))
            except IKError as e:
                if self._fallback is None:
                    raise InvalidActionError("Jacobian IK failed; target too far from the current pose") from e
                self.fallbacks += 1
                self._fallback.set_callable_each_step(self._callable_each_step)
                return self._fallback.action(scene, action, [True, True] if ignore_collisions is None else ignore_collisions)
        for arm, joints in zip(arms, targets):
            arm.set_joint_target_positions(joints)
        prev = None
        for _ in range(self._max_substeps):
            scene.step()
            if self._callable_each_step is not None:
                self._callable_each_step(scene.get_observation())
            cur = np.concatenate([arm.get_joint_positions() for arm in arms])
            if np.allclose(cur, np.concatenate(targets), atol=0.01) or (
                    self._stop_when_still and prev is not None and np.allclose(cur, prev, atol=0.001)):
                break
            prev = cur

    def action_shape(self, scene):
        return 14,

    def unimanual_action_shape(self, scene):
        return 7,


class IKMover:
    """Mover for the IK controller: one environment step per chunk pose, no planner retries."""

    def __init__(self, task):
        self._task = task

    def __call__(self, action, collision_checking=False):
        step = np.ones((action.shape[0], action.shape[1] + 1))
        step[:, :-1] = action
        # Peract2 takes (right, left); we predict (left, right).
        return self._task.step(step[::-1].ravel())


class HandoverDiscrete(BimanualDiscrete):
    """
    A custom gripper action mode for the handover task.
    It forces one gripper to release so that the other grasps.
    """

    def action(self, scene, action):
        assert_action_shape(action, self.action_shape(scene.robot))
        if 0.0 > action[0] > 1.0:
            raise InvalidActionError(
                'Gripper action expected to be within 0 and 1.')

        if 0.0 > action[1] > 1.0:
            raise InvalidActionError(
                'Gripper action expected to be within 0 and 1.')

        right_open_condition = all(
            x > 0.9 for x in scene.robot.right_gripper.get_open_amount())

        left_open_condition = all(
            x > 0.9 for x in scene.robot.left_gripper.get_open_amount())

        right_current_ee = 1.0 if right_open_condition else 0.0
        left_current_ee = 1.0 if left_open_condition else 0.0

        right_action = float(action[0] > 0.5)
        left_action = float(action[1] > 0.5)

        if right_current_ee != right_action or left_current_ee != left_action:
            if not self._detach_before_open:
                self._actuate(scene, action)

        # Move objects between grippers
        if right_current_ee != right_action:
            if right_action == 0.0 and self._attach_grasped_objects:
                left_grasped_objects = scene.robot.left_gripper.get_grasped_objects()
                for g_obj in scene.task.get_graspable_objects():
                    if g_obj in left_grasped_objects:
                        scene.robot.left_gripper.release()
                        scene.robot.right_gripper.grasp(g_obj)
                    else:
                        scene.robot.right_gripper.grasp(g_obj)
            else:
                scene.robot.right_gripper.release()
        if left_current_ee != left_action:
            if left_action == 0.0 and self._attach_grasped_objects:
                right_grasped_objects = scene.robot.right_gripper.get_grasped_objects()
                for g_obj in scene.task.get_graspable_objects():
                    if g_obj in right_grasped_objects:
                        scene.robot.right_gripper.release()
                        scene.robot.left_gripper.grasp(g_obj)
                    else:
                        scene.robot.left_gripper.grasp(g_obj)
            else:
                scene.robot.left_gripper.release()

        if right_current_ee != right_action or left_current_ee != left_action:
            if self._detach_before_open:
                self._actuate(scene, action)
            if right_action == 1.0 or left_action == 1.0:
                # Step a few more times to allow objects to drop
                for _ in range(10):
                    scene.pyrep.step()
                    scene.task.step()
