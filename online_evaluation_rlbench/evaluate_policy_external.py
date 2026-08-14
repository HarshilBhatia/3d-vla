"""Online evaluation script for an external policy served over WebSocket.

The policy server must accept observations via client.infer(obs_dict) and return
{"actions": np.ndarray of shape (action_horizon, action_dim)}.

Observation dict keys sent to the server:
    "observation/orbital_left"   — uint8 (H, W, 3)
    "observation/orbital_right"  — uint8 (H, W, 3)
    "observation/wrist"          — uint8 (H, W, 3)
    "observation/state"          — float32 (8,)  [gripper_pose(7) + gripper_open(1)]
    "observation/prompt"         — str

Server address: set env var POLICY_SOCKET_PATH (default: /tmp/policy.sock)
    Make sure to bind the socket into apptainer:
        --bind /tmp/policy.sock:/tmp/policy.sock

Usage (same CLI surface as evaluate_policy.py, minus checkpoint):
    python online_evaluation_rlbench/evaluate_policy_external.py \
        dataset=OrbitalWrist \
        task=open_drawer \
        output_file=eval_logs/external/open_drawer.json \
        data_dir=/path/to/demos \
        cameras_file=instructions/orbital_cameras_grouped.json \
        task_group_mapping_file=instructions/task_group_mapping_subset.json \
        bimanual=false \
        spawn_camera_group=G5
"""

import json
import os
import random
import sys
from pathlib import Path

import asyncio

import numpy as np
import torch
import websockets
from openpi_client import msgpack_numpy

from datasets import fetch_dataset_class
from utils.common_utils import round_floats
from utils.hydra_utils import get_config, get_config_path


_CAM_NAMES = ["orbital_left", "orbital_right", "wrist"]


class WebsocketClientPolicy:
    """WebSocket client over a Unix socket using the openpi msgpack wire format."""

    def __init__(self, socket_path: str):
        self._socket_path = socket_path
        self._packer = msgpack_numpy.Packer()
        self._ws = None
        self._loop = asyncio.new_event_loop()
        self._loop.run_until_complete(self._connect())

    async def _connect(self):
        self._ws = await websockets.unix_connect(self._socket_path, max_size=None)
        # server sends a metadata frame on connect — consume it
        metadata = msgpack_numpy.unpackb(await self._ws.recv())
        print(f"[WebSocketClient] connected to {self._socket_path}, metadata: {metadata}", flush=True)

    def infer(self, obs: dict) -> dict:
        return self._loop.run_until_complete(self._infer(obs))

    async def _infer(self, obs: dict) -> dict:
        await self._ws.send(self._packer.pack(obs))
        return msgpack_numpy.unpackb(await self._ws.recv())

# Per-server observation key remapping. Each entry maps the canonical RLBench
# camera keys to whatever the target server expects.
_SERVER_OBS_REMAP: dict[str, dict[str, str]] = {
    "openpi": {
        "observation/orbital_left": "observation/base_image",
        "observation/orbital_right": "observation/left_wrist_image",
        "observation/wrist": "observation/right_wrist_image",
        "observation/prompt": "prompt",
    },
    # Add new server types here, e.g.:
    # "groot": { ... },
}


def _remap_obs(obs: dict, server_type: str | None) -> dict:
    if server_type is None or server_type not in _SERVER_OBS_REMAP:
        return obs
    remap = _SERVER_OBS_REMAP[server_type]
    return {remap.get(k, k): v for k, v in obs.items()}


class WebSocketActioner:
    """Actioner that forwards observations to an external WebSocket policy server."""

    def __init__(self, socket_path: str, server_type: str | None = None):
        self._client = WebsocketClientPolicy(socket_path=socket_path)
        self._server_type = server_type
        self._prompt = ""

    def load_episode(self, descriptions):
        self._prompt = random.choice(descriptions)

    def predict(self, rgbs, pcds, gripper, prediction_len=1):
        """
        Args:
            rgbs:    (1, ncam, 3, H, W)  float32 [0, 1]
            pcds:    (1, ncam, 3, H, W)  float32  (unused — policy uses RGB only)
            gripper: (1, nhist, 8)       float32  [xyz, quat(4), gripper_open]
            prediction_len: int          how many steps to execute from the chunk
        Returns:
            (1, prediction_len, action_dim) float32 tensor on CPU
        """
        # --- build obs dict ---
        obs = {}

        # cameras: (ncam, 3, H, W) float [0,1] → (H, W, 3) uint8 per cam
        rgb_np = rgbs[0].cpu().numpy()  # (ncam, 3, H, W)
        for i, cam_name in enumerate(_CAM_NAMES):
            img = (rgb_np[i].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
            obs[f"observation/{cam_name}"] = img

        # proprioception: latest history step, shape (8,)
        obs["observation/state"] = gripper[0, -1].cpu().numpy().astype(np.float32)

        obs["observation/prompt"] = self._prompt

        # --- call server ---
        response = self._client.infer(_remap_obs(obs, self._server_type))
        actions = np.array(response["actions"], dtype=np.float32)  # (action_horizon, action_dim)
        print(f"[debug] raw actions[0]: {actions[0]}", flush=True)

        # take the first prediction_len steps
        actions = actions[:prediction_len]  # (prediction_len, action_dim)

        # normalize quaternion (indices 3:7) to unit length
        quat = actions[:, 3:7]
        quat_norm = np.linalg.norm(quat, axis=-1, keepdims=True)
        actions[:, 3:7] = quat / np.clip(quat_norm, 1e-6, None)

        return torch.from_numpy(actions).float().unsqueeze(0)  # (1, prediction_len, action_dim)


if __name__ == "__main__":
    SOCKET_PATH = os.environ.get("POLICY_SOCKET_PATH", "/tmp/policy.sock")
    SERVER_TYPE = os.environ.get("POLICY_SERVER_TYPE", None)

    args = get_config(
        overrides=sys.argv[1:],
        config_name="config",
        config_path=get_config_path(),
    )

    _script_dir = Path(__file__).resolve().parent
    if args.eval_data_dir is not None and str(args.data_dir) == "demos":
        args.data_dir = args.eval_data_dir
    if args.data_dir is not None and not args.data_dir.is_absolute():
        args.data_dir = _script_dir / args.data_dir
    if args.output_file is not None and not args.output_file.is_absolute():
        args.output_file = _script_dir / args.output_file

    print(f"Policy socket: {SOCKET_PATH}")
    print("Arguments:")
    for k, v in sorted(vars(args).items()):
        print(f"  {k}: {v}")
    print("-" * 100)

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    progress_file = str(args.output_file).replace(".json", ".progress.json")

    if os.path.exists(args.output_file):
        print(f"[skip] output file already exists: {args.output_file}", flush=True)
        sys.exit(0)

    # Bimanual vs single-arm utils (RLBenchEnv only — Actioner is replaced)
    if args.bimanual:
        from online_evaluation_rlbench.utils_with_bimanual_rlbench import RLBenchEnv
    elif "orbital" in args.dataset.lower():
        from online_evaluation_rlbench.utils_with_orbital_rlbench import RLBenchEnv
    elif "peract" in args.dataset.lower():
        from online_evaluation_rlbench.utils_with_rlbench import RLBenchEnv
    else:
        from online_evaluation_rlbench.utils_with_hiveformer_rlbench import RLBenchEnv

    dataset_class = fetch_dataset_class(args.dataset)

    actioner = WebSocketActioner(socket_path=SOCKET_PATH, server_type=SERVER_TYPE)

    task_success_rates = {}
    for task_str in [args.task]:

        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

        if "orbital" in args.dataset.lower():
            _env_extra = dict(
                cameras_file=str(args.cameras_file),
                task_group_mapping_file=str(args.task_group_mapping_file),
                fov_deg=float(args.fov_deg),
                orbital_miscal_noise_level=getattr(args, "orbital_miscal_noise_level", None),
                miscal_rot_level=getattr(args, "miscal_rot_level", None),
                miscal_trans_level=getattr(args, "miscal_trans_level", None),
                camera_groups=[g.strip() for g in args.camera_groups.split(",")] if args.camera_groups else None,
                spawn_camera_group=args.spawn_camera_group if args.spawn_camera_group else None,
            )
        elif "peract" in args.dataset.lower():
            _env_extra = dict(use_depth2cloud=args.eval_use_depth2cloud)
        else:
            _env_extra = dict()

        print(args.data_dir)
        env = RLBenchEnv(
            data_path=args.data_dir,
            task_str=task_str,
            image_size=[int(x) for x in args.image_size.split(",")],
            apply_rgb=True,
            apply_pc=True,
            headless=bool(args.headless),
            apply_cameras=dataset_class.cameras,
            collision_checking=bool(args.collision_checking),
            **_env_extra,
        )

        var_success_rates = env.evaluate_task_on_multiple_variations(
            task_str,
            max_steps=args.max_steps,
            actioner=actioner,
            max_tries=args.max_tries,
            prediction_len=args.prediction_len,
            num_history=args.num_history,
            save_trajectory=args.save_trajectory,
            save_video=args.save_video,
            output_file=args.output_file,
            progress_file=progress_file,
        )
        print()
        print(f"{task_str} variation success rates:", round_floats(var_success_rates))
        print(f"{task_str} mean success rate:", round_floats(var_success_rates["mean"]))

        task_success_rates[task_str] = var_success_rates
        with open(args.output_file, "w") as f:
            json.dump(round_floats(task_success_rates), f, indent=4)
        if os.path.exists(progress_file):
            os.remove(progress_file)
