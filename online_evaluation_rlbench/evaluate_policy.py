"""Online evaluation script on RLBench."""

import json
import os
import random
import sys
from pathlib import Path

import inspect

import numpy as np
import torch

from datasets import fetch_dataset_class
from modeling.policy import fetch_model_class
from utils.common_utils import round_floats
from utils.hydra_utils import get_config, get_config_path


# These keys belong to the eval invocation, not the model — never overridden from checkpoint.
_EVAL_RUNTIME_KEYS = frozenset({
    "checkpoint", "data_dir", "eval_data_dir", "output_file",
    "task", "headless", "max_tries", "seed",
    "cameras_file", "task_group_mapping_file", "camera_groups",
    "miscalibration_noise_level", "fov_deg",
    "spawn_camera_group",
    "val_instructions", "log_dir", "base_log_dir",
    "save_video", "save_trajectory", "debug_pcd_dir",
    # PerAct online-eval runtime controls
    "eval_use_depth2cloud", "image_size", "collision_checking",
})


def load_models(args):
    print("Loading model from", args.checkpoint, flush=True)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    # Overlay saved training config onto args for all non-eval keys so the
    # caller doesn't need to pass model arch flags on the CLI.
    ckpt_cfg = ckpt.get("config", {})
    if ckpt_cfg:
        # Emit a compact provenance summary to catch train/eval mixups early.
        for k, v in ckpt_cfg.items():
            if k not in _EVAL_RUNTIME_KEYS:
                setattr(args, k, v)
        # Runtime-vs-checkpoint consistency warnings (non-fatal).
        if str(getattr(args, "dataset", "")) != str(ckpt_cfg.get("dataset", "")):
            print(
                f"[warn] runtime dataset={args.dataset} differs from ckpt dataset={ckpt_cfg.get('dataset')}"
            )
    else:
        raise ValueError("model missing config")

    model_class = fetch_model_class(args.model_type)
    # Config uses different names for a few constructor params.
    _cfg = vars(args) | {
        "nhist": args.num_history,
        "nhand": 2 if args.bimanual else 1,
        "relative": args.relative_action,
    }
    _sig = inspect.signature(model_class.__init__).parameters
    model = model_class(**{k: v for k, v in _cfg.items() if k in _sig})

    model_dict_weight = {}
    for key in ckpt["weight"]:
        _key = key[7:]
        model_dict_weight[_key] = ckpt["weight"][key]
    model.load_state_dict(model_dict_weight, strict=False)
    model.eval()

    return model.cuda()


if __name__ == "__main__":
    # Compose config from config/config.yaml + CLI overrides (e.g. checkpoint=path task=close_jar)
    args = get_config(
        overrides=sys.argv[1:],
        config_name="config",
        config_path=get_config_path(),
    )
    # Resolve relative paths relative to this script's directory
    _script_dir = Path(__file__).resolve().parent
    # Backward-compat: many wrappers still pass eval_data_dir for online eval.
    # If data_dir is left at default while eval_data_dir is overridden, use eval_data_dir.
    if args.eval_data_dir is not None and str(args.data_dir) == "demos":
        args.data_dir = args.eval_data_dir
    if args.data_dir is not None and not args.data_dir.is_absolute():
        args.data_dir = _script_dir / args.data_dir
    if args.output_file is not None and not args.output_file.is_absolute():
        args.output_file = _script_dir / args.output_file

    print("Arguments:")
    for k, v in sorted(vars(args).items()):
        print(f"  {k}: {v}")
    print("-" * 100)

    # Save results here
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # Bimanual vs single-arm utils
    if args.bimanual:
        from online_evaluation_rlbench.utils_with_bimanual_rlbench import RLBenchEnv, Actioner
    elif "orbital" in args.dataset.lower():
        from online_evaluation_rlbench.utils_with_orbital_rlbench import RLBenchEnv, Actioner
    elif "peract" in args.dataset.lower():
        from online_evaluation_rlbench.utils_with_rlbench import RLBenchEnv, Actioner
    else:
        from online_evaluation_rlbench.utils_with_hiveformer_rlbench import RLBenchEnv, Actioner

    # Dataset class (for getting cameras and tasks/variations)
    dataset_class = fetch_dataset_class(args.dataset)

    # Load models
    model = load_models(args)
    print("workspace_normalizer:", model.workspace_normalizer)
    if getattr(args, 'debug_pcd_dir', None):
        model.encoder.debug_dir = str(args.debug_pcd_dir)
        print(f"[debug] saving PCDs to {model.encoder.debug_dir}")

    # Evaluate - reload environment for each task (crashes otherwise)
    task_success_rates = {}
    for task_str in [args.task]:

        # Seeds - re-seed for each task
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

        # Per-backend extra kwargs
        if "orbital" in args.dataset.lower():
            _env_extra = dict(
                cameras_file=str(args.cameras_file),
                task_group_mapping_file=str(args.task_group_mapping_file),
                fov_deg=float(args.fov_deg),
                miscalibration_noise_level=args.miscalibration_noise_level,
                camera_groups=[g.strip() for g in args.camera_groups.split(",")] if args.camera_groups else None,
                spawn_camera_group=args.spawn_camera_group if args.spawn_camera_group else None,
            )
        elif "peract" in args.dataset.lower():
            _env_extra = dict(
                use_depth2cloud=args.eval_use_depth2cloud,
                miscalibration_noise_level=args.miscalibration_noise_level,
            )
        else:
            _env_extra = dict()

        # Load RLBench environment

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

        # Actioner (runs the policy online)
        actioner = Actioner(model, backbone=args.backbone)

        # Evaluate
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
        )
        print()
        print(
            f"{task_str} variation success rates:",
            round_floats(var_success_rates)
        )
        print(
            f"{task_str} mean success rate:",
            round_floats(var_success_rates["mean"])
        )

        task_success_rates[task_str] = var_success_rates
        with open(args.output_file, "w") as f:
            json.dump(round_floats(task_success_rates), f, indent=4)
