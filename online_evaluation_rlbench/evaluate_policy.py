"""Online evaluation script on RLBench."""

import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

from datasets import fetch_dataset_class
from modeling.policy import fetch_model_class
from modeling.policy.construction import (
    assert_model_kwargs_complete,
    build_model_kwargs,
)
from utils.common_utils import round_floats
from utils.hydra_utils import get_config, get_config_path


# These keys belong to the eval invocation, not the model — never overridden from checkpoint.
_EVAL_RUNTIME_KEYS = frozenset({
    "checkpoint", "data_dir", "eval_data_dir", "output_file",
    "task", "headless", "max_tries", "seed",
    "cameras_file", "task_group_mapping_file", "camera_groups",
    "orbital_miscal_noise_level", "miscal_rot_level", "miscal_trans_level", "fov_deg",
    "num_demos", "num_demos_total",
    # The trainer used to drop image_space_sampling from model_kwargs, so a
    # checkpoint's saved value does not reliably describe the sampler it was
    # trained with. The caller must state the sampler explicitly at eval time.
    "image_space_sampling",
    "spawn_camera_group",
    "val_instructions", "log_dir", "base_log_dir",
    "save_video", "save_trajectory",
    # PerAct online-eval runtime controls
    "eval_use_depth2cloud", "image_size", "collision_checking",
    "cfg_scale", "prediction_len", "max_steps",
})


def load_models(args):
    print("Loading model from", args.checkpoint, flush=True)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    # Overlay saved training config onto args for all non-eval keys so the
    # caller doesn't need to pass model arch flags on the CLI.
    ckpt_cfg = ckpt.get("config", {})
    if ckpt_cfg:
        # Emit a compact provenance summary to catch train/eval mixups early.
        loaded_from_ckpt = {}
        for k, v in ckpt_cfg.items():
            if k not in _EVAL_RUNTIME_KEYS:
                setattr(args, k, v)
                loaded_from_ckpt[k] = v
        print("Arguments loaded from checkpoint:")
        for k, v in sorted(loaded_from_ckpt.items()):
            print(f"  {k}: {v}")
        print("-" * 100, flush=True)
        # Runtime-vs-checkpoint consistency warnings (non-fatal).
        if str(getattr(args, "dataset", "")) != str(ckpt_cfg.get("dataset", "")):
            print(
                f"[warn] runtime dataset={args.dataset} differs from ckpt dataset={ckpt_cfg.get('dataset')}"
            )
    else:
        raise ValueError("model missing config")

    model_class = fetch_model_class(args.model_type)
    # Same helper the trainer uses, so eval-time construction cannot diverge.
    model_kwargs = build_model_kwargs(args, model_class)
    assert_model_kwargs_complete(args, model_class, model_kwargs)
    model = model_class(**model_kwargs)

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
    progress_file = str(args.output_file).replace('.json', '.progress.json')

    if os.path.exists(args.output_file):
        print(f"[skip] output file already exists: {args.output_file}", flush=True)
        sys.exit(0)

    # Bimanual vs single-arm utils
    if args.bimanual and "orbital" in args.dataset.lower():
        from online_evaluation_rlbench.utils_with_orbital_bimanual_rlbench import RLBenchEnv, Actioner
    elif args.bimanual:
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
    # Evaluate - reload environment for each task (crashes otherwise)
    task_success_rates = {}
    for task_str in [args.task]:

        # Seeds - re-seed for each task
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

        # Per-backend extra kwargs
        if args.bimanual and "orbital" in args.dataset.lower():
            # Scene state comes from the stored test demo; the camera group is an
            # independent choice, so it must be named explicitly.
            _env_extra = dict(
                cameras_file=str(args.cameras_file),
                spawn_camera_group=args.spawn_camera_group,
                fov_deg=float(args.fov_deg),
                miscal_rot_level=getattr(args, "miscal_rot_level", None),
                miscal_trans_level=getattr(args, "miscal_trans_level", None),
            )
        elif "orbital" in args.dataset.lower():
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
            _env_extra = dict(
                use_depth2cloud=args.eval_use_depth2cloud,
            )
        elif args.bimanual:
            _env_extra = dict()
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
        # When backbone=dino, text_backbone=clip selects the text tokenizer; fall back to backbone.
        _text_backbone = getattr(args, "text_backbone", None) or args.backbone
        actioner = Actioner(model, backbone=_text_backbone, cfg_scale=getattr(args, "cfg_scale", None))

        # Evaluate
        _eval_extra = {}
        # A whole-task episode budget: only the bimanual harness spreads it over
        # variations, so pass it only where it is understood.
        if getattr(args, "num_demos_total", None) is not None:
            _eval_extra["num_demos_total"] = int(args.num_demos_total)
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
            num_demos=getattr(args, "num_demos", None),
            **_eval_extra,
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
        if os.path.exists(progress_file):
            os.remove(progress_file)
