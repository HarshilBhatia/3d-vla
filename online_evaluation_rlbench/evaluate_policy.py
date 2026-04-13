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
from utils.common_utils import round_floats
from utils.hydra_utils import get_config, get_config_path


def load_models(args):
    print("Loading model from", args.checkpoint, flush=True)

    model_class = fetch_model_class(args.model_type)
    model = model_class(
        backbone=args.backbone,
        finetune_backbone=args.finetune_backbone,
        finetune_text_encoder=args.finetune_text_encoder,
        num_vis_instr_attn_layers=args.num_vis_instr_attn_layers,
        fps_subsampling_factor=args.fps_subsampling_factor,
        embedding_dim=args.embedding_dim,
        num_attn_heads=args.num_attn_heads,
        nhist=args.num_history,
        nhand=2 if args.bimanual else 1,
        num_shared_attn_layers=args.num_shared_attn_layers,
        relative=args.relative_action,
        rotation_format=args.rotation_format,
        denoise_timesteps=args.denoise_timesteps,
        denoise_model=args.denoise_model,
        learn_extrinsics=args.learn_extrinsics,
        traj_scene_rope=args.traj_scene_rope,
        sa_blocks_use_rope=args.sa_blocks_use_rope,
        predict_extrinsics=args.predict_extrinsics,
        extrinsics_prediction_mode=args.extrinsics_prediction_mode,
        dynamic_rope_from_camtoken=args.dynamic_rope_from_camtoken,
        rope_type=args.rope_type,
        use_com_rope=args.use_com_rope,
        com_rope_block_size=args.com_rope_block_size,
        com_rope_num_axes=args.com_rope_num_axes,
        com_rope_init_std=args.com_rope_init_std,
    )

    # Load model weights
    model_dict = torch.load(
        args.checkpoint, map_location="cpu", weights_only=True
    )

    model_dict_weight = {}
    for key in model_dict["weight"]:
        _key = key[7:]
        model_dict_weight[_key] = model_dict["weight"][key]
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
    # print(model.workspace_normalizer)

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
            )
        else:
            _env_extra = dict(
                use_front_camera_frame=args.use_front_camera_frame,
                pc_rotate_by_front_camera=args.pc_rotate_by_front_camera,
            )

        # Load RLBench environment
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
            num_history=args.num_history
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
