"""Canonical online-evaluation orchestration.

Simulator-specific implementations remain behind the RLBench adapter while
this module owns reproducibility, paths, calibration routing, and results.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Callable

import numpy as np

from evaluation.artifacts import write_json_atomic
from evaluation.online.rlbench import RLBenchRequest, load_backend


def resolve_runtime_paths(args: Any, base_dir: str | Path) -> None:
    """Resolve legacy relative evaluation paths without changing their meaning."""
    base_dir = Path(base_dir)
    if getattr(args, "eval_data_dir", None) is not None and str(args.data_dir) == "demos":
        args.data_dir = args.eval_data_dir
    if getattr(args, "data_dir", None) is not None:
        data_dir = Path(args.data_dir)
        args.data_dir = data_dir if data_dir.is_absolute() else base_dir / data_dir
    if getattr(args, "output_file", None) is not None:
        output_file = Path(args.output_file)
        args.output_file = output_file if output_file.is_absolute() else base_dir / output_file


def build_environment_kwargs(args: Any) -> dict[str, Any]:
    """Build backend-specific environment options from an explicit eval config."""
    dataset = args.dataset.lower()
    if args.bimanual and "orbital" in dataset:
        registry = getattr(args, "eval_calibration_registry", None) or None
        calibration_id = getattr(args, "eval_calibration_id", None) or None
        if (registry is None) != (calibration_id is None):
            raise ValueError("eval_calibration_registry and eval_calibration_id must be supplied together")
        materialized = registry is not None
        return {
            "cameras_file": str(args.cameras_file),
            "spawn_camera_group": args.spawn_camera_group,
            "fov_deg": float(args.fov_deg),
            "orbital_miscal_noise_level": None if materialized else getattr(args, "orbital_miscal_noise_level", None),
            "orbital_miscal_noise_file": None if materialized else (getattr(args, "orbital_miscal_noise_file", None) or None),
            "eval_miscal_rot_level": None if materialized else getattr(args, "eval_miscal_rot_level", None),
            "eval_miscal_trans_level": None if materialized else getattr(args, "eval_miscal_trans_level", None),
            "miscal_camera_indices": None if materialized else getattr(args, "miscal_camera_indices", None),
            "calibration_registry": registry,
            "calibration_id": calibration_id,
            "scene_frame": getattr(args, "scene_frame", "world"),
            "blackout_cameras": list(getattr(args, "eval_blackout_cameras", None) or []),
            **({"clean_proprio": True} if getattr(args, "eval_clean_proprio", False) else {}),
            **({"controller": args.eval_controller} if getattr(args, "eval_controller", "planner") != "planner" else {}),
        }
    if "orbital" in dataset:
        return {
            "cameras_file": str(args.cameras_file),
            "task_group_mapping_file": str(args.task_group_mapping_file),
            "fov_deg": float(args.fov_deg),
            "orbital_miscal_noise_level": getattr(args, "orbital_miscal_noise_level", None),
            "eval_miscal_rot_level": getattr(args, "eval_miscal_rot_level", None),
            "eval_miscal_trans_level": getattr(args, "eval_miscal_trans_level", None),
            "camera_groups": [group.strip() for group in args.camera_groups.split(",")] if args.camera_groups else None,
            "spawn_camera_group": args.spawn_camera_group if args.spawn_camera_group else None,
            "scene_frame": getattr(args, "scene_frame", "world"),
        }
    if "peract" in dataset:
        registry = getattr(args, "eval_calibration_registry", None) or None
        calibration_id = getattr(args, "eval_calibration_id", None) or None
        # The standard rig only takes a materialized delta, and only when it is not the identity "calibrated" one.
        extra = {"calibration_registry": registry, "calibration_id": calibration_id} if args.bimanual and registry and calibration_id != "calibrated" else {}
        return {"use_depth2cloud": args.eval_use_depth2cloud, **extra}
    return {}


def run_online_evaluation(args: Any, *, model_loader: Callable[[Any], Any]) -> bool:
    """Execute one online-evaluation task and return whether work was performed."""
    import torch

    from datasets import fetch_dataset_class
    from utils.common_utils import round_floats
    from utils.hydra_utils import write_eval_manifest

    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    progress_file = str(output_file).replace(".json", ".progress.json")
    if output_file.exists():
        print(f"[skip] output file already exists: {output_file}", flush=True)
        return False

    model = model_loader(args)
    # Resolve after loading: the checkpoint's saved dataset (e.g. OrbitalPeract2NoWrist) owns the camera set.
    backend = load_backend(RLBenchRequest(args.dataset, bool(args.bimanual)))
    dataset_class = fetch_dataset_class(args.dataset)
    write_eval_manifest(args, output_file.with_suffix(".manifest.json"))
    print("workspace_normalizer:", model.workspace_normalizer)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    task = args.task
    environment = backend.RLBenchEnv(
        data_path=args.data_dir,
        task_str=task,
        image_size=[int(value) for value in args.image_size.split(",")],
        apply_rgb=True,
        apply_pc=True,
        headless=bool(args.headless),
        apply_cameras=dataset_class.cameras,
        collision_checking=bool(args.collision_checking),
        **build_environment_kwargs(args),
        **({"model_camera_indices": dataset_class.camera_inds} if getattr(dataset_class, "camera_inds", None) is not None else {}),
    )
    text_backbone = getattr(args, "text_backbone", None) or args.backbone
    actioner = backend.Actioner(model, backbone=text_backbone, cfg_scale=getattr(args, "cfg_scale", None))
    evaluation_extra = {}
    if getattr(args, "num_demos_total", None) is not None:
        evaluation_extra["num_demos_total"] = int(args.num_demos_total)
    if getattr(args, "eval_history_probe", "none") != "none":
        evaluation_extra["history_probe"] = args.eval_history_probe
    if getattr(args, "eval_execute_steps", None) is not None:
        evaluation_extra["execute_steps"] = int(args.eval_execute_steps)
    if getattr(args, "eval_proprio_stride", None) is not None:
        evaluation_extra["proprio_stride"] = int(args.eval_proprio_stride)
    if getattr(args, "visual_history_dense", False):
        evaluation_extra["history_every_step"] = True
    if getattr(args, "visual_history_sampling", "consecutive") != "consecutive":
        evaluation_extra["visual_history_sampling"] = args.visual_history_sampling
        evaluation_extra["history_seed"] = int(args.seed)
    success_rates = environment.evaluate_task_on_multiple_variations(
        task,
        max_steps=args.max_steps,
        actioner=actioner,
        max_tries=args.max_tries,
        prediction_len=args.prediction_len,
        visual_num_history=args.visual_num_history,
        proprio_num_history=getattr(args, "proprio_num_history", args.visual_num_history),
        proprio_history_order=args.eval_proprio_history_order,
        save_trajectory=args.save_trajectory,
        save_video=args.save_video,
        output_file=output_file,
        progress_file=progress_file,
        num_demos=getattr(args, "num_demos", None),
        **evaluation_extra,
    )
    print(f"{task} variation success rates:", round_floats(success_rates))
    print(f"{task} mean success rate:", round_floats(success_rates["mean"]))
    write_json_atomic(output_file, {task: round_floats(success_rates)})
    if os.path.exists(progress_file) and not os.environ.get("EVAL_DEMO_SHARD"):  # shards keep per-demo results for aggregation
        os.remove(progress_file)
    return True
