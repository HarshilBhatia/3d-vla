"""Checkpoint compatibility rules shared by online-evaluation entry points."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


# These values describe an evaluation invocation. A saved training config must
# never override them when a checkpoint is loaded for rollout.
EVALUATION_RUNTIME_KEYS = frozenset({
    "checkpoint", "data_dir", "eval_data_dir", "output_file",
    "task", "headless", "max_tries", "seed",
    "cameras_file", "task_group_mapping_file", "camera_groups",
    "orbital_miscal_noise_level", "orbital_miscal_noise_file",
    "miscal_rot_level", "miscal_trans_level", "fov_deg",
    "miscal_camera_indices", "eval_protocol", "eval_viewpoint_regime",
    "eval_calibration_id", "eval_calibration_registry", "num_demos",
    "num_demos_total", "image_space_sampling", "spawn_camera_group",
    "val_instructions", "log_dir", "base_log_dir", "save_video",
    "save_trajectory", "eval_use_depth2cloud", "image_size",
    "collision_checking", "cfg_scale", "prediction_len", "max_steps",
    "eval_proprio_history_order",
})


def _non_frozen_checkpoint_incompatibilities(model, incompatible):
    """Return checkpoint mismatches that could change model behavior.

    Frozen pretrained visual/text modules are deliberately omitted from some
    rollout checkpoints.  Every other mismatch is an architecture error and
    must never be hidden by ``strict=False``.
    """
    frozen_prefixes = ("encoder.backbone.", "encoder.text_encoder.", "encoder.normalize.")
    return (
        [key for key in incompatible.missing_keys if not key.startswith(frozen_prefixes)],
        [key for key in incompatible.unexpected_keys if not key.startswith(frozen_prefixes)],
    )


def overlay_checkpoint_config(args: Any, checkpoint_config: Mapping[str, Any]) -> dict[str, Any]:
    """Overlay model configuration while preserving evaluation-owned fields.

    Returns exactly the values loaded from the checkpoint, which callers can
    log into an evaluation manifest or console provenance summary.
    """
    loaded: dict[str, Any] = {}
    for key, value in checkpoint_config.items():
        if key not in EVALUATION_RUNTIME_KEYS:
            setattr(args, key, value)
            loaded[key] = value
    return loaded


def load_model_for_evaluation(args: Any):
    """Load a checkpoint using its saved architecture configuration.

    Heavy dependencies are imported here, rather than at module import time,
    so protocol and artifact tooling stays usable outside the simulator image.
    """
    import torch

    from modeling.policy import fetch_model_class
    from modeling.policy.construction import assert_model_kwargs_complete, build_model_kwargs
    from utils.hydra_utils import normalize_public_vocabulary_args

    print("Loading model from", args.checkpoint, flush=True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint_config = checkpoint.get("config", {})
    if not checkpoint_config:
        raise ValueError("model missing config")

    loaded = overlay_checkpoint_config(args, checkpoint_config)
    normalize_public_vocabulary_args(args)
    print("Arguments loaded from checkpoint:")
    for key, value in sorted(loaded.items()):
        print(f"  {key}: {value}")
    print("-" * 100, flush=True)
    if str(getattr(args, "dataset", "")) != str(checkpoint_config.get("dataset", "")):
        print(
            f"[warn] runtime dataset={args.dataset} differs from "
            f"ckpt dataset={checkpoint_config.get('dataset')}"
        )

    model_class = fetch_model_class(args.model_type)
    model_kwargs = build_model_kwargs(args, model_class)
    assert_model_kwargs_complete(args, model_class, model_kwargs)
    model = model_class(**model_kwargs)
    state = {
        key[7:]: value
        for key, value in checkpoint["weight"].items()
    }
    incompatible = model.load_state_dict(state, strict=False)
    missing, unexpected = _non_frozen_checkpoint_incompatibilities(model, incompatible)
    if missing or unexpected:
        raise RuntimeError(
            "checkpoint/model architecture mismatch; refusing an invalid evaluation. "
            f"Missing trained keys: {missing}; unexpected checkpoint keys: {unexpected}"
        )
    model.eval()
    return model.cuda()
