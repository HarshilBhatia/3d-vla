"""Compose Hydra config and return an args-like object for existing code.

Hydra returns OmegaConf; we convert to SimpleNamespace so that vars(args) and
getattr(args, k) work everywhere (e.g. wandb config=vars(args), printing).
We also convert path strings to pathlib.Path and resolve relative paths to
the project root; Hydra does not do that by default.
"""

from pathlib import Path
from types import SimpleNamespace
import json

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra


# Canonical public config vocabulary. The model/runtime API deliberately keeps
# its historical names because they occur in old configs and checkpoint state.
# Normalize here, before any trainer or model is constructed.
_VIEW_ALIGN_TO_LEGACY = {
    "none": (False, "delta_m"),
    "rope_6d": (True, "delta_m"),
    "rope_full": (True, "delta_m_full"),
    "physical_se3": (True, "rt"),
}
_LEGACY_TO_VIEW_ALIGN = {
    (False, "delta_m"): "none",
    (False, "delta_m_full"): "none",
    (False, "rt"): "none",
    (True, "delta_m"): "rope_6d",
    (True, "delta_m_full"): "rope_full",
    (True, "rt"): "physical_se3",
}


def _normalize_public_vocabulary(out: dict) -> None:
    """Resolve short public keys and legacy runtime keys bidirectionally.

    A non-null public value wins. Null public values are populated from the
    legacy setting, which makes old configs self-describing in logs/manifests.
    No model parameter or legacy key is renamed, preserving checkpoint and CLI
    compatibility.
    """
    mode = out.get("view_align_mode")
    if mode is not None:
        if mode not in _VIEW_ALIGN_TO_LEGACY:
            allowed = ", ".join(_VIEW_ALIGN_TO_LEGACY)
            raise ValueError(f"view_align_mode must be one of {allowed}, got {mode!r}")
        out["predict_extrinsics"], out["extrinsics_prediction_mode"] = _VIEW_ALIGN_TO_LEGACY[mode]
    else:
        legacy = (out.get("predict_extrinsics", False), out.get("extrinsics_prediction_mode", "delta_m"))
        out["view_align_mode"] = _LEGACY_TO_VIEW_ALIGN.get(legacy, "none")

    aliases = {
        "view_align_cameras": "delta_m_camera_ids",
        "layerwise_view_align": "dynamic_rope_from_camtoken",
        "miscal_cameras": "miscal_camera_ids",
        "group_miscal_level": "orbital_miscal_noise_level",
        "group_miscal_file": "orbital_miscal_noise_file",
        "sampled_miscal_max_rot_deg": "miscal_max_angle_deg",
        "sampled_miscal_max_trans_m": "miscal_max_translation_m",
        "ee_aux": "predict_ee_aux",
        "ee_aux_weight": "lambda_aux",
        "ee_aux_cameras": "ee_aux_cam_ids",
        "causal_cam_history": "video_deltam",
        "causal_cam_history_depth": "video_deltam_depth",
    }
    for public, legacy in aliases.items():
        if out.get(public) is not None:
            out[legacy] = out[public]
        else:
            out[public] = out.get(legacy)

def get_config_path() -> Path:
    """Return the project config directory (absolute). Use from any entry point so config path is consistent."""
    return Path(__file__).resolve().parent.parent / "config"


# Paths that get resolved relative to project root (absolute). exp_log_dir and run_log_dir
# are kept as relative so log_dir = base_log_dir / exp_log_dir / run_log_dir works.
_PATH_KEYS = frozenset({
    "train_data_dir", "eval_data_dir", "train_instructions", "val_instructions",
    "base_log_dir",
    "checkpoint",
    "data_dir", "output_file",
})


def _resolve_relative_paths(args: SimpleNamespace, base: Path) -> None:
    for k in _PATH_KEYS:
        v = getattr(args, k, None)
        if v is not None and v != "" and isinstance(v, Path) and not v.is_absolute():
            setattr(args, k, (base / v).resolve())


def _cfg_to_args(cfg, base_dir: Path = None) -> SimpleNamespace:
    from omegaconf import OmegaConf
    raw = OmegaConf.to_container(cfg, resolve=True)
    out = {}
    for k, v in raw.items():
        if k in _PATH_KEYS and v is not None and v != "":
            out[k] = Path(v) if not isinstance(v, Path) else v
        else:
            out[k] = v
    _normalize_public_vocabulary(out)
    args = SimpleNamespace(**out)
    if base_dir is not None:
        _resolve_relative_paths(args, base_dir)
    return args


def normalize_public_vocabulary_args(args: SimpleNamespace) -> None:
    """Refresh public aliases after an eval checkpoint overlays legacy config.

    Checkpoints contain legacy *model* keys. Clear only their derived public
    counterparts first; evaluation-condition aliases remain runtime-owned.
    """
    values = vars(args)
    for key in (
        "view_align_mode", "view_align_cameras", "layerwise_view_align",
        "ee_aux", "ee_aux_weight", "ee_aux_cameras",
        "causal_cam_history", "causal_cam_history_depth",
    ):
        values[key] = None
    _normalize_public_vocabulary(values)


# Config groups that use @_global_ in defaults; CLI override "group=option" must be passed as "group@_global_=option"
_GLOBAL_GROUP_OVERRIDES = ("data", "rope_mode", "experiment", "miscal")


def _normalize_overrides(overrides):
    """Convert data=x, rope_mode=y, experiment=z to data@_global_=x etc. so Hydra accepts them."""
    out = []
    for s in overrides or []:
        if "=" not in s:
            out.append(s)
            continue
        key, _, value = s.partition("=")
        if key in _GLOBAL_GROUP_OVERRIDES and "@" not in key:
            out.append(f"{key}@_global_={value}")
        else:
            out.append(s)
    return out


def get_config(
    overrides=None,
    config_name: str = "config",
    config_path: Path = None,
):
    if config_path is None:
        raise ValueError("config_path must be set (e.g. Path(__file__).parent / 'config')")
    config_path = Path(config_path).resolve()
    if not config_path.is_dir():
        raise FileNotFoundError(f"Config directory not found: {config_path}")

    normalized = _normalize_overrides(overrides)
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.1", config_dir=str(config_path)):
        cfg = compose(config_name=config_name, overrides=normalized)
    base_dir = config_path.resolve().parent
    return _cfg_to_args(cfg, base_dir=base_dir)


def write_experiment_manifest(args: SimpleNamespace, path: Path) -> None:
    """Write a compact, paper-facing description beside a new run's logs.

    This supplements the full resolved Hydra argument dump; it does not replace
    legacy config values or alter checkpoint serialization.
    """
    manifest = {
        "schema_version": 1,
        "view_alignment": {
            "mode": args.view_align_mode,
            "camera_ids": args.view_align_cameras,
            "layerwise_refinement": args.layerwise_view_align,
        },
        "miscalibration": {
            "camera_ids": args.miscal_cameras,
            "group_level": args.group_miscal_level,
            "group_file": str(args.group_miscal_file) if args.group_miscal_file else None,
            "sampled_max_rotation_deg": args.sampled_miscal_max_rot_deg,
            "sampled_max_translation_m": args.sampled_miscal_max_trans_m,
            "composition": "T_applied = T_sampled @ T_group @ T_true",
        },
        "ee_aux": {
            "enabled": args.ee_aux,
            "weight": args.ee_aux_weight,
            "camera_ids": args.ee_aux_cameras,
        },
        "causal_camera_history": {
            "enabled": args.causal_cam_history,
            "depth": args.causal_cam_history_depth,
        },
        "legacy_runtime": {
            "predict_extrinsics": args.predict_extrinsics,
            "extrinsics_prediction_mode": args.extrinsics_prediction_mode,
            "delta_m_camera_ids": args.delta_m_camera_ids,
            "dynamic_rope_from_camtoken": args.dynamic_rope_from_camtoken,
        },
    }
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def write_eval_manifest(args: SimpleNamespace, path: Path) -> None:
    """Write a task-local sidecar describing one online-evaluation cell.

    This is called after checkpoint configuration has been overlaid, so the
    runtime block captures the *effective* history layout and rollout controls
    that produced a result.  These values are required to compare a legacy
    result to a replay without relying on an external Slurm log.
    """
    manifest = {
        "schema_version": 3,
        "protocol": args.eval_protocol,
        "calibration_id": args.eval_calibration_id,
        "viewpoint_regime": args.eval_viewpoint_regime,
        "task": args.task,
        "checkpoint": str(args.checkpoint),
        "camera_group": args.spawn_camera_group,
        "episode_budget": args.num_demos_total if args.num_demos_total is not None else args.num_demos,
        "seed": args.seed,
        "calibration_realization": {
            "id": args.eval_calibration_id,
            "registry": str(getattr(args, "eval_calibration_registry", None)) if getattr(args, "eval_calibration_registry", None) else None,
            "composition": "T_observed = delta @ T_true",
        },
        "checkpoint_method": {
            "view_align_mode": args.view_align_mode,
            "view_align_cameras": args.view_align_cameras,
            "layerwise_view_align": args.layerwise_view_align,
            "ee_aux": args.ee_aux,
            "causal_cam_history": args.causal_cam_history,
        },
        "effective_runtime": {
            "num_history": args.num_history,
            "proprio_num_history": args.proprio_num_history,
            "eval_proprio_history_order": args.eval_proprio_history_order,
            "image_space_sampling": args.image_space_sampling,
            "prediction_len": args.prediction_len,
            "max_steps": args.max_steps,
            "max_tries": args.max_tries,
            "eval_use_depth2cloud": args.eval_use_depth2cloud,
            "collision_checking": args.collision_checking,
        },
    }
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
