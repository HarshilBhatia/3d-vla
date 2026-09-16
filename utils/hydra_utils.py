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
def _normalize_public_vocabulary(out: dict) -> None:
    """No-op retained as the hook for future config-key normalisation.

    The legacy/public alias layer is gone: every setting now has exactly one
    name. Configs saved inside older checkpoints are translated forward by
    :mod:`utils.config_migrations` instead, which keeps the mapping in one place
    and versioned rather than resolving two spellings at read time.
    """


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


def normalize_public_vocabulary_args(args) -> None:
    """No-op retained as the hook for post-overlay config normalisation.

    The legacy/public alias layer is gone: every setting has exactly one name.
    Configs saved inside older checkpoints are translated forward by
    :mod:`utils.config_migrations` instead, so the mapping lives in one
    versioned place rather than being re-resolved on every read.
    """


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
            "mode": args.miscal_mode,
            "camera_ids": args.miscal_cameras,
            "camera_groups": args.miscal_camera_groups,
            "group_level": args.miscal_group_level,
            "group_file": str(args.miscal_group_file) if args.miscal_group_file else None,
        },
        "perturbation_noise": {
            "rotation_deg": args.perturbation_noise_rot_deg,
            "translation_m": args.perturbation_noise_trans_m,
            "fixed_rotation_deg": args.perturbation_noise_fixed_rot_deg,
            "fixed_translation_m": args.perturbation_noise_fixed_trans_m,
            "composition": "T_applied = T_sampled @ T_group @ T_true",
        },
        "ee_aux": {
            "enabled": args.ee_aux,
            "weight": args.ee_aux_weight,
            "camera_ids": args.ee_aux_cameras,
        },
        "causal_camera_history": {
            "enabled": args.video_deltam,
            "depth": args.video_deltam_depth,
            "full_image": args.video_deltam_full_image,
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
            "video_deltam": args.video_deltam,
        },
        "effective_runtime": {
            "visual_num_history": args.visual_num_history,
            "proprio_num_history": args.proprio_num_history,
            "eval_proprio_history_order": args.eval_proprio_history_order,
            "scene_sampling": args.scene_sampling,
            "prediction_len": args.prediction_len,
            "max_steps": args.max_steps,
            "max_tries": args.max_tries,
            "eval_use_depth2cloud": args.eval_use_depth2cloud,
            "collision_checking": args.collision_checking,
        },
    }
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
