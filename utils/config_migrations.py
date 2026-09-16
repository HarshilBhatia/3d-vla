"""Forward-migration of configs saved inside older checkpoints.

Every checkpoint stores the config it was trained with, and eval rebuilds the
model from it. When a key is renamed or retired, an old checkpoint's config
would otherwise fall back to the current default: a wrong architecture, or --
for flags that do not change parameter shapes, such as the scene-sampling
switches -- a silent accuracy regression no state-dict check can catch.

So every config change that retires or renames a key adds one @migration here.
They run in order and must each be idempotent, so a checkpoint from any era
arrives at today's vocabulary.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable

# Every config key this project has retired. The migrations below translate
# them forward; tests use this set to prove no launch script still passes one.
RETIRED_KEYS = frozenset({
    # head positional encoding
    "traj_scene_rope", "use_learned_abs_pe", "use_proprio_rope",
    # RoPE backward variants and their schedule
    "rope_type", "rope_schedule_type", "rope_schedule_start_k",
    "rope_schedule_end_k", "rope_schedule_steps",
    # scene-token sampling
    "skip_fps", "image_space_sampling", "position_based_sampling",
    # miscalibration / perturbation noise
    "miscal_max_angle_deg", "miscal_max_translation_m", "miscal_camera_ids",
    "orbital_miscal_noise_levels", "cotrain_miscal_group_ids",
    "cotrain_miscal_level", "cotrain_miscal_levels", "group_miscal_level",
    "group_miscal_file", "sampled_miscal_max_rot_deg", "sampled_miscal_max_trans_m",
    "miscal_rot_level", "miscal_trans_level",
    "noise_curriculum", "noise_curriculum_warmup_frac",
    # view alignment / EE aux / video-deltaM aliases
    "predict_extrinsics", "extrinsics_prediction_mode", "delta_m_camera_ids",
    "dynamic_rope_from_camtoken", "predict_ee_aux", "lambda_aux", "ee_aux_cam_ids",
    "causal_cam_history", "causal_cam_history_depth",
    # renamed for clarity
    "num_history", "val_freq", "last_ckpt_freq", "interm_ckpt_freq",
    # upstream delta_M refiner, never enabled in any config
    "use_recursive_set_encoder", "recursive_set_encoder_num_layers",
    "recursive_set_encoder_ncam",
    # never wired to anything
    "keep_last_k",
})

MIGRATIONS: list[Callable[[dict[str, Any]], None]] = []


def migration(fn: Callable[[dict[str, Any]], None]) -> Callable[[dict[str, Any]], None]:
    MIGRATIONS.append(fn)
    return fn


@migration
def _head_positional_encoding(cfg: dict[str, Any]) -> None:
    """2026-09: traj_scene_rope + use_learned_abs_pe + use_proprio_rope -> head_positional_encoding."""
    if "head_positional_encoding" not in cfg and "traj_scene_rope" in cfg:
        if not cfg["traj_scene_rope"]:
            mode = "none"
        elif cfg.get("use_learned_abs_pe", False):
            mode = "learned_abs"
        elif cfg.get("use_proprio_rope", False):
            mode = "rope3d_proprio"
        else:
            mode = "rope3d"
        cfg["head_positional_encoding"] = mode
    for key in ("traj_scene_rope", "use_learned_abs_pe", "use_proprio_rope"):
        cfg.pop(key, None)


@migration
def _retire_rope_type_and_schedule(cfg: dict[str, Any]) -> None:
    """2026-09: only 'normal' RoPE remains; the adam/stopgrad paths and their schedule are gone."""
    for key in ("rope_type", "rope_schedule_type", "rope_schedule_start_k",
                "rope_schedule_end_k", "rope_schedule_steps"):
        cfg.pop(key, None)


@migration
def _scene_sampling(cfg: dict[str, Any]) -> None:
    """2026-09: skip_fps + image_space_sampling + position_based_sampling -> scene_sampling."""
    if "scene_sampling" not in cfg and "image_space_sampling" in cfg:
        if cfg.get("skip_fps", False):
            mode = "none"
        elif cfg["image_space_sampling"]:
            mode = "image_space"
        elif cfg.get("position_based_sampling", False):
            mode = "fps_xyz"
        else:
            mode = "fps"
        cfg["scene_sampling"] = mode
    for key in ("skip_fps", "image_space_sampling", "position_based_sampling"):
        cfg.pop(key, None)


@migration
def _retire_keep_last_k(cfg: dict[str, Any]) -> None:
    """2026-09: keep_last_k was never wired to anything."""
    cfg.pop("keep_last_k", None)


@migration
def _retire_noise_curriculum(cfg: dict[str, Any]) -> None:
    """2026-09: the miscal noise curriculum was never wired to the preprocessor, so it
    never ran; every existing checkpoint trained at full noise from step 0."""
    for key in ("noise_curriculum", "noise_curriculum_warmup_frac"):
        cfg.pop(key, None)


@migration
def _miscal_and_perturbation(cfg: dict[str, Any]) -> None:
    """2026-09: the miscal keys conflated two different things. A persistent
    per-camera-group calibration error is miscalibration; a per-sample random
    jitter is train-time perturbation noise (no bias, correct on average). They
    are now separate, orthogonal axes."""
    OLD = ("miscal_max_angle_deg", "miscal_max_translation_m", "miscal_camera_ids",
           "orbital_miscal_noise_level", "orbital_miscal_noise_file",
           "orbital_miscal_noise_levels", "cotrain_miscal_group_ids",
           "cotrain_miscal_level", "cotrain_miscal_levels", "group_miscal_level",
           "group_miscal_file", "sampled_miscal_max_rot_deg", "sampled_miscal_max_trans_m")
    if "miscal_mode" in cfg or not any(k in cfg for k in OLD):
        return
    pick = lambda *names: next((cfg[n] for n in names if cfg.get(n) is not None), None)
    level = pick("group_miscal_level", "orbital_miscal_noise_level",
                 "orbital_miscal_noise_levels", "cotrain_miscal_level", "cotrain_miscal_levels")
    rot = pick("sampled_miscal_max_rot_deg", "miscal_max_angle_deg") or 0.0
    trans = pick("sampled_miscal_max_trans_m", "miscal_max_translation_m") or 0.0

    cfg["miscal_mode"] = "group" if level is not None else "none"
    cfg["miscal_group_level"] = level
    cfg["miscal_group_file"] = pick("group_miscal_file", "orbital_miscal_noise_file")
    cfg["miscal_camera_groups"] = pick("cotrain_miscal_group_ids")
    cfg["miscal_cameras"] = pick("miscal_cameras", "miscal_camera_ids")
    # The old magnitude pair was always a per-sample random draw, whether it
    # stood alone or sat on top of a fixed base. That is perturbation noise.
    cfg["perturbation_noise_rot_deg"] = rot or None
    cfg["perturbation_noise_trans_m"] = trans or None
    for key in OLD:
        cfg.pop(key, None)


@migration
def _view_align_and_ee_aux(cfg: dict[str, Any]) -> None:
    """2026-09: the public/legacy alias layer is gone. view_align_mode absorbs
    predict_extrinsics + extrinsics_prediction_mode; the ee_aux and camera-list
    keys keep only their public spelling; causal_cam_history* drop in favour of
    the video_deltam_* family, which was only half-aliased."""
    MODE = {(False, "delta_m"): "none", (False, "delta_m_full"): "none", (False, "rt"): "none",
            (True, "delta_m"): "rope_6d", (True, "delta_m_full"): "rope_full",
            (True, "rt"): "physical_se3"}
    if cfg.get("view_align_mode") is None and "predict_extrinsics" in cfg:
        cfg["view_align_mode"] = MODE[(bool(cfg["predict_extrinsics"]),
                                       cfg.get("extrinsics_prediction_mode") or "delta_m")]
    for public, legacy in (("view_align_cameras", "delta_m_camera_ids"),
                           ("layerwise_view_align", "dynamic_rope_from_camtoken"),
                           ("ee_aux", "predict_ee_aux"),
                           ("ee_aux_weight", "lambda_aux"),
                           ("ee_aux_cameras", "ee_aux_cam_ids"),
                           ("video_deltam", "causal_cam_history"),
                           ("video_deltam_depth", "causal_cam_history_depth")):
        if cfg.get(public) is None and cfg.get(legacy) is not None:
            cfg[public] = cfg[legacy]
        cfg.pop(legacy, None)
    for key in ("predict_extrinsics", "extrinsics_prediction_mode"):
        cfg.pop(key, None)


@migration
def _eval_miscal_level_prefix(cfg: dict[str, Any]) -> None:
    """2026-09: the eval-only level keys gain an eval_ prefix, so they are not
    mistaken for the training-time miscal_* family."""
    for old, new in (("miscal_rot_level", "eval_miscal_rot_level"),
                     ("miscal_trans_level", "eval_miscal_trans_level")):
        if old in cfg:
            cfg.setdefault(new, cfg[old])
            cfg.pop(old)


@migration
def _clarify_training_names(cfg: dict[str, Any]) -> None:
    """2026-09: num_history was the *visual* history while its sibling was
    explicitly proprio_num_history; the *_freq cadences are step intervals, not
    frequencies. The recursive set encoder is gone -- never enabled in any config."""
    for old, new in (("num_history", "visual_num_history"),
                     ("val_freq", "val_interval_steps"),
                     ("last_ckpt_freq", "ckpt_interval_steps"),
                     ("interm_ckpt_freq", "interm_ckpt_interval_steps")):
        if old in cfg:
            cfg.setdefault(new, cfg[old])
            cfg.pop(old)
    for key in ("use_recursive_set_encoder", "recursive_set_encoder_num_layers",
                "recursive_set_encoder_ncam"):
        cfg.pop(key, None)


def migrate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return ``config`` translated into the current key vocabulary."""
    out = dict(config)
    for fn in MIGRATIONS:
        fn(out)
    return out
