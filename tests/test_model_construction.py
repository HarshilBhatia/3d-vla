"""Tests for the shared model-kwargs construction helper.

Covers signature filtering, the explicit rename/derivation map, the startup
guardrail, and behavioral compatibility with the trainer's old hand-written dict
(reproduced here as a fixture) for the shipped config defaults.
"""

import inspect
from types import SimpleNamespace

import pytest

from modeling.policy import fetch_model_class
from modeling.policy.construction import (
    MODEL_KWARG_MAP,
    assert_model_kwargs_complete,
    build_model_kwargs,
    model_signature_params,
)
from utils.hydra_utils import get_config, get_config_path


# --------------------------------------------------------------------------
# The trainer's model_kwargs dict exactly as it stood after eb06b4c, frozen as
# a fixture. The new helper must reproduce it (plus only intentional additions).
# --------------------------------------------------------------------------
def legacy_trainer_model_kwargs(args):
    model_kwargs = dict(
        backbone=args.backbone,
        text_backbone=getattr(args, 'text_backbone', None),
        finetune_backbone=args.finetune_backbone,
        finetune_text_encoder=args.finetune_text_encoder,
        num_vis_instr_attn_layers=args.num_vis_instr_attn_layers,
        fps_subsampling_factor=args.fps_subsampling_factor,
        position_based_sampling=args.position_based_sampling,
        image_space_sampling=args.image_space_sampling,
        skip_fps=args.skip_fps,
        use_proprio_rope=args.use_proprio_rope,
        embedding_dim=args.embedding_dim,
        num_attn_heads=args.num_attn_heads,
        nhist=args.num_history,
        nhand=2 if args.bimanual else 1,
        num_shared_attn_layers=args.num_shared_attn_layers,
        relative=args.relative_action,
        rotation_format=args.rotation_format,
        denoise_timesteps=args.denoise_timesteps,
        denoise_model=args.denoise_model,
        lv2_batch_size=args.lv2_batch_size,
        traj_scene_rope=args.traj_scene_rope,
    )
    for _key in (
        'predict_extrinsics', 'extrinsics_prediction_mode',
        'dynamic_rope_from_camtoken', 'rope_type',
        'use_recursive_set_encoder', 'recursive_set_encoder_num_layers',
        'recursive_set_encoder_ncam',
        'lang_dropout_prob',
        'predict_ee_aux', 'lambda_aux', 'ee_aux_cam_ids',
    ):
        if hasattr(args, _key):
            model_kwargs[_key] = getattr(args, _key)
    return model_kwargs


# The one parameter the old trainer dict dropped: it is in config/config.yaml
# and in DenoiseActor3D.__init__, but was never forwarded. This is the same class
# of bug as image_space_sampling, found by the audit.
KNOWN_LEGACY_DROPS = {"use_learned_abs_pe"}


@pytest.fixture(scope="module")
def default_args():
    return get_config(overrides=[], config_name="config", config_path=get_config_path())


# --------------------------------------------------------------------------
# Signature filtering
# --------------------------------------------------------------------------
def test_signature_filter_drops_non_constructor_keys(default_args):
    model_class = fetch_model_class("denoise3d")
    kwargs = build_model_kwargs(default_args, model_class)
    sig = set(model_signature_params(model_class))
    assert set(kwargs) <= sig
    # loader/logging keys must never reach the constructor
    for k in ("num_workers", "batch_size", "wandb_project", "train_data_dir", "lr"):
        assert k not in kwargs


def test_denoise2d_gets_only_its_own_params(default_args):
    """A narrower model must not receive 3d-only flags."""
    kwargs = build_model_kwargs(default_args, fetch_model_class("denoise2d"))
    for k in ("image_space_sampling", "rope_type", "predict_extrinsics", "traj_scene_rope"):
        assert k not in kwargs
    # and it must actually construct-check as a valid call signature
    inspect.signature(fetch_model_class("denoise2d").__init__).bind(
        None, **kwargs
    )


# --------------------------------------------------------------------------
# Rename / derivation map
# --------------------------------------------------------------------------
def test_rename_map(default_args):
    kwargs = build_model_kwargs(default_args, fetch_model_class("denoise3d"))
    assert kwargs["nhist"] == default_args.num_history
    assert kwargs["relative"] == default_args.relative_action
    assert kwargs["nhand"] == (2 if default_args.bimanual else 1)
    # the config-side names are not passed through under their own names
    assert "num_history" not in kwargs
    assert "relative_action" not in kwargs
    assert "bimanual" not in kwargs


def test_nhand_derivation_both_ways():
    model_class = fetch_model_class("denoise3d")
    base = dict(num_history=2, relative_action=False, bimanual=False)
    assert build_model_kwargs(SimpleNamespace(**base), model_class)["nhand"] == 1
    base["bimanual"] = True
    assert build_model_kwargs(SimpleNamespace(**base), model_class)["nhand"] == 2


def test_rename_source_missing_raises():
    args = SimpleNamespace(relative_action=False, bimanual=False)  # no num_history
    with pytest.raises(KeyError, match="num_history"):
        build_model_kwargs(args, fetch_model_class("denoise3d"))


# --------------------------------------------------------------------------
# Guardrail
# --------------------------------------------------------------------------
def test_guardrail_passes_on_shipped_config(default_args):
    model_class = fetch_model_class("denoise3d")
    assert_model_kwargs_complete(default_args, model_class)


def test_guardrail_raises_on_dropped_key(default_args):
    """Reproduce the image_space_sampling failure mode: config key silently dropped."""
    model_class = fetch_model_class("denoise3d")
    kwargs = build_model_kwargs(default_args, model_class)
    kwargs.pop("image_space_sampling")
    with pytest.raises(ValueError, match="image_space_sampling"):
        assert_model_kwargs_complete(default_args, model_class, kwargs)


def test_guardrail_raises_on_uncovered_constructor_param(default_args):
    """A constructor param with no config key and no allowlist entry must raise."""

    class ModelWithNewFlag:
        def __init__(self, embedding_dim=60, brand_new_flag=False):
            pass

    with pytest.raises(ValueError, match="brand_new_flag"):
        assert_model_kwargs_complete(default_args, ModelWithNewFlag)


def test_guardrail_allows_documented_defaults(default_args):
    """denoise2d's legacy RoPE-ΔM knobs are allowlisted, so it must pass."""
    assert_model_kwargs_complete(default_args, fetch_model_class("denoise2d"))


# --------------------------------------------------------------------------
# Behavioral compatibility with the post-eb06b4c trainer
# --------------------------------------------------------------------------
def test_kwargs_match_legacy_trainer_dict(default_args):
    model_class = fetch_model_class("denoise3d")
    legacy = legacy_trainer_model_kwargs(default_args)
    new = build_model_kwargs(default_args, model_class)

    assert set(legacy) - set(new) == set(), "new helper dropped a legacy kwarg"
    added = set(new) - set(legacy)
    assert added == KNOWN_LEGACY_DROPS, f"unexpected new kwargs: {added - KNOWN_LEGACY_DROPS}"
    for k in legacy:
        assert new[k] == legacy[k], f"value differs for {k}: {legacy[k]} -> {new[k]}"

    # The only added kwarg must equal the constructor default at shipped config
    # values, so existing runs are byte-identical.
    defaults = inspect.signature(model_class.__init__).parameters
    for k in added:
        assert new[k] == defaults[k].default, (
            f"{k} config value {new[k]} differs from constructor default "
            f"{defaults[k].default}; construction is NOT compatible"
        )


@pytest.mark.slow
def test_state_dict_shapes_identical(default_args):
    """Instantiate both ways; state dicts must match key-for-key and shape-for-shape."""
    model_class = fetch_model_class("denoise3d")
    old = model_class(**legacy_trainer_model_kwargs(default_args))
    new = model_class(**build_model_kwargs(default_args, model_class))
    sd_old, sd_new = old.state_dict(), new.state_dict()
    assert set(sd_old) == set(sd_new)
    assert all(sd_old[k].shape == sd_new[k].shape for k in sd_old)


def test_map_entries_are_all_real_params():
    """Every MODEL_KWARG_MAP key must exist on at least one reachable model."""
    all_params = set()
    for mt in ("denoise3d", "denoise2d"):
        all_params |= set(model_signature_params(fetch_model_class(mt)))
    assert set(MODEL_KWARG_MAP) <= all_params
