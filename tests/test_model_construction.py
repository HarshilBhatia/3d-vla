"""Tests for the shared model-kwargs construction helper.

Covers signature filtering, the explicit rename/derivation map, the startup
guardrail, and the single-knob enums (head_positional_encoding, scene_sampling).

The former "reproduce the trainer's hand-written kwargs dict" fixture is gone:
it froze a contract from eb06b4c and silently rotted as constructor params were
added, so it stopped catching the bug it was written for. The guardrail tests
below cover that intent directly.
"""

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from modeling.encoder.multimodal.base_encoder import SCENE_SAMPLING
from modeling.policy import fetch_model_class
from modeling.policy.base_denoise_actor import HEAD_POSITIONAL_ENCODINGS
from modeling.policy.construction import (
    MODEL_KWARG_MAP,
    assert_model_kwargs_complete,
    build_model_kwargs,
    model_signature_params,
)
from utils.hydra_utils import get_config, get_config_path

EXPERIMENTS = sorted(p.stem for p in (get_config_path() / "experiment").glob("*.yaml"))


@pytest.fixture(scope="module")
def default_args():
    return get_config(overrides=[], config_name="config", config_path=get_config_path())


# --------------------------------------------------------------------------
# Signature filtering
# --------------------------------------------------------------------------
def test_signature_filter_drops_non_constructor_keys(default_args):
    kwargs = build_model_kwargs(default_args, fetch_model_class("denoise3d"))
    assert set(kwargs) <= set(model_signature_params(fetch_model_class("denoise3d")))
    for k in ("num_workers", "batch_size", "wandb_project", "train_data_dir", "lr"):
        assert k not in kwargs


def test_denoise2d_gets_only_its_own_params(default_args):
    """A narrower model must not receive 3d-only flags."""
    kwargs = build_model_kwargs(default_args, fetch_model_class("denoise2d"))
    for k in ("video_deltam", "video_deltam_full_image", "view_align_cameras",
              "layerwise_view_align"):
        assert k not in kwargs
    inspect.signature(fetch_model_class("denoise2d").__init__).bind(None, **kwargs)


# --------------------------------------------------------------------------
# Rename / derivation map
# --------------------------------------------------------------------------
def test_nhist_comes_from_proprio_history_not_visual_history(default_args):
    """nhist sizes curr_gripper_embed and the proprio tensor, so it is the
    proprio window. visual_num_history is the *visual* history K and is unrelated."""
    kwargs = build_model_kwargs(default_args, fetch_model_class("denoise3d"))
    assert kwargs["nhist"] == default_args.proprio_num_history
    assert "visual_num_history" not in kwargs
    assert "proprio_num_history" not in kwargs


def test_rename_map(default_args):
    kwargs = build_model_kwargs(default_args, fetch_model_class("denoise3d"))
    assert kwargs["relative"] == default_args.relative_action
    assert kwargs["nhand"] == (2 if default_args.bimanual else 1)
    assert "relative_action" not in kwargs
    assert "bimanual" not in kwargs


def test_nhand_derivation_both_ways():
    model_class = fetch_model_class("denoise3d")
    base = dict(proprio_num_history=3, relative_action=False, bimanual=False)
    assert build_model_kwargs(SimpleNamespace(**base), model_class)["nhand"] == 1
    base["bimanual"] = True
    assert build_model_kwargs(SimpleNamespace(**base), model_class)["nhand"] == 2


def test_rename_source_missing_raises():
    args = SimpleNamespace(relative_action=False, bimanual=False)  # no proprio_num_history
    with pytest.raises(KeyError, match="proprio_num_history"):
        build_model_kwargs(args, fetch_model_class("denoise3d"))


def test_map_entries_are_all_real_params():
    all_params = set()
    for mt in ("denoise3d", "denoise2d"):
        all_params |= set(model_signature_params(fetch_model_class(mt)))
    assert set(MODEL_KWARG_MAP) <= all_params


# --------------------------------------------------------------------------
# Single-knob enums
# --------------------------------------------------------------------------
@pytest.mark.parametrize("mode", sorted(HEAD_POSITIONAL_ENCODINGS))
def test_head_positional_encoding_reaches_the_model(default_args, mode):
    default_args.head_positional_encoding = mode
    kwargs = build_model_kwargs(default_args, fetch_model_class("denoise3d"))
    assert kwargs["head_positional_encoding"] == mode
    default_args.head_positional_encoding = "rope3d"


@pytest.mark.parametrize("mode", sorted(SCENE_SAMPLING))
def test_scene_sampling_reaches_the_model(default_args, mode):
    default_args.scene_sampling = mode
    kwargs = build_model_kwargs(default_args, fetch_model_class("denoise3d"))
    assert kwargs["scene_sampling"] == mode
    default_args.scene_sampling = "fps"


def test_scene_sampling_modes_are_distinct():
    """Each mode must select a different (skip, image_space, xyz) combination."""
    assert len(set(SCENE_SAMPLING.values())) == len(SCENE_SAMPLING)


def test_scene_sampling_rejects_unknown_mode(default_args):
    from modeling.encoder.multimodal.base_encoder import Encoder
    with pytest.raises(ValueError, match="scene_sampling"):
        Encoder(backbone="clip", scene_sampling="bogus")


def test_view_align_requires_a_rope_basis(default_args):
    """learned_abs / none leave delta_M nothing to correct, so they must raise."""
    model_class = fetch_model_class("denoise3d")
    for mode in ("learned_abs", "none"):
        kwargs = build_model_kwargs(default_args, model_class)
        kwargs.update(head_positional_encoding=mode, view_align_mode='rope_6d')
        with pytest.raises(ValueError, match="head_positional_encoding"):
            model_class(**kwargs)


# --------------------------------------------------------------------------
# Public vocabulary
# --------------------------------------------------------------------------
def test_public_paper_config_resolves_to_legacy_runtime_api():
    """The paper-facing config must resolve to the expected settings."""
    args = get_config(
        overrides=["experiment=paper_external_view_align_eeaux"],
        config_name="config",
        config_path=get_config_path(),
    )
    assert args.view_align_mode == "rope_6d"
    assert args.view_align_cameras == [0, 1]
    assert args.layerwise_view_align is True
    assert args.miscal_cameras == [0, 1]
    assert args.miscal_mode == "group"
    assert args.miscal_group_level == "medium"
    assert args.perturbation_noise_rot_deg == 3.0
    assert args.perturbation_noise_trans_m == 0.01
    assert args.ee_aux is True
    assert args.ee_aux_weight == 1.0
    assert args.ee_aux_cameras == [0, 1]


def test_every_setting_has_exactly_one_name(default_args):
    """The legacy/public alias layer is gone; retired spellings must not resurface."""
    for retired in ("predict_extrinsics", "extrinsics_prediction_mode", "delta_m_camera_ids",
                    "dynamic_rope_from_camtoken", "predict_ee_aux", "lambda_aux",
                    "ee_aux_cam_ids", "causal_cam_history", "causal_cam_history_depth"):
        assert not hasattr(default_args, retired), retired
    assert default_args.view_align_mode == "none"


# --------------------------------------------------------------------------
# Guardrail
# --------------------------------------------------------------------------
def test_guardrail_passes_on_shipped_config(default_args):
    assert_model_kwargs_complete(default_args, fetch_model_class("denoise3d"))


def test_guardrail_allows_documented_defaults(default_args):
    assert_model_kwargs_complete(default_args, fetch_model_class("denoise2d"))


def test_guardrail_raises_on_dropped_key(default_args):
    """The scene_sampling failure mode: a config key silently not forwarded."""
    model_class = fetch_model_class("denoise3d")
    kwargs = build_model_kwargs(default_args, model_class)
    kwargs.pop("scene_sampling")
    with pytest.raises(ValueError, match="scene_sampling"):
        assert_model_kwargs_complete(default_args, model_class, kwargs)


def test_guardrail_raises_on_uncovered_constructor_param(default_args):
    class ModelWithNewFlag:
        def __init__(self, embedding_dim=60, brand_new_flag=False):
            pass

    with pytest.raises(ValueError, match="brand_new_flag"):
        assert_model_kwargs_complete(default_args, ModelWithNewFlag)


# --------------------------------------------------------------------------
# Every shipped experiment must stay constructible
# --------------------------------------------------------------------------
@pytest.mark.parametrize("experiment", EXPERIMENTS)
def test_every_experiment_config_builds_valid_kwargs(experiment):
    """Catches an experiment config left behind by a config rename."""
    args = get_config(overrides=[f"experiment={experiment}"],
                      config_name="config", config_path=get_config_path())
    model_class = fetch_model_class(args.model_type)
    kwargs = build_model_kwargs(args, model_class)
    assert_model_kwargs_complete(args, model_class, kwargs)
    inspect.signature(model_class.__init__).bind(None, **kwargs)
    assert kwargs["head_positional_encoding"] in HEAD_POSITIONAL_ENCODINGS
    assert kwargs["scene_sampling"] in SCENE_SAMPLING
