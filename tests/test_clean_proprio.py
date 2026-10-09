from types import SimpleNamespace

import pytest
import torch

from evaluation.checkpoints import EVALUATION_RUNTIME_KEYS
from evaluation.online.runner import build_environment_kwargs
from modeling.encoder.multimodal.base_encoder import pcd_to_tokens, proprio_context_pos
from modeling.policy.base_denoise_actor import DenoiseActor
from utils.hydra_utils import get_config, get_config_path


def _pcd(*shape):
    torch.manual_seed(0)
    return torch.randn(*shape, 3, 32, 32)


def test_same_cloud_gives_bit_identical_proprio_positions():
    pcd = _pcd(2, 4)
    tokens = pcd_to_tokens(pcd, 8, 8)
    assert torch.equal(proprio_context_pos(pcd, tokens), tokens)


def test_history_cloud_uses_the_current_frame_only():
    pcd = _pcd(2, 5, 4)
    tokens = pcd_to_tokens(pcd[:, -1], 8, 8)
    other = pcd.clone()
    other[:, :-1] += 3.0
    assert torch.equal(proprio_context_pos(other, tokens), tokens)
    other[:, -1, 0] += 1.0
    assert not torch.equal(proprio_context_pos(other, tokens), tokens)


def test_non_square_grid_is_rejected():
    with pytest.raises(ValueError):
        proprio_context_pos(_pcd(2, 4), torch.zeros(2, 4 * 8 * 6, 3))


def test_forward_requires_proprio_pcd_exactly_when_switch_is_on():
    off, on = SimpleNamespace(eval_clean_proprio=False), SimpleNamespace(eval_clean_proprio=True)
    with pytest.raises(ValueError):
        DenoiseActor.forward(off, None, None, None, None, None, None, None, run_inference=True, proprio_pcd=torch.zeros(1))
    with pytest.raises(ValueError):
        DenoiseActor.forward(on, None, None, None, None, None, None, None, run_inference=True)
    with pytest.raises(ValueError):
        DenoiseActor.forward(on, None, None, None, None, None, None, None, run_inference=True, cfg_scale=1.0,
                             proprio_pcd=torch.zeros(1))


def test_switch_is_eval_owned_and_off_by_default():
    assert "eval_clean_proprio" in EVALUATION_RUNTIME_KEYS
    args = get_config(overrides=[], config_name="config", config_path=get_config_path())
    assert getattr(args, "eval_clean_proprio") is False


def test_orbital_bimanual_environment_gets_the_flag_only_when_on():
    base = dict(bimanual=True, dataset="OrbitalPeract2", cameras_file="c.json", spawn_camera_group="G1", fov_deg=60,
                eval_calibration_registry=None, eval_calibration_id=None)
    assert "clean_proprio" not in build_environment_kwargs(SimpleNamespace(**base))
    assert build_environment_kwargs(SimpleNamespace(**base, eval_clean_proprio=True))["clean_proprio"] is True
