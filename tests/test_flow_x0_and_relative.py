import pytest
import torch

from datasets.utils import from_relative_keypose, to_relative_keypose
from modeling.policy.base_denoise_actor import FLOW_X0_MIN_T, DenoiseActor


def _poses(*shape):
    q = torch.nn.functional.normalize(torch.randn(*shape, 4), dim=-1)
    return torch.cat([torch.randn(*shape, 3), q, torch.rand(*shape, 1)], -1)


@pytest.mark.parametrize("qform", ["xyzw", "wxyz"])
def test_relative_keypose_round_trip_is_per_arm(qform):
    torch.manual_seed(0)
    action, proprio = _poses(2, 1, 2), _poses(2, 3, 2)
    rel = to_relative_keypose(action, proprio, qform)
    back = from_relative_keypose(rel, proprio, qform)
    assert torch.allclose(back[..., :3], action[..., :3], atol=1e-5)
    # q and -q are the same rotation.
    dot = (back[..., 3:7] * action[..., 3:7]).sum(-1).abs()
    assert torch.allclose(dot, torch.ones_like(dot), atol=1e-5)
    # Each arm's delta is taken from its own current state, never from the other arm.
    assert torch.allclose(rel[:, 0, 1, :3], action[:, 0, 1, :3] - proprio[:, -1, 1, :3], atol=1e-6)


def test_x0_prediction_maps_to_velocity_and_euler_step_lands_on_x0():
    actor = DenoiseActor(embedding_dim=24, num_attn_heads=4, denoise_model="rectified_flow", denoise_timesteps=5, flow_prediction="x0")
    z, x0 = torch.randn(2, 1, 2, 9), torch.randn(2, 1, 2, 9)
    out = torch.cat([x0, torch.zeros(2, 1, 2, 1)], -1)
    t = torch.full((2,), 0.2)
    v = actor._as_velocity(out, z, t)
    assert torch.allclose(z - 0.2 * v[..., :-1], x0, atol=1e-6)
    v_small = actor._as_velocity(out, z, torch.full((2,), 1e-4))
    assert torch.allclose(v_small[..., :-1], (z - x0) / FLOW_X0_MIN_T)


def test_velocity_prediction_is_untouched_and_bad_values_rejected():
    actor = DenoiseActor(embedding_dim=24, num_attn_heads=4, denoise_model="rectified_flow")
    out = torch.randn(2, 1, 2, 10)
    assert actor._as_velocity(out, torch.randn(2, 1, 2, 9), torch.rand(2)) is out
    with pytest.raises(ValueError):
        DenoiseActor(embedding_dim=24, num_attn_heads=4, denoise_model="rectified_flow", flow_prediction="eps")
