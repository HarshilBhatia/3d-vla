import torch

from datasets.utils import from_relative_keypose, to_relative_keypose
from training.base import offset_proprio


def _batch():
    torch.manual_seed(0)
    prop = torch.randn(4, 3, 2, 8)
    prop[..., 3:7] = torch.nn.functional.normalize(prop[..., 3:7], dim=-1)
    absolute = torch.randn(4, 8, 2, 8)
    absolute[..., 3:7] = torch.nn.functional.normalize(absolute[..., 3:7], dim=-1)
    return prop, absolute


def test_relative_target_keeps_absolute_pose():
    prop, absolute = _batch()
    rel = to_relative_keypose(absolute, prop)
    new_prop, new_rel = offset_proprio(prop, rel, 0.05, relative=True)
    assert not torch.allclose(new_prop[..., :3], prop[..., :3])
    back = from_relative_keypose(new_rel, new_prop)
    assert torch.allclose(back[..., :3], absolute[..., :3], atol=1e-5)
    # q and -q are the same rotation.
    assert torch.allclose((back[..., 3:7] * absolute[..., 3:7]).sum(-1).abs(), torch.ones(back.shape[:-1]), atol=1e-5)


def test_absolute_target_untouched_and_offset_shared_over_history():
    prop, absolute = _batch()
    new_prop, new_abs = offset_proprio(prop, absolute, 0.05, relative=False)
    assert torch.equal(new_abs, absolute)
    d = new_prop[..., :3] - prop[..., :3]
    assert torch.allclose(d, d[:, :1].expand_as(d))
    assert torch.equal(new_prop[..., 3:], prop[..., 3:])
