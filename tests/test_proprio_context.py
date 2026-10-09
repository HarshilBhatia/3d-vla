from types import SimpleNamespace

import pytest
import torch
from torch import nn

from modeling.encoder.multimodal.encoder3d import Encoder
from modeling.utils.layers import AttentionModule
from modeling.utils.position_encodings import RotaryPositionEncoding3D

C, NHIST, NCAM, P = 12, 6, 4, 5


def _encoder(context):
    torch.manual_seed(0)
    enc = SimpleNamespace(proprio_context=context, curr_gripper_embed=nn.Embedding(NHIST, C),
                          relative_pe_layer=RotaryPositionEncoding3D(C))
    if context != "none":
        enc.gripper_context_head = AttentionModule(num_layers=2, d_model=C, dim_fw=C, n_heads=3, rotary_pe=True,
                                                   use_adaln=False, pre_norm=False).eval()
    else:
        enc.proprio_value_mlp = nn.Sequential(nn.Linear(8, C), nn.ReLU(), nn.Linear(C, C))
    return enc


def _run(enc, proprio, feats, pos):
    with torch.no_grad():
        return Encoder.encode_proprio(enc, proprio, feats, pos, ncam=NCAM)


def _inputs():
    torch.manual_seed(1)
    return torch.randn(2, NHIST, 8), torch.randn(2, NCAM * P, C), torch.rand(2, NCAM * P, 3)


def _perturb(t, cams, scale=5.0):
    t = t.clone()
    for c in cams:
        t[:, c * P:(c + 1) * P] += scale
    return t


def test_wrist_context_ignores_external_cameras_but_not_wrists():
    enc, (proprio, feats, pos) = _encoder("wrist"), _inputs()
    base = _run(enc, proprio, feats, pos)
    assert torch.allclose(base, _run(enc, proprio, _perturb(feats, [0, 1]), _perturb(pos, [0, 1], 0.6)))
    assert not torch.allclose(base, _run(enc, proprio, _perturb(feats, [2, 3]), pos))


def test_all_context_depends_on_external_cameras():
    enc, (proprio, feats, pos) = _encoder("all"), _inputs()
    assert not torch.allclose(_run(enc, proprio, feats, pos), _run(enc, proprio, _perturb(feats, [0, 1]), pos))


def test_none_context_ignores_scene_and_reads_gripper_state():
    enc, (proprio, feats, pos) = _encoder("none"), _inputs()
    base = _run(enc, proprio, feats, pos)
    assert torch.allclose(base, _run(enc, proprio, _perturb(feats, [0, 1, 2, 3]), _perturb(pos, [0, 1, 2, 3])))
    assert not torch.allclose(base, _run(enc, proprio + 1.0, feats, pos))


def test_wrist_context_rejects_other_camera_layouts():
    enc, (proprio, feats, pos) = _encoder("wrist"), _inputs()
    with pytest.raises(ValueError):
        Encoder.encode_proprio(enc, proprio, feats, pos, ncam=2)
