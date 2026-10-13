import torch

from modeling.utils.position_encodings import SinusoidalPosEmb


def _layout(traj_len, nhand, c=8):
    emb = SinusoidalPosEmb(c)(torch.arange(traj_len))
    pos = emb[None, :, None].expand(2, -1, nhand, -1).reshape(2, -1, c)
    return emb, pos


def test_each_token_gets_its_own_step_embedding():
    emb, pos = _layout(8, 2)
    # traj_feats flatten (B, T, nhand, C) -> (B, T*nhand, C), so token i is step i // nhand.
    for i in range(16):
        assert torch.equal(pos[0, i], emb[i // 2])


def test_single_step_matches_old_layout():
    emb = SinusoidalPosEmb(8)(torch.arange(1))
    old = emb[None, None].repeat(2, 1, 2, 1).reshape(2, -1, 8)
    _, new = _layout(1, 2)
    assert torch.equal(old, new)
