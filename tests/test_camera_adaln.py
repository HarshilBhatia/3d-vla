import pytest
import torch

from modeling.policy.denoise_actor_3d import TransformerHead

B, T, NHAND, C, NCAM, P, M, L = 2, 1, 2, 12, 4, 6, 8, 3


def _head(mode):
    return TransformerHead(embedding_dim=C, num_attn_heads=3, nhist=3 * NHAND, num_shared_attn_layers=2,
                           view_align_mode='none', camera_adaln=mode)


def _inputs():
    g = torch.Generator().manual_seed(0)
    r = lambda *s: torch.randn(*s, generator=g)
    fps_cam_ids = torch.arange(NCAM).repeat_interleave(M // NCAM)[None].expand(B, -1)
    return dict(traj_feats=r(B, T, NHAND, C), trajectory=r(B, T, NHAND, 9), timesteps=torch.ones(B),
                rgb3d_feats=r(B, NCAM * P, C), rgb3d_pos=r(B, NCAM * P, 3), rgb2d_feats=None, rgb2d_pos=None,
                instr_feats=r(B, L, C), instr_pos=r(B, L, 3), proprio_feats=r(B, 3 * NHAND, C),
                fps_scene_feats=r(B, M + NCAM, C), fps_scene_pos=r(B, M + NCAM, 3), fps_cam_ids=fps_cam_ids)


def _wake_adaln(head):
    # Zero-init adaLN is an identity, so give it weights to make the C-token signal visible.
    torch.manual_seed(1)
    for name, p in head.named_parameters():
        if "adaln.modulation" in name:
            p.data.normal_(std=0.1)


def _out(head, inputs):
    return head(**inputs)[0][0]


@pytest.mark.parametrize("mode", ["per_camera", "pooled", "static_pooled"])
def test_camera_adaln_output_depends_on_c_tokens_and_trains(mode):
    head = _head(mode).eval()
    _wake_adaln(head)
    inputs = _inputs()
    base = _out(head, inputs)
    changed = dict(inputs, fps_scene_feats=inputs["fps_scene_feats"].clone())
    changed["fps_scene_feats"][:, M:] += 5.0
    assert not torch.allclose(base, _out(head, changed))
    head.train()
    _out(head, inputs).sum().backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in head.camera_adaln_proj.parameters())


def test_per_camera_signal_is_per_token_and_traj_gets_time_only():
    head = _head("per_camera")
    inputs = _inputs()
    time_embs = torch.randn(B, C)
    summaries = inputs["fps_scene_feats"][:, M:]
    sig = head._camera_adaln_signal(time_embs, summaries, inputs["fps_cam_ids"], T * NHAND)
    assert sig.shape == (B, T * NHAND + M + NCAM + 5, C)
    assert torch.allclose(sig[:, :T * NHAND], time_embs[:, None].expand(-1, T * NHAND, -1))
    assert torch.allclose(sig[:, -5:], time_embs[:, None].expand(-1, 5, -1))
    cam0 = time_embs + head.camera_adaln_proj(summaries[:, 0])
    assert torch.allclose(sig[:, T * NHAND], cam0, atol=1e-6)


def test_camera_adaln_off_builds_nothing_and_rejects_delta_m():
    assert not hasattr(_head("none"), "camera_adaln_proj")
    with pytest.raises(ValueError):
        TransformerHead(embedding_dim=C, num_attn_heads=3, view_align_mode='rope_6d', camera_adaln='pooled')


def test_eval_ablation_zero_camera_term_returns_time_signal():
    head = _head("pooled")
    head.eval_ablation = "camera_adaln_zero"
    inputs = _inputs()
    t = torch.randn(B, C)
    assert torch.equal(head._camera_adaln_signal(t, inputs["fps_scene_feats"][:, M:], inputs["fps_cam_ids"], T * NHAND), t)


def test_eval_ablation_identity_delta_m():
    head = TransformerHead(embedding_dim=C, num_attn_heads=3, view_align_mode='rope_6d', view_align_cameras=[0, 1])
    torch.nn.init.normal_(head.camera_predictor.weight, std=1.0)
    feats = torch.randn(B, NCAM, C)
    _, dm = head._predict_from_cam_feat(feats)
    assert not torch.allclose(dm[:, 0], torch.eye(6).expand(B, 6, 6), atol=1e-3)
    head.eval_ablation = "delta_m_identity"
    _, dm = head._predict_from_cam_feat(feats)
    assert torch.allclose(dm, torch.eye(6).expand_as(dm))
