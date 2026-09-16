import torch

from modeling.policy.video_deltam import VideoDeltaM


def test_video_deltam_is_causal_per_camera():
    torch.manual_seed(0)
    model = VideoDeltaM(12, 3, depth=2, dropout=0.0).eval()
    frames = torch.randn(2, 5, 3, 12)
    camera_token = torch.randn(2, 1, 12)

    baseline_frames, baseline_context, _ = model(frames, camera_token)
    future_changed = frames.clone()
    future_changed[:, 4, 0] += 100.0
    changed_frames, changed_context, _ = model(future_changed, camera_token)

    # A current-frame change affects the current global context.
    assert baseline_context.shape == (2, 1, 12)
    assert not torch.allclose(baseline_context, changed_context)
    assert torch.allclose(baseline_frames[:, :4], changed_frames[:, :4], atol=1e-5)


def test_full_image_path_self_attends_over_patches_and_keeps_video_causality():
    torch.manual_seed(0)
    model = VideoDeltaM(12, 3, depth=2, dropout=0.0, full_image=True).eval()
    images = torch.randn(2, 5, 3, 7, 12)
    camera_token = torch.randn(2, 1, 12)

    # Full-image inputs are first pooled by the legacy per-image cross-attn
    # readout, then same-time attention receives one token per camera.
    same_time_lengths = []
    hook = model.same_time[0].attn.register_forward_hook(
        lambda _module, args, _output: same_time_lengths.append(args[0].shape[1])
    )

    baseline_frames, baseline_context, _ = model(images, camera_token)
    hook.remove()
    assert same_time_lengths == [3]
    future_changed = images.clone()
    future_changed[:, 4, 0] += 100.0
    changed_frames, changed_context, _ = model(future_changed, camera_token)

    assert baseline_frames.shape == (2, 5, 3, 12)
    assert baseline_context.shape == (2, 1, 12)
    assert not torch.allclose(baseline_context, changed_context)

    # Same-time attention mixes all patch/image-summary tokens across cameras.
    other_camera_changed = images.clone()
    other_camera_changed[:, 2, 1] += 100.0
    _, other_camera_context, _ = model(other_camera_changed, camera_token)
    assert not torch.allclose(baseline_context, other_camera_context)


def test_predict_delta_m_role_emits_only_delta_m():
    """In the predict_delta_m role the stack's sole output is one orthogonal
    correction per camera; it builds no register readout at all."""
    torch.manual_seed(0)
    model = VideoDeltaM(12, 3, depth=2, dropout=0.0, predict_delta_m=True).eval()
    assert not hasattr(model, "camera_readout")
    # The head is zero-initialised (see the identity test below); give it weight
    # so this exercises the mechanism rather than the initialisation.
    torch.nn.init.normal_(model.delta_m_head[-1].weight, std=0.1)

    frames = torch.randn(2, 5, 3, 12)
    refined, context, delta_M = model(frames, torch.randn(2, 1, 12))
    assert context is None
    assert delta_M.shape == (2, 3, 6, 6)

    # Orthogonal: the RoPE basis must be rotated, never rescaled.
    eye = torch.eye(6).expand_as(delta_M)
    assert torch.allclose(delta_M @ delta_M.transpose(-1, -2), eye, atol=1e-5)

    # Still causal: a change to the newest frame cannot alter earlier ones.
    future_changed = frames.clone()
    future_changed[:, 4, 0] += 100.0
    refined_changed, _, delta_M_changed = model(future_changed, torch.randn(2, 1, 12))
    assert torch.allclose(refined[:, :4], refined_changed[:, :4], atol=1e-5)
    assert not torch.allclose(delta_M, delta_M_changed)


def test_delta_m_starts_at_identity():
    """The last layer is zero-initialised, so training starts from exactly no
    correction -- important for warm-starting from a checkpoint trained with a
    head-predicted delta_M."""
    torch.manual_seed(0)
    model = VideoDeltaM(12, 3, depth=2, dropout=0.0, predict_delta_m=True).eval()
    _, _, delta_M = model(torch.randn(2, 5, 3, 12), torch.randn(2, 1, 12))
    eye = torch.eye(6).expand_as(delta_M)
    assert torch.allclose(delta_M, eye, atol=1e-6)
