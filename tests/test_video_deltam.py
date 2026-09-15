import torch

from modeling.policy.video_deltam import VideoDeltaM


def test_video_deltam_is_causal_per_camera():
    torch.manual_seed(0)
    model = VideoDeltaM(12, 3, depth=2, dropout=0.0).eval()
    frames = torch.randn(2, 5, 3, 12)
    camera_token = torch.randn(2, 1, 12)

    baseline_frames, baseline_context = model(frames, camera_token)
    future_changed = frames.clone()
    future_changed[:, 4, 0] += 100.0
    changed_frames, changed_context = model(future_changed, camera_token)

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

    baseline_frames, baseline_context = model(images, camera_token)
    hook.remove()
    assert same_time_lengths == [3]
    future_changed = images.clone()
    future_changed[:, 4, 0] += 100.0
    changed_frames, changed_context = model(future_changed, camera_token)

    assert baseline_frames.shape == (2, 5, 3, 12)
    assert baseline_context.shape == (2, 1, 12)
    assert not torch.allclose(baseline_context, changed_context)

    # Same-time attention mixes all patch/image-summary tokens across cameras.
    other_camera_changed = images.clone()
    other_camera_changed[:, 2, 1] += 100.0
    _, other_camera_context = model(other_camera_changed, camera_token)
    assert not torch.allclose(baseline_context, other_camera_context)
