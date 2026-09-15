import torch

from modeling.policy.video_deltam import VideoDeltaM


def test_video_deltam_is_causal_per_camera():
    torch.manual_seed(0)
    model = VideoDeltaM(12, 3, depth=2, dropout=0.0).eval()
    frames = torch.randn(2, 5, 3, 12)

    baseline_context = model(frames)
    future_changed = frames.clone()
    future_changed[:, 4, 0] += 100.0
    changed_context = model(future_changed)

    # A current-frame change affects the current global context.
    assert baseline_context.shape == (2, 12)
    assert not torch.allclose(baseline_context, changed_context)


def test_full_image_path_self_attends_over_patches_and_keeps_video_causality():
    torch.manual_seed(0)
    model = VideoDeltaM(12, 3, depth=2, dropout=0.0, full_image=True).eval()
    images = torch.randn(2, 5, 3, 7, 12)

    # Same-time SA receives only M * P patch tokens; image summary tokens are
    # reserved for the causal, per-camera temporal operation.
    same_time_lengths = []
    hook = model.same_time[0].attn.register_forward_hook(
        lambda _module, args, _output: same_time_lengths.append(args[0].shape[1])
    )

    baseline_context = model(images)
    hook.remove()
    assert same_time_lengths == [3 * 7]
    future_changed = images.clone()
    future_changed[:, 4, 0] += 100.0
    changed_context = model(future_changed)

    assert baseline_context.shape == (2, 12)
    assert not torch.allclose(baseline_context, changed_context)

    # Same-time attention mixes all patch/image-summary tokens across cameras.
    other_camera_changed = images.clone()
    other_camera_changed[:, 2, 1] += 100.0
    other_camera_context = model(other_camera_changed)
    assert not torch.allclose(baseline_context, other_camera_context)
