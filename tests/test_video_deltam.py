import torch

from modeling.policy.video_deltam import VideoDeltaM


def test_video_deltam_is_causal_per_camera():
    torch.manual_seed(0)
    model = VideoDeltaM(12, 3, depth=2, dropout=0.0).eval()
    frames = torch.randn(2, 5, 3, 12)

    baseline_frames = model(frames)
    future_changed = frames.clone()
    future_changed[:, 4, 0] += 100.0
    changed_frames = model(future_changed)

    # Frame attention is causal in time: a t=4 perturbation cannot reach t<4.
    assert torch.allclose(baseline_frames[:, :4], changed_frames[:, :4], atol=1e-5)


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

    baseline_frames = model(images)
    hook.remove()
    assert same_time_lengths == [3 * 7]
    future_changed = images.clone()
    future_changed[:, 4, 0] += 100.0
    changed_frames = model(future_changed)

    assert baseline_frames.shape == (2, 5, 3, 12)
    # A future image can change its own full-image summary but cannot leak
    # backward through the causal per-camera temporal operation.
    assert torch.allclose(baseline_frames[:, :4], changed_frames[:, :4], atol=1e-5)
    assert not torch.allclose(baseline_frames[:, 4, 0], changed_frames[:, 4, 0])

    # Same-time attention mixes all patch/image-summary tokens across cameras.
    other_camera_changed = images.clone()
    other_camera_changed[:, 2, 1] += 100.0
    other_camera_frames = model(other_camera_changed)
    assert not torch.allclose(baseline_frames[:, 2, 0], other_camera_frames[:, 2, 0])
