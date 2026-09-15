from types import SimpleNamespace

from evaluation.checkpoints import overlay_checkpoint_config


def test_checkpoint_overlay_changes_model_keys_but_not_evaluation_runtime_keys():
    args = SimpleNamespace(
        backbone="siglip2",
        checkpoint="runtime.pth",
        task="bimanual_push_box",
        output_file="result.json",
    )

    loaded = overlay_checkpoint_config(args, {
        "backbone": "clip",
        "checkpoint": "checkpoint_saved_path.pth",
        "task": "checkpoint_saved_task",
        "output_file": "checkpoint_saved_result.json",
    })

    assert loaded == {"backbone": "clip"}
    assert args.backbone == "clip"
    assert args.checkpoint == "runtime.pth"
    assert args.task == "bimanual_push_box"
    assert args.output_file == "result.json"
