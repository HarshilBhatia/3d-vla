from pathlib import Path

from evaluation.planning.checkpoint_ladder import checkpoint_steps
from evaluation.planning.run_plan import cells, command_for_cell, load_plan


def test_checkpoint_ladder_discovers_only_aligned_steps(tmp_path):
    for name in ("interm_step_60000.pth", "interm_step_70000.pth", "interm_step_80000.pth", "last.pth"):
        (tmp_path / name).touch()

    found = checkpoint_steps(tmp_path, min_step=70000, interval=10000)

    assert [(step, path.name) for step, path in found] == [
        (70000, "interm_step_70000.pth"),
        (80000, "interm_step_80000.pth"),
    ]


def test_plan_cells_use_canonical_evaluation_cli():
    root = Path(__file__).resolve().parents[1]
    plan, _ = load_plan(root / "instructions/eval_plans/g7_external_calibration.json")
    cell = cells(plan)[0]

    command, output = command_for_cell(plan, cell, output_root="/tmp/evaluation-artifacts")

    assert command[1:3] == ["-m", "evaluation.cli"]
    assert output.parts[-2:] == ("calibrated", "results_bimanual_push_box.json")


def test_standard_peract2_plan_has_no_orbital_camera_plumbing():
    root = Path(__file__).resolve().parents[1]
    plan, _ = load_plan(root / "instructions/eval_plans/peract2_original_clean_v01.json")

    command, output = command_for_cell(plan, cells(plan)[0], output_root="/tmp/evaluation-artifacts")

    assert not any(argument.startswith("cameras_file=") for argument in command)
    assert not any(argument.startswith("spawn_camera_group=") for argument in command)
    assert "model_type=denoise2d" in command
    assert output.parts[-3:] == ("standard_front_wrist", "calibrated", "results_bimanual_push_box.json")
