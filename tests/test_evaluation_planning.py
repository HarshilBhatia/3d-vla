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


def test_eval_weights_is_only_set_where_the_evaluator_reads_it():
    import glob
    import json
    for path in glob.glob("instructions/eval_plans/*.json"):
        runtime = json.load(open(path)).get("runtime", {})
        assert "eval_weights" not in runtime, f"{path}: eval_weights must go in runtime.overrides"


def test_robustness_ladder_builds_one_plan_per_source_with_only_kept_levels(tmp_path):
    from evaluation.planning.checkpoint_ladder import build_plans, load_config
    config = load_config("instructions/eval_plans/robustness_ladder_ema_v01.json")
    checkpoint = tmp_path / "interm_step_100000.pth"
    built = build_plans(config, checkpoint, "arm", 100000)

    assert [key for key, _, _ in built] == ["arm:s100000:idlt_calibrated", "arm:s100000:idlt_e60t60"]
    calibrated, extreme = (plan for _, _, plan in built)
    assert len(cells(calibrated)) == len(cells(extreme)) == 13
    assert all(ids == ["calibrated"] for ids in calibrated["task_calibrations"].values())
    assert all(len(ids) == 1 and "-unknown-e60deg-t60cm-" in ids[0] for ids in extreme["task_calibrations"].values())
    for plan in (calibrated, extreme):
        assert plan["output_root"].endswith("3dfa_online_eval_20rollouts")
        assert plan["campaign_id"] == "robustness_ladder_v01_arm_s100000"
        command, _ = command_for_cell(plan, cells(plan)[0])
        assert {"num_demos_total=20", "eval_weights=ema", "visual_num_history=5"} <= set(command)


def test_sbrs_ladder_keeps_single_plan_and_ledger_key(tmp_path):
    from evaluation.planning.checkpoint_ladder import build_plan, build_plans, load_config
    config = load_config("instructions/eval_plans/video_deltam_checkpoint_ladder_ema_v01.json")
    checkpoint = tmp_path / "interm_step_50000.pth"

    [(key, name, plan)] = build_plans(config, checkpoint, "arm", 50000)

    assert key == "arm:s050000" and name == plan["campaign_id"] == "video_deltam_checkpoint_ladder_ema_v01_arm_s050000"
    assert plan == build_plan(config, checkpoint, "arm", 50000)


def test_source_plan_ladder_rejects_runtime_level_eval_weights(tmp_path):
    import json

    import pytest
    from evaluation.planning.checkpoint_ladder import load_config
    config = json.load(open("instructions/eval_plans/robustness_ladder_ema_v01.json"))
    config["runtime"]["eval_weights"] = "ema"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(config))

    with pytest.raises(ValueError, match="runtime.overrides"):
        load_config(path)
