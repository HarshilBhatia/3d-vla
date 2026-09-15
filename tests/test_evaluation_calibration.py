from pathlib import Path

from evaluation.calibration import load_calibration_registry


def test_standard_peract2_registry_resolves_from_repo_relative_path():
    registry = load_calibration_registry("instructions/eval_calibrations_peract2_standard.json")

    assert registry["camera_order"] == ["front", "wrist_left", "wrist_right"]
    assert Path(registry["_path"]).name == "eval_calibrations_peract2_standard.json"
