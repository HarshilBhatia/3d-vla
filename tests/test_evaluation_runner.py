from pathlib import Path
from types import SimpleNamespace

import pytest

from evaluation.online.runner import build_environment_kwargs, resolve_runtime_paths


def test_resolve_runtime_paths_preserves_eval_data_compatibility(tmp_path):
    args = SimpleNamespace(data_dir=Path("demos"), eval_data_dir=Path("test_data"), output_file=Path("result.json"))

    resolve_runtime_paths(args, tmp_path)

    assert args.data_dir == tmp_path / "test_data"
    assert args.output_file == tmp_path / "result.json"


def test_materialized_calibration_disables_legacy_composition():
    args = SimpleNamespace(
        bimanual=True,
        dataset="OrbitalPeract2",
        cameras_file="cameras.json",
        spawn_camera_group="G7",
        fov_deg=70,
        eval_calibration_registry="registry.json",
        eval_calibration_id="external_5deg_5cm",
        orbital_miscal_noise_level="medium",
        orbital_miscal_noise_file="noise.json",
        miscal_rot_level=5,
        miscal_trans_level=5,
        miscal_camera_indices=[0, 1],
    )

    kwargs = build_environment_kwargs(args)

    assert kwargs["calibration_registry"] == "registry.json"
    assert kwargs["calibration_id"] == "external_5deg_5cm"
    assert kwargs["orbital_miscal_noise_level"] is None
    assert kwargs["miscal_rot_level"] is None


def test_partial_materialized_calibration_is_rejected():
    args = SimpleNamespace(
        bimanual=True,
        dataset="OrbitalPeract2",
        eval_calibration_registry="registry.json",
        eval_calibration_id=None,
    )

    with pytest.raises(ValueError, match="supplied together"):
        build_environment_kwargs(args)
