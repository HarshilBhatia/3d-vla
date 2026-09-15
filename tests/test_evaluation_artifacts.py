import json

import pytest

from evaluation.artifacts import ArtifactPathError, EvaluationCell, cell_result_path, write_json_atomic


def test_cell_result_path_is_structured_and_stable(tmp_path):
    cell = EvaluationCell(
        protocol_id="orbital_camera_subset_v1",
        method_id="view_align",
        condition_id="external_5deg_5cm",
        task_id="bimanual_push_box",
    )

    assert cell_result_path(tmp_path, cell) == (
        tmp_path / "online_eval" / "orbital_camera_subset_v1" / "view_align"
        / "external_5deg_5cm" / "bimanual_push_box" / "result.json"
    )


def test_cell_rejects_path_like_identifiers():
    with pytest.raises(ArtifactPathError, match="safe identifier"):
        EvaluationCell("protocol", "../checkpoint", "clean", "task")


def test_atomic_write_never_replaces_completed_result_without_permission(tmp_path):
    path = tmp_path / "result.json"
    write_json_atomic(path, {"success": 1})

    with pytest.raises(FileExistsError):
        write_json_atomic(path, {"success": 2})

    write_json_atomic(path, {"success": 2}, overwrite=True)
    assert json.loads(path.read_text()) == {"success": 2}
