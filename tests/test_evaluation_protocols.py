import json
from pathlib import Path

import pytest

from evaluation.protocols import ProtocolValidationError, load_protocol


def test_loads_every_shipped_plan():
    root = Path(__file__).resolve().parents[1]
    plans = sorted((root / "instructions" / "eval_plans").glob("*.json"))

    assert plans
    for path in plans:
        protocol = load_protocol(path)
        assert protocol.identifier
        assert protocol.tasks
        assert protocol.runtime["dataset"]
        assert isinstance(protocol.runtime["bimanual"], bool)


def test_protocol_requires_an_identifier(tmp_path):
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps({
        "schema_version": 1,
        "tasks": ["bimanual_push_box"],
        "runtime": {"dataset": "OrbitalPeract2", "bimanual": True},
    }))

    with pytest.raises(ProtocolValidationError, match="protocol_id or campaign_id"):
        load_protocol(path)


def test_protocol_rejects_empty_task_list(tmp_path):
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps({
        "schema_version": 1,
        "campaign_id": "empty_tasks_v1",
        "tasks": [],
        "runtime": {"dataset": "OrbitalPeract2", "bimanual": True},
    }))

    with pytest.raises(ProtocolValidationError, match="non-empty tasks"):
        load_protocol(path)
