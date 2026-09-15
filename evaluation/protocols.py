"""Validation and loading for versioned evaluation protocol JSON files."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


class ProtocolValidationError(ValueError):
    """Raised when a protocol cannot describe a reproducible evaluation."""


@dataclass(frozen=True)
class EvaluationProtocol:
    """A validated protocol while retaining its source-compatible payload."""

    identifier: str
    source_path: Path
    tasks: tuple[str, ...]
    runtime: Mapping[str, Any]
    raw: Mapping[str, Any]


def _task_name(entry: Any, path: Path) -> str:
    if isinstance(entry, str) and entry:
        return entry
    if isinstance(entry, Mapping) and isinstance(entry.get("task"), str) and entry["task"]:
        return entry["task"]
    raise ProtocolValidationError(f"{path}: every task must be a non-empty string or object with a task field")


def load_protocol(path: str | Path) -> EvaluationProtocol:
    """Load a schema-v1 evaluation plan and validate its universal contract.

    Existing plans have two compatible identifiers (``protocol_id`` and
    ``campaign_id``), so both remain accepted during the migration.
    """
    path = Path(path)
    try:
        with path.open() as handle:
            raw = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ProtocolValidationError(f"{path}: invalid JSON") from exc

    if not isinstance(raw, dict):
        raise ProtocolValidationError(f"{path}: protocol must be a JSON object")
    if raw.get("schema_version") != 1:
        raise ProtocolValidationError(f"{path}: unsupported schema_version")

    identifier = raw.get("protocol_id") or raw.get("campaign_id")
    if not isinstance(identifier, str) or not identifier:
        raise ProtocolValidationError(f"{path}: protocol must define protocol_id or campaign_id")

    task_entries = raw.get("tasks")
    if not isinstance(task_entries, list) or not task_entries:
        raise ProtocolValidationError(f"{path}: protocol must define non-empty tasks")
    tasks = tuple(_task_name(task, path) for task in task_entries)

    runtime = raw.get("runtime")
    if not isinstance(runtime, dict):
        raise ProtocolValidationError(f"{path}: protocol must define a runtime object")
    if not isinstance(runtime.get("dataset"), str) or not runtime["dataset"]:
        raise ProtocolValidationError(f"{path}: runtime.dataset must be a non-empty string")
    if not isinstance(runtime.get("bimanual"), bool):
        raise ProtocolValidationError(f"{path}: runtime.bimanual must be a boolean")

    return EvaluationProtocol(
        identifier=identifier,
        source_path=path.resolve(),
        tasks=tasks,
        runtime=runtime,
        raw=raw,
    )
