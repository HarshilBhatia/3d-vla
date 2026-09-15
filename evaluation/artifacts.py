"""Safe, deterministic paths and writes for evaluation artifacts."""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ArtifactPathError(ValueError):
    """Raised when an artifact identifier cannot safely become a path part."""


_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _validate_identifier(value: str) -> None:
    if not isinstance(value, str) or not _SAFE_IDENTIFIER.fullmatch(value):
        raise ArtifactPathError(f"{value!r} is not a safe identifier")


@dataclass(frozen=True)
class EvaluationCell:
    """The four semantic coordinates of one materialized evaluation result."""

    protocol_id: str
    method_id: str
    condition_id: str
    task_id: str

    def __post_init__(self) -> None:
        for value in (self.protocol_id, self.method_id, self.condition_id, self.task_id):
            _validate_identifier(value)


def cell_result_path(artifact_root: str | Path, cell: EvaluationCell) -> Path:
    """Return the canonical result location without creating it."""
    return (
        Path(artifact_root) / "online_eval" / cell.protocol_id / cell.method_id
        / cell.condition_id / cell.task_id / "result.json"
    )


def write_json_atomic(path: str | Path, payload: Any, *, overwrite: bool = False) -> Path:
    """Write JSON atomically and never replace a completed result by default."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite completed artifact: {path}")

    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return path
