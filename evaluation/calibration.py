"""Materialized calibration realizations used by online evaluation.

A realization is the complete per-camera ``delta`` used in one evaluation
cell.  It intentionally contains matrices, rather than ingredients from which
to compose or sample matrices at runtime.  This gives a result name one
unambiguous meaning across machines and over time.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_registry_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else _repo_root() / path


def load_calibration_registry(path: str | Path) -> dict[str, Any]:
    """Load and validate a materialized calibration registry."""
    path = resolve_registry_path(path)
    with path.open() as handle:
        registry = json.load(handle)
    if registry.get("schema_version") != 1:
        raise ValueError(f"Unsupported calibration registry schema in {path}")
    cameras = registry.get("camera_order")
    if not isinstance(cameras, list) or not cameras or len(set(cameras)) != len(cameras):
        raise ValueError(f"{path}: camera_order must be a non-empty unique list")
    realizations = registry.get("realizations")
    if not isinstance(realizations, dict) or not realizations:
        raise ValueError(f"{path}: realizations must be a non-empty object")
    for realization_id, entry in realizations.items():
        transforms = entry.get("transforms") if isinstance(entry, dict) else None
        if set(transforms or ()) != set(cameras):
            raise ValueError(f"{path}: realization {realization_id!r} must specify every camera exactly once")
        for camera in cameras:
            matrix = np.asarray(transforms[camera], dtype=np.float64)
            if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
                raise ValueError(f"{path}: {realization_id!r}/{camera} is not a finite 4x4 matrix")
            if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8):
                raise ValueError(f"{path}: {realization_id!r}/{camera} has an invalid homogeneous row")
            rotation = matrix[:3, :3]
            if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
                raise ValueError(f"{path}: {realization_id!r}/{camera} rotation is not in SO(3)")
    registry["_path"] = str(path.resolve())
    return registry


def calibration_transform_table(
    registry_path: str | Path,
    realization_id: str,
    camera_order: tuple[str, ...] | list[str],
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return the exact ``(N,4,4)`` table for ``realization_id``.

    Camera order is checked deliberately: an ID cannot silently apply a valid
    matrix to the wrong stream.
    """
    registry = load_calibration_registry(registry_path)
    requested = list(camera_order)
    if requested != registry["camera_order"]:
        raise ValueError(
            "Calibration registry camera_order does not match evaluator order: "
            f"registry={registry['camera_order']}, evaluator={requested}"
        )
    try:
        transforms = registry["realizations"][realization_id]["transforms"]
    except KeyError as exc:
        known = ", ".join(registry["realizations"])
        raise ValueError(f"Unknown calibration realization {realization_id!r}; known: {known}") from exc
    return torch.tensor([transforms[camera] for camera in requested], dtype=dtype)
