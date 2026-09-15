#!/usr/bin/env python3
"""Materialize ``epsilon @ delta_group`` evaluation tables.

The Seen-Base Residual Sweep (SBRS) evaluates a camera group under the exact
fixed calibration table it saw in training, optionally with one fixed residual
on top.  It deliberately writes final SE(3) tables to a registry: evaluation
must never re-sample a residual at rollout time.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE = "instructions/orbital_miscalibration_noise_bimanual_external_only.json"
DEFAULT_RESIDUAL = "instructions/random_miscal_noise_bimanual.json"
DEFAULT_OUTPUT = "instructions/eval_calibrations_seen_base_residual_v01.json"


def axis_angle_to_rotation(axis_angle: list[float]) -> np.ndarray:
    vector = np.asarray(axis_angle, dtype=np.float64)
    angle = np.linalg.norm(vector)
    if angle < 1e-12:
        return np.eye(3)
    axis = vector / angle
    skew = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * (skew @ skew)


def transform(entry: dict | None) -> np.ndarray:
    matrix = np.eye(4)
    if entry is not None:
        matrix[:3, :3] = axis_angle_to_rotation(entry["axis_angle_rad"])
        matrix[:3, 3] = entry["translation_m"]
    return matrix


def repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-file", default=DEFAULT_BASE)
    parser.add_argument("--base-level", default="medium")
    parser.add_argument("--residual-file", default=DEFAULT_RESIDUAL)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--version", default="v01")
    parser.add_argument(
        "--base-condition-id", default="e0",
        help="Identifier for the no-residual trained-base condition (default: e0).",
    )
    parser.add_argument(
        "--include-calibrated", action="store_true",
        help="Add an identity calibration realization named calibrated.",
    )
    parser.add_argument(
        "--residual",
        action="append",
        default=[],
        metavar="ID:ROT:TRANS",
        help="Residual condition, e.g. e2deg-t2cm:2deg:2cm. Repeatable; e0:none:none is implicit.",
    )
    parser.add_argument(
        "--camera-ids", default="0,1",
        help="Comma-separated camera IDs receiving epsilon; defaults to the external-only training mask.",
    )
    args = parser.parse_args()

    base_path, residual_path, output_path = map(repo_path, (args.base_file, args.residual_file, args.output))
    with base_path.open() as handle:
        base_data = json.load(handle)
    with residual_path.open() as handle:
        residual_data = json.load(handle)

    cameras = base_data["cameras"]
    if cameras != residual_data["cameras"]:
        raise ValueError(f"Camera order differs: base={cameras}, residual={residual_data['cameras']}")
    camera_ids = {int(value) for value in args.camera_ids.split(",") if value}
    invalid = camera_ids - set(range(len(cameras)))
    if invalid:
        raise ValueError(f"Invalid camera IDs: {sorted(invalid)}")
    if args.base_level not in base_data["levels"]:
        raise ValueError(f"Unknown base level {args.base_level!r}")

    residuals = [(args.base_condition_id, None, None)]
    for spec in args.residual:
        try:
            condition_id, rot_level, trans_level = spec.split(":")
        except ValueError as exc:
            raise ValueError(f"Invalid --residual {spec!r}; expected ID:ROT:TRANS") from exc
        if rot_level not in residual_data["rotation"] or trans_level not in residual_data["translation"]:
            raise ValueError(f"Unknown residual levels in {spec!r}")
        residuals.append((condition_id, rot_level, trans_level))

    realizations = {}
    if args.include_calibrated:
        realizations["calibrated"] = {
            "description": "Identity calibration realization.",
            "transforms": {camera: np.eye(4).tolist() for camera in cameras},
        }
    for group in base_data["groups"]:
        group_base = base_data["levels"][args.base_level][group]
        for condition_id, rot_level, trans_level in residuals:
            transforms = {}
            for index, camera in enumerate(cameras):
                base = transform(group_base.get(camera))
                epsilon = np.eye(4)
                if rot_level is not None and index in camera_ids:
                    epsilon[:3, :3] = axis_angle_to_rotation(residual_data["rotation"][rot_level][camera]["axis_angle_rad"])
                    epsilon[:3, 3] = residual_data["translation"][trans_level][camera]["translation_m"]
                transforms[camera] = (epsilon @ base).tolist()
            realization_id = f"seen-{group.lower()}-{args.base_level}-{condition_id}-{args.version}"
            residual_description = "no residual (epsilon = identity)" if rot_level is None else f"fixed {rot_level}/{trans_level} residual"
            realizations[realization_id] = {
                "description": (
                    f"Seen-base residual sweep: trained {args.base_level} base for {group}, "
                    f"then {residual_description}; epsilon applies only to camera IDs {sorted(camera_ids)}."
                ),
                "transforms": transforms,
            }

    output = {
        "schema_version": 1,
        "camera_order": cameras,
        "composition": "T_observed = (epsilon @ delta_group) @ T_true",
        "source": {
            "base_file": str(args.base_file), "base_level": args.base_level,
            "residual_file": str(args.residual_file), "residual_camera_ids": sorted(camera_ids),
        },
        "realizations": realizations,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        json.dump(output, handle, indent=2)
        handle.write("\n")
    print(f"Wrote {len(realizations)} calibration realizations to {output_path}")


if __name__ == "__main__":
    main()
