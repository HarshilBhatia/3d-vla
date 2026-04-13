#!/usr/bin/env python3
"""
Generate task_extrinsics_offsets.json with random (reproducible) rotation and
translation per task. Offsets are in world frame: new_cam_to_world = offset @ cam_to_world.
"""
import json
import numpy as np
from pathlib import Path


def euler_to_rotation_matrix(euler_deg):
    """Rx(roll) @ Ry(pitch) @ Rz(yaw), angles in degrees."""
    r, p, y = np.deg2rad(euler_deg)
    cx, sx = np.cos(r), np.sin(r)
    cy, sy = np.cos(p), np.sin(p)
    cz, sz = np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    return (Rx @ Ry @ Rz).tolist()


def main():
    instructions_path = Path(__file__).resolve().parent / "instructions.json"
    with open(instructions_path) as f:
        tasks = list(json.load(f).keys())

    out_path = Path(__file__).resolve().parent / "task_extrinsics_offsets.json"
    out = {}
    for task in sorted(tasks):
        seed = hash(task) % (2**32)
        rng = np.random.default_rng(seed)
        # Small random rotation: ±20 deg per axis
        euler = (rng.uniform(-20, 20, 3)).tolist()
        R = euler_to_rotation_matrix(euler)
        # Small random translation in meters: ±0.05
        t = (rng.uniform(-0.05, 0.05, 3)).tolist()
        out[task] = {"R": R, "t": t}

    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {out_path} with {len(out)} tasks.")


if __name__ == "__main__":
    main()
