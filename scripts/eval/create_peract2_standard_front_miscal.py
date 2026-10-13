#!/usr/bin/env python3
"""Standard PerAct2 rig calibrations with a fixed front-camera miscalibration per magnitude; wrists stay calibrated."""
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
LEVELS = [(3, 1), (5, 5), (10, 10), (20, 20)]
EYE = np.eye(4).tolist()


def delta(rot_deg, trans_cm, rng):
    axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
    direction = rng.normal(size=3); direction /= np.linalg.norm(direction)
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(np.radians(rot_deg) * axis).as_matrix()
    T[:3, 3] = direction * trans_cm / 100.0
    return T.tolist()


rng = np.random.default_rng(0)
real = {"calibrated": {"description": "Native calibrated PerAct2 camera rig: front, wrist_left, wrist_right.",
                       "transforms": {"front": EYE, "wrist_left": EYE, "wrist_right": EYE}}}
for r, t in LEVELS:
    real[f"front-e{r}deg-t{t}cm-v01"] = {
        "description": f"Front camera fixed {r}deg/{t}cm error (seeded random axis and direction); wrists calibrated.",
        "transforms": {"front": delta(r, t, rng), "wrist_left": EYE, "wrist_right": EYE}}
out = ROOT / "instructions/eval_calibrations_peract2_standard_front_miscal_v01.json"
out.write_text(json.dumps({"schema_version": 1, "camera_order": ["front", "wrist_left", "wrist_right"],
                           "composition": "T_observed = delta @ T_true", "realizations": real}, indent=2) + "\n")
print(out.name, list(real))
