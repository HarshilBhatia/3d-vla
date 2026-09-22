#!/usr/bin/env python3
"""Write the ten G7 viewpoint-sweep camera groups: mirrored orbital pairs moved in and out of the G1-G6 grid."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "instructions/orbital_cameras_grouped.json"
OUT = ROOT / "instructions/orbital_cameras_g7_sweep_v01.json"
# Least-squares intersection of all G1-G7 optical axes; reproduces G7's stored R to within 0.01.
CENTER = np.array([0.248, -0.001, 0.954])
G7 = dict(az=45.0, el=47.5, r=1.31)
# (name, azimuth deg, elevation deg, radius m); left camera at +az, right mirrored at -az.
SWEEP = [
    ("G7_el72", 45, 72, 1.31),
    ("G7_el85", 45, 85, 1.31),
    ("G7_az25", 25, 47.5, 1.31),
    ("G7_az65", 65, 47.5, 1.31),
    ("G7_az90", 90, 47.5, 1.31),
    ("G7_r180", 45, 47.5, 1.80),
    # Slightly OOD: just outside the training grid in azimuth, or in elevation and radius.
    ("G7_slight_az56", 56, 47.5, 1.31),
    ("G7_slight_el66_r145", 45, 66, 1.45),
    # Extreme OOD: far from the table, and from the opposite side of the workspace.
    ("G7_extreme_r350", 45, 35, 3.50),
    ("G7_extreme_az150_r250", 150, 30, 2.50),
]


def pose(az, el, r):
    a, e = np.radians(az), np.radians(el)
    p = CENTER + r * np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    z = (CENTER - p) / np.linalg.norm(CENTER - p)
    # Straight-down cameras have no horizontal right vector; fall back to world -y.
    x = np.cross([0, 0, 1], z)
    x = x / np.linalg.norm(x) if np.linalg.norm(x) > 1e-6 else np.array([0.0, -1.0, 0.0])
    return p, np.stack([x, np.cross(z, x), z], 1)


def optical_axes(groups):
    for g in groups:
        for side in ("left", "right"):
            yield g["group"], side, np.array(g[side]["pos"]) - CENTER


def angle(u, v):
    return float(np.degrees(np.arccos(np.clip(u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1, 1))))


def main():
    base = json.loads(SRC.read_text())
    train = [(g, s, v) for g, s, v in optical_axes(base) if g != "G7"]
    out = []
    for name, az, el, r in SWEEP:
        entry = {"group": name, "sweep": {"azimuth_deg": az, "elevation_deg": el, "radius_m": r}}
        for side, sign in (("left", 1), ("right", -1)):
            p, R = pose(sign * az, el, r)
            nearest = min(angle(p - CENTER, v) for g, s, v in train if s == side)
            entry[side] = {"name": f"{side[0].upper()}_{name}", "pos": p.tolist(), "R": R.tolist(),
                           "deg_to_nearest_train": round(nearest, 1)}
        out.append(entry)
        print(f"{name:24} az={az:5.1f} el={el:5.1f} r={r:.2f}  nearest-train L={entry['left']['deg_to_nearest_train']:5.1f} R={entry['right']['deg_to_nearest_train']:5.1f} deg")
    # Keep G7 itself so the file is self-contained for plots and as the sweep's anchor.
    out.insert(0, next(g for g in base if g["group"] == "G7"))
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
