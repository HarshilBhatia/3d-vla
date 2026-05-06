"""
Generate and store per-group orbital miscalibration noise.

Produces instructions/orbital_miscalibration_noise.json with fixed axis-angle +
translation vectors for each (level, group, camera) combination.  Run once;
check the file into version control so every training run uses the same noise.

Usage:
    python scripts/generate_orbital_miscal_noise.py
    python scripts/generate_orbital_miscal_noise.py --seed 7 --out /path/to/out.json

Noise is sampled as:
  axis  ~ uniform on S²  (via normalised Gaussian)
  angle ~ uniform in [0, max_angle_deg]   (rad after conversion)
  translation ~ uniform in [-max_translation_m, +max_translation_m] per axis
  axis_angle_rad = axis * angle  (norm = rotation magnitude)
"""
import argparse
import json
import os
import sys

import numpy as np


CAMERAS = ["orbital_left", "orbital_right", "wrist"]
GROUPS  = ["G1", "G2", "G3", "G4", "G5", "G6"]

LEVELS = {
    "small": {
        "max_angle_deg":       5.0,
        "max_translation_m":   0.02,
        "_comment": "max_angle=5 deg, max_translation=0.02 m. "
                    "Point error ~60 mm at 0.7 m range.",
    },
    "medium": {
        "max_angle_deg":       10.0,
        "max_translation_m":   0.05,
        "_comment": "max_angle=10 deg, max_translation=0.05 m. "
                    "Point error ~125 mm at 0.7 m range.",
    },
    "large": {
        "max_angle_deg":       15.0,
        "max_translation_m":   0.08,
        "_comment": "max_angle=15 deg, max_translation=0.08 m. "
                    "Point error ~190 mm at 0.7 m range.",
    },
}


def _sample_axis_angle(rng, max_angle_deg):
    axis = rng.standard_normal(3)
    axis /= np.linalg.norm(axis) + 1e-12
    angle = rng.uniform(0.0, np.deg2rad(max_angle_deg))
    return (axis * angle).tolist()


def _sample_translation(rng, max_m):
    return rng.uniform(-max_m, max_m, size=3).tolist()


def generate(seed, cameras, groups, levels):
    rng = np.random.default_rng(seed)
    out = {"cameras": cameras, "groups": groups, "levels": {}}

    for level_name, cfg in levels.items():
        level_data = {"_comment": cfg["_comment"]}
        for group in groups:
            group_data = {}
            for cam in cameras:
                group_data[cam] = {
                    "axis_angle_rad": [
                        round(v, 6) for v in
                        _sample_axis_angle(rng, cfg["max_angle_deg"])
                    ],
                    "translation_m": [
                        round(v, 6) for v in
                        _sample_translation(rng, cfg["max_translation_m"])
                    ],
                }
            level_data[group] = group_data
        out["levels"][level_name] = level_data

    return out


def _print_summary(data):
    for level, ldata in data["levels"].items():
        print(f"\n  {level}:")
        for group in data["groups"]:
            gdata = ldata[group]
            for cam, cdata in gdata.items():
                aa = cdata["axis_angle_rad"]
                angle_deg = np.degrees(np.linalg.norm(aa))
                t_norm = np.linalg.norm(cdata["translation_m"]) * 100  # cm
                print(f"    {group}/{cam}: angle={angle_deg:.2f} deg, |t|={t_norm:.1f} cm")


def parse_args():
    repo_root = os.path.join(os.path.dirname(__file__), "..")
    default_out = os.path.normpath(
        os.path.join(repo_root, "instructions", "orbital_miscalibration_noise.json")
    )
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out",    default=default_out,
                   help="Output JSON path (default: instructions/orbital_miscalibration_noise.json)")
    p.add_argument("--seed",   type=int, default=42,
                   help="Random seed (default: 42)")
    p.add_argument("--groups", nargs="+", default=GROUPS,
                   help="Group names (default: G1..G6)")
    p.add_argument("--cameras", nargs="+", default=CAMERAS,
                   help="Camera names (default: orbital_left orbital_right wrist)")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing file")
    return p.parse_args()


def main():
    args = parse_args()

    if os.path.exists(args.out) and not args.overwrite:
        print(f"[SKIP] {args.out} already exists. Use --overwrite to regenerate.")
        return

    print(f"Generating orbital miscal noise  seed={args.seed}  groups={args.groups}  cameras={args.cameras}")
    data = generate(args.seed, args.cameras, args.groups, LEVELS)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nWrote {args.out}")
    _print_summary(data)


if __name__ == "__main__":
    main()
