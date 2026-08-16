"""Generate instructions/random_miscal_noise.json.

Pre-samples one random direction per camera for each rotation level and
translation level. Rotation and translation levels are independent so they
can be combined freely at eval time.

Run from the repo root:
    python scripts/generate_random_miscal_noise.py [--seed 42] [--overwrite]

The bimanual orbital setup has four cameras instead of three, so it gets its own
file (regenerating the three-camera file with extra cameras would shift every
existing level's sampled direction and invalidate past sweeps):

    python scripts/generate_random_miscal_noise.py --cameras bimanual \
        --out instructions/random_miscal_noise_bimanual.json
"""
import argparse
import json
import math
import os

import numpy as np

CAMERA_SETS = {
    "single": ["orbital_left", "orbital_right", "wrist"],
    "bimanual": ["orbital_left", "orbital_right", "wrist_left", "wrist_right"],
}

# Rotation levels: label -> magnitude in degrees (1..10, 15, 20)
ROTATION_LEVELS = {f"{i}deg": float(i) for i in range(1, 11)}
ROTATION_LEVELS.update({"15deg": 15.0, "20deg": 20.0})

# Translation levels: label -> magnitude in metres (1cm..10cm, 15cm, 20cm)
TRANSLATION_LEVELS = {f"{i}cm": i / 100.0 for i in range(1, 11)}
TRANSLATION_LEVELS.update({"15cm": 0.15, "20cm": 0.20})


def _sample_axis_angle(angle_deg, rng):
    axis = rng.standard_normal(3)
    axis /= np.linalg.norm(axis)
    return (axis * angle_deg * math.pi / 180.0).tolist()


def _sample_translation(magnitude_m, rng):
    direction = rng.standard_normal(3)
    direction /= np.linalg.norm(direction)
    return (direction * magnitude_m).tolist()


def generate(seed, out_path, overwrite, cameras):
    if os.path.exists(out_path) and not overwrite:
        print(f"File already exists: {out_path}. Use --overwrite to regenerate.")
        return

    rng = np.random.default_rng(seed)

    rotation_levels = {}
    for label, deg in ROTATION_LEVELS.items():
        rotation_levels[label] = {
            cam: {"axis_angle_rad": _sample_axis_angle(deg, rng)}
            for cam in cameras
        }

    translation_levels = {}
    for label, mag_m in TRANSLATION_LEVELS.items():
        translation_levels[label] = {
            cam: {"translation_m": _sample_translation(mag_m, rng)}
            for cam in cameras
        }

    data = {
        "cameras": cameras,
        "rotation_levels": list(ROTATION_LEVELS.keys()),
        "translation_levels": list(TRANSLATION_LEVELS.keys()),
        "rotation": rotation_levels,
        "translation": translation_levels,
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Written: {out_path}")
    print(f"  rotation levels:    {list(ROTATION_LEVELS.keys())}")
    print(f"  translation levels: {list(TRANSLATION_LEVELS.keys())}")
    print(f"  cameras:            {cameras}")
    print(f"  seed:               {seed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="instructions/random_miscal_noise.json")
    parser.add_argument("--cameras", default="single", choices=sorted(CAMERA_SETS))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    generate(args.seed, args.out, args.overwrite, CAMERA_SETS[args.cameras])
