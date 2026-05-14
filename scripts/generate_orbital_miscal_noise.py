"""
Generate and store per-group and per-task orbital miscalibration noise.

Produces instructions/orbital_miscalibration_noise.json with fixed axis-angle +
translation vectors for each (level, group, camera) and (level, task, camera)
combination.  Run once; check the file into version control so every training
run uses the same noise.

Usage:
    python scripts/generate_orbital_miscal_noise.py
    python scripts/generate_orbital_miscal_noise.py --seed 7 --out /path/to/out.json

Noise is sampled as:
  axis  ~ uniform on S²  (via normalised Gaussian)
  angle ~ uniform in [0, max_angle_deg]   (rad after conversion)
  translation ~ uniform in [-max_translation_m, +max_translation_m] per axis
  axis_angle_rad = axis * angle  (norm = rotation magnitude)

Per-task noise uses the same magnitude configs as per-group noise but is sampled
with seed+1000 so the values are independent.
"""
import argparse
import json
import os
import sys

import numpy as np


CAMERAS = ["orbital_left", "orbital_right", "wrist"]
GROUPS  = ["G1", "G2", "G3", "G4", "G5", "G6"]
TASKS = [
    "place_cups", "close_jar", "insert_onto_square_peg",
    "light_bulb_in", "meat_off_grill", "open_drawer",
    "place_shape_in_shape_sorter", "place_wine_at_rack_location",
    "push_buttons", "put_groceries_in_cupboard",
    "put_item_in_drawer", "put_money_in_safe", "reach_and_drag",
    "slide_block_to_color_target", "stack_blocks", "stack_cups",
    "sweep_to_dustpan_of_size", "turn_tap",
]

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


def _sample_entries(rng, keys, cameras, cfg):
    """Return {key: {cam: {axis_angle_rad, translation_m}}} for each key in keys."""
    result = {}
    for key in keys:
        key_data = {}
        for cam in cameras:
            key_data[cam] = {
                "axis_angle_rad": [
                    round(v, 6) for v in
                    _sample_axis_angle(rng, cfg["max_angle_deg"])
                ],
                "translation_m": [
                    round(v, 6) for v in
                    _sample_translation(rng, cfg["max_translation_m"])
                ],
            }
        result[key] = key_data
    return result


def generate(seed, cameras, groups, tasks, levels):
    rng_group      = np.random.default_rng(seed)
    rng_task_group = np.random.default_rng(seed + 2000)
    task_group_keys = [f"{task}_{group}" for task in tasks for group in groups]
    out = {
        "cameras": cameras,
        "groups": groups,
        "tasks": tasks,
        "task_group_keys": task_group_keys,
        "levels": {},
        "per_task_group_levels": {},
    }

    for level_name, cfg in levels.items():
        out["levels"][level_name] = {
            "_comment": cfg["_comment"],
            **_sample_entries(rng_group, groups, cameras, cfg),
        }
        out["per_task_group_levels"][level_name] = {
            "_comment": cfg["_comment"] + " (per-task-group variant: one noise per (task, group) pair)",
            **_sample_entries(rng_task_group, task_group_keys, cameras, cfg),
        }

    return out


def _print_summary(data):
    sections = [
        ("levels", data["groups"]),
        ("per_task_group_levels", data["task_group_keys"][:6]),  # first 6 to keep output short
    ]
    for section, key_list in sections:
        print(f"\n  [{section}] (showing first {len(key_list)} keys)")
        for level, ldata in data[section].items():
            print(f"  {level}:")
            for key in key_list:
                if key not in ldata:
                    continue
                kdata = ldata[key]
                for cam, cdata in kdata.items():
                    aa = cdata["axis_angle_rad"]
                    angle_deg = np.degrees(np.linalg.norm(aa))
                    t_norm = np.linalg.norm(cdata["translation_m"]) * 100  # cm
                    print(f"    {key}/{cam}: angle={angle_deg:.2f} deg, |t|={t_norm:.1f} cm")


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
    p.add_argument("--tasks", nargs="+", default=TASKS,
                   help="Task names for per-task noise (default: all 18 PerAct2 tasks)")
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

    print(f"Generating orbital miscal noise  seed={args.seed}  groups={args.groups}  tasks={args.tasks}  cameras={args.cameras}")
    data = generate(args.seed, args.cameras, args.groups, args.tasks, LEVELS)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nWrote {args.out}")
    _print_summary(data)


if __name__ == "__main__":
    main()
