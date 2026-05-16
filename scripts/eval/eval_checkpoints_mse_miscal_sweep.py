"""Sweep fixed-magnitude miscal noise (rot deg × trans cm) on G1/G3 samples only.

Loads each checkpoint once, filters the dataset to ``camera_group ∈ allowed_groups``
(default ``[1, 3]``), then iterates a sweep of (rot_deg, trans_cm) cells:

  diagonal:  (k, k)  for k = 0..max_level
  rot_only:  (k, 0)  for k = 1..max_level
  trans_only:(0, k)  for k = 1..max_level

Each cell rebuilds the preprocessor with ``miscal_fixed_angle_deg`` /
``miscal_fixed_translation_m`` set, so every perturbation has *exactly* that
rotation angle and translation length (direction random). Each cell writes one
row per data_path to ``output_csv``.

Usage:
    python scripts/eval/eval_checkpoints_mse_miscal_sweep.py \
        checkpoints=train_logs/Orbital/cotrain_mixed_miscal/last.pth,train_logs/Orbital/multi_cam_G3G4/last.pth \
        data_paths=/grogu/user/harshilb/multi_cam/train.zarr \
        val_instructions=instructions/peract/instructions.json \
        bimanual=false \
        output_csv=results/miscal_sweep/G1G3.csv \
        num_batches=100 \
        max_level=10 \
        allowed_groups=[1,3] \
        seed=0
"""
import sys
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from scripts.eval.eval_utils import (
    BASE_SCRIPT_KEYS, METRIC_KEYS, camera_group_bucket_fn,
    eval_bucketed, extract_script_args, load_args, load_model,
    make_camera_group_filter, make_loader, make_preprocessor, make_tokenizer,
    metric_row, parse_csv_list, parse_group_ids, pick_amp_dtype, write_csv_rows,
)


SCRIPT_KEYS = BASE_SCRIPT_KEYS | {"max_level", "allowed_groups", "seed", "sweep_axis"}
VALID_AXES = ("all", "diagonal", "rot", "trans")

# Sweep cells: ckpt-name is added at write time
CSV_HEADER = [
    "ckpt_name", "step", "dataset", "sweep_axis",
    "rot_deg", "trans_cm", "camera_group", "n_samples",
] + METRIC_KEYS


def build_sweep_cells(max_level, axis_filter="all"):
    """(rot_deg, trans_cm, axis_label) tuples for the requested sweep axis.

    ``axis_filter`` ∈ {"all", "diagonal", "rot", "trans"}. "all" runs the
    diagonal, then rot-only (skipping the (0,0) duplicate), then trans-only.
    """
    if axis_filter not in VALID_AXES:
        raise ValueError(f"sweep_axis must be one of {VALID_AXES}, got {axis_filter!r}")
    cells = []
    if axis_filter in ("all", "diagonal"):
        for k in range(max_level + 1):
            cells.append((k, k, "diagonal"))
    if axis_filter in ("all", "rot"):
        start = 1 if axis_filter == "all" else 0
        for k in range(start, max_level + 1):
            cells.append((k, 0, "rot"))
    if axis_filter in ("all", "trans"):
        start = 1 if axis_filter == "all" else 0
        for k in range(start, max_level + 1):
            cells.append((0, k, "trans"))
    return cells


def main():
    custom, hydra_argv = extract_script_args(sys.argv[1:], SCRIPT_KEYS)

    checkpoints = parse_csv_list(custom.get("checkpoints"))
    data_paths = parse_csv_list(custom.get("data_paths"))
    output_csv = Path(custom.get("output_csv", "results/miscal_sweep/results.csv"))
    num_batches = int(custom.get("num_batches", 100))
    max_level = int(custom.get("max_level", 10))
    allowed_groups = parse_group_ids(custom.get("allowed_groups")) or [1, 3]
    seed = int(custom.get("seed", 0))
    sweep_axis = custom.get("sweep_axis", "all")

    if not checkpoints:
        raise ValueError("Pass at least one checkpoint via checkpoints=path.pth,...")
    if not data_paths:
        raise ValueError("Pass at least one data path via data_paths=path.zarr,...")

    args = load_args(hydra_argv)
    args.chunk_size = 1  # required by CameraGroupFilteredDataset (1 dataset idx = 1 zarr row)

    cells = build_sweep_cells(max_level, sweep_axis)

    print(f"Checkpoints ({len(checkpoints)}):")
    for p in checkpoints:
        print(f"  {p}")
    print(f"Data paths ({len(data_paths)}):")
    for p in data_paths:
        print(f"  {p}")
    print(
        f"allowed_groups={allowed_groups}  num_batches={num_batches}  "
        f"max_level={max_level}  sweep_axis={sweep_axis}  n_cells={len(cells)}  "
        f"seed={seed}  output_csv={output_csv}"
    )
    print("-" * 80)

    amp_dtype = pick_amp_dtype()
    print(f"AMP dtype: {amp_dtype}")

    cam_group_wrapper = make_camera_group_filter(allowed_groups)

    for ckpt_path in checkpoints:
        ckpt_name = Path(ckpt_path).parent.name  # e.g. cotrain_mixed_miscal
        args_copy = deepcopy(args)
        model, ckpt_step = load_model(args_copy, ckpt_path)
        # args.chunk_size may have been overwritten by the checkpoint's saved config — force back to 1.
        args_copy.chunk_size = 1
        tokenizer = make_tokenizer(args_copy)

        # Build loaders once per data_path (same across cells).
        loaders = {
            dp: make_loader(args_copy, dp, chunk_size=1, dataset_wrapper=cam_group_wrapper)
            for dp in data_paths
        }

        for rot_deg, trans_cm, axis in cells:
            trans_m = trans_cm / 100.0
            preprocessor = make_preprocessor(
                args_copy,
                miscal_fixed_angle_deg=rot_deg,
                miscal_fixed_translation_m=trans_m,
            )

            for data_path in data_paths:
                # Fix the seed per cell so each ckpt sees the same per-batch noise pattern.
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)

                print(
                    f"\n[{ckpt_name}] axis={axis}  rot={rot_deg}deg  trans={trans_cm}cm  "
                    f"data={Path(data_path).name}",
                    flush=True,
                )
                per_group = eval_bucketed(
                    model, tokenizer, preprocessor, loaders[data_path],
                    num_batches, amp_dtype, args_copy.relative_action,
                    bucket_fn=camera_group_bucket_fn,
                )

                rows = []
                for g in sorted(per_group.keys()):
                    metrics, n = per_group[g]
                    rows.append(metric_row(
                        metrics,
                        ckpt_name=ckpt_name,
                        step=ckpt_step,
                        dataset=Path(data_path).stem,
                        sweep_axis=axis,
                        rot_deg=rot_deg,
                        trans_cm=trans_cm,
                        camera_group=f"G{g}",
                        n_samples=n,
                    ))
                    print(
                        f"  G{g} (n={n})  "
                        f"pos_l2={metrics.get('traj_pos_l2', float('nan')):.4f}  "
                        f"rot_l1={metrics.get('traj_rot_l1', float('nan')):.4f}  "
                        f"pos_acc={metrics.get('traj_pos_acc_001', float('nan')):.3f}  "
                        f"rot_acc={metrics.get('traj_rot_acc_0025', float('nan')):.3f}",
                        flush=True,
                    )
                write_csv_rows(output_csv, CSV_HEADER, rows)

        del model, loaders
        torch.cuda.empty_cache()

    print(f"\nAll results written to {output_csv}")


if __name__ == "__main__":
    main()
