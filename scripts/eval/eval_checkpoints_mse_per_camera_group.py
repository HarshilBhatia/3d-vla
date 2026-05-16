"""Evaluate a checkpoint on a zarr dataset and report metrics bucketed by camera_group.

One pass over the data: per-sample metrics are aggregated separately for each
camera_group value observed in the dataset. Optionally applies cotrain-style
miscalibration noise to a subset of groups (matches training-time noise).

Usage:
    python scripts/eval/eval_checkpoints_mse_per_camera_group.py \
        checkpoints=train_logs/Orbital/multi_cam_G3G4/best.pth \
        data_paths=/grogu/user/harshilb/multi_cam_G3G4.zarr \
        val_instructions=instructions/peract/instructions.json \
        bimanual=false \
        output_csv=results/per_camera_group/multi_cam_G3G4_clean.csv \
        num_batches=200

    # cotrain run with noise on G1+G2:
    python scripts/eval/eval_checkpoints_mse_per_camera_group.py \
        checkpoints=train_logs/Orbital/cotrain_mixed_miscal/best.pth \
        data_paths=/grogu/user/harshilb/multi_cam/train.zarr \
        cotrain_miscal_group_ids=[1,2] \
        cotrain_miscal_level=medium \
        output_csv=results/per_camera_group/cotrain_mixed_miscal_medium.csv \
        num_batches=200

Output CSV columns: step, miscal_level, dataset, camera_group, n_samples, <METRIC_KEYS>.
"""
import sys
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from scripts.eval.eval_utils import (
    BASE_SCRIPT_KEYS, METRIC_KEYS,
    camera_group_bucket_fn, eval_bucketed, extract_script_args,
    load_args, load_model, make_loader, make_preprocessor, make_tokenizer,
    metric_row, parse_csv_list, parse_group_ids, pick_amp_dtype,
    print_bucketed_results, write_csv_rows,
)


SCRIPT_KEYS = BASE_SCRIPT_KEYS | {
    "cotrain_miscal_group_ids", "cotrain_miscal_level",
    "orbital_miscal_noise_level",
    "predict_extrinsics", "dynamic_rope_from_camtoken",
}
CSV_HEADER = ["step", "miscal_level", "dataset", "camera_group", "n_samples"] + METRIC_KEYS


def main():
    custom, hydra_argv = extract_script_args(sys.argv[1:], SCRIPT_KEYS)

    checkpoints = parse_csv_list(custom.get("checkpoints"))
    data_paths = parse_csv_list(custom.get("data_paths"))
    output_csv = Path(custom.get("output_csv", "results/per_camera_group/results.csv"))
    num_batches = int(custom.get("num_batches", 200))
    cotrain_group_ids = parse_group_ids(custom.get("cotrain_miscal_group_ids"))
    cotrain_level = custom.get("cotrain_miscal_level") or None
    orbital_miscal_noise_level = custom.get("orbital_miscal_noise_level") or None

    def _parse_bool(raw):
        return None if raw is None else str(raw).strip().lower() not in ("false", "0", "no")

    arch_overrides = {}
    if "predict_extrinsics" in custom:
        arch_overrides["predict_extrinsics"] = _parse_bool(custom["predict_extrinsics"])
    if "dynamic_rope_from_camtoken" in custom:
        arch_overrides["dynamic_rope_from_camtoken"] = _parse_bool(custom["dynamic_rope_from_camtoken"])

    if not checkpoints:
        raise ValueError("Pass at least one checkpoint path via checkpoints=path1.pth")
    if not data_paths:
        raise ValueError("Pass at least one data path via data_paths=path1.zarr,path2.zarr")

    args = load_args(hydra_argv)

    if cotrain_group_ids and cotrain_level:
        miscal_label = f"cotrain_{cotrain_level}_G{'+G'.join(map(str, cotrain_group_ids))}"
    elif orbital_miscal_noise_level:
        miscal_label = f"orbital_{orbital_miscal_noise_level}"
    else:
        miscal_label = "clean"

    rope_label = ""
    if arch_overrides.get("predict_extrinsics") is False:
        rope_label = "_no_deltaM"
    elif arch_overrides.get("dynamic_rope_from_camtoken") is False:
        rope_label = "_static_deltaM"

    print(f"Checkpoints ({len(checkpoints)}):")
    for p in checkpoints:
        print(f"  {p}")
    print(f"Data paths ({len(data_paths)}):")
    for p in data_paths:
        print(f"  {p}")
    if arch_overrides:
        print(f"arch_overrides: {arch_overrides}")
    print(f"miscal_label={miscal_label}{rope_label}  num_batches={num_batches}  output_csv={output_csv}")
    print("-" * 80)

    amp_dtype = pick_amp_dtype()
    print(f"AMP dtype: {amp_dtype}")

    for ckpt_path in checkpoints:
        args_copy = deepcopy(args)
        model, ckpt_step = load_model(args_copy, ckpt_path, arch_overrides=arch_overrides or None)
        preprocessor = make_preprocessor(
            args_copy,
            orbital_miscal_noise_level=orbital_miscal_noise_level,
            cotrain_miscal_group_ids=cotrain_group_ids,
            cotrain_miscal_level=cotrain_level,
        )
        tokenizer = make_tokenizer(args_copy)

        for data_path in data_paths:
            print(f"\nEvaluating on: {data_path}")
            loader = make_loader(args_copy, data_path)
            per_group_results = eval_bucketed(
                model, tokenizer, preprocessor, loader,
                num_batches, amp_dtype, args_copy.relative_action,
                bucket_fn=camera_group_bucket_fn,
            )

            full_label = f"{miscal_label}{rope_label}"
            print_bucketed_results(
                ckpt_path, ckpt_step, data_path, per_group_results,
                bucket_name="G", bucket_labels={g: g for g in per_group_results},
                extra=f"miscal={full_label}",
            )
            rows = [
                metric_row(metrics,
                           step=ckpt_step, miscal_level=full_label,
                           dataset=Path(data_path).stem,
                           camera_group=f"G{g}", n_samples=n)
                for g in sorted(per_group_results)
                for metrics, n in [per_group_results[g]]
            ]
            write_csv_rows(output_csv, CSV_HEADER, rows)
            print(f"Rows written to {output_csv}")

        del model
        torch.cuda.empty_cache()

    print(f"\nAll results written to {output_csv}")


if __name__ == "__main__":
    main()
