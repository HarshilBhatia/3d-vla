"""Evaluate checkpoints on multiple zarr data paths across miscalibration noise levels.

Usage (single GPU, no torchrun needed):
    python scripts/eval/eval_checkpoints_mse_miscal.py \
        checkpoints=/path/to/ckpt1.pth \
        data_paths=/path/to/val1.zarr,/path/to/val2.zarr \
        val_instructions=instructions/peract/instructions.json \
        bimanual=false \
        orbital_miscal_noise_level=medium \
        output_csv=results/checkpoint_mse_miscal/step_10000_noise_medium.csv \
        num_batches=100

Identical to eval_checkpoints_mse.py but adds an `orbital_miscal_noise_level` argument
that is forwarded to the data preprocessor. The CSV gains a `miscal_level` column.

When running as a SLURM array job (one task per checkpoint × noise level), pass a
per-task output_csv so parallel tasks never write the same file.
Use scripts/eval/plot_checkpoint_mse.py (grouping by miscal_level) to aggregate.

Dataset type, model architecture, and all other config are auto-detected from
each checkpoint's saved config dict — no need to pass arch flags on the CLI.
"""
import sys
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from scripts.eval.eval_utils import (
    BASE_SCRIPT_KEYS, METRIC_KEYS,
    eval_scalar, extract_script_args, load_args, load_model,
    make_loader, make_preprocessor, make_tokenizer, metric_row,
    parse_csv_list, pick_amp_dtype, print_scalar_results, write_csv_rows,
)


SCRIPT_KEYS = BASE_SCRIPT_KEYS | {
    "orbital_miscal_noise_level",
    "orbital_miscal_noise_level_per_task_group",
}
CSV_HEADER = ["step", "miscal_level", "dataset", "n_samples"] + METRIC_KEYS


def main():
    custom, hydra_argv = extract_script_args(sys.argv[1:], SCRIPT_KEYS)

    checkpoints = parse_csv_list(custom.get("checkpoints"))
    data_paths = parse_csv_list(custom.get("data_paths"))
    output_csv = Path(custom.get("output_csv", "results/checkpoint_mse_miscal/results.csv"))
    num_batches = int(custom.get("num_batches", 100))
    miscal_level = custom.get("orbital_miscal_noise_level") or None
    miscal_level_per_task = custom.get("orbital_miscal_noise_level_per_task_group") or None

    if not checkpoints:
        raise ValueError("Pass at least one checkpoint path via checkpoints=path1.pth")
    if not data_paths:
        raise ValueError("Pass at least one data path via data_paths=path1.zarr,path2.zarr")

    args = load_args(hydra_argv)

    csv_miscal_label = miscal_level_per_task or miscal_level or "none"

    print(f"Checkpoints ({len(checkpoints)}):")
    for p in checkpoints:
        print(f"  {p}")
    print(f"Data paths ({len(data_paths)}):")
    for p in data_paths:
        print(f"  {p}")
    print(f"miscal_level={csv_miscal_label}  num_batches={num_batches}  output_csv={output_csv}")
    print("-" * 80)

    amp_dtype = pick_amp_dtype()
    print(f"AMP dtype: {amp_dtype}")

    for ckpt_path in checkpoints:
        args_copy = deepcopy(args)
        model, ckpt_step = load_model(args_copy, ckpt_path)
        preprocessor = make_preprocessor(
            args_copy,
            orbital_miscal_noise_level=miscal_level,
            orbital_miscal_noise_level_per_task_group=miscal_level_per_task,
        )
        tokenizer = make_tokenizer(args_copy)

        ckpt_results = {}
        for data_path in data_paths:
            print(f"\nEvaluating on: {data_path}")
            loader = make_loader(args_copy, data_path)
            metrics, n = eval_scalar(
                model, tokenizer, preprocessor, loader,
                num_batches, amp_dtype, args_copy.relative_action,
            )
            ckpt_results[data_path] = (metrics, n)

        print_scalar_results(ckpt_path, ckpt_step, ckpt_results,
                             extra=f"miscal={csv_miscal_label}")
        rows = [
            metric_row(metrics, step=ckpt_step, miscal_level=csv_miscal_label,
                       dataset=Path(p).stem, n_samples=n)
            for p, (metrics, n) in ckpt_results.items()
        ]
        write_csv_rows(output_csv, CSV_HEADER, rows)
        print(f"Rows written to {output_csv}")

        del model
        torch.cuda.empty_cache()

    print(f"\nAll results written to {output_csv}")


if __name__ == "__main__":
    main()
