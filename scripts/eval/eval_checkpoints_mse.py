"""Evaluate a list of checkpoints on multiple zarr data paths; report offline L2/MSE metrics.

Usage (single GPU, no torchrun needed):
    python scripts/eval/eval_checkpoints_mse.py \
        checkpoints=/path/to/ckpt1.pth,/path/to/ckpt2.pth \
        data_paths=/path/to/val1.zarr,/path/to/val2.zarr,/path/to/val3.zarr \
        val_instructions=instructions/peract2/instructions.json \
        bimanual=true \
        output_csv=results/checkpoint_mse/step_10000.csv \
        num_batches=100 \
        "$@"

Results are written as CSV rows — one row per (checkpoint, data_path):
    step, dataset, n_samples, traj_pos_l2, traj_rot_l1, traj_pos_acc_001, traj_rot_acc_0025, traj_gripper

When running as a SLURM array job (one task per checkpoint), pass a per-checkpoint
output_csv path so parallel tasks never write to the same file simultaneously.
Use scripts/eval/plot_checkpoint_mse.py to aggregate all CSVs and plot.

num_batches controls how many batches are evaluated per (checkpoint, data_path) pair,
giving approximately num_batches * batch_size_val total samples.

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


CSV_HEADER = ["step", "dataset", "n_samples"] + METRIC_KEYS


def main():
    custom, hydra_argv = extract_script_args(sys.argv[1:], BASE_SCRIPT_KEYS)

    checkpoints = parse_csv_list(custom.get("checkpoints"))
    data_paths = parse_csv_list(custom.get("data_paths"))
    output_csv = Path(custom.get("output_csv", "results/checkpoint_mse/results.csv"))
    num_batches = int(custom.get("num_batches", 100))

    if not checkpoints:
        raise ValueError("Pass at least one checkpoint path via checkpoints=path1.pth,path2.pth")
    if not data_paths:
        raise ValueError("Pass at least one data path via data_paths=path1.zarr,path2.zarr")

    args = load_args(hydra_argv)

    print(f"Checkpoints ({len(checkpoints)}):")
    for p in checkpoints:
        print(f"  {p}")
    print(f"Data paths ({len(data_paths)}):")
    for p in data_paths:
        print(f"  {p}")
    print(f"num_batches={num_batches}  output_csv={output_csv}")
    print("-" * 80)

    amp_dtype = pick_amp_dtype()
    print(f"AMP dtype: {amp_dtype}")

    for ckpt_path in checkpoints:
        args_copy = deepcopy(args)
        model, ckpt_step = load_model(args_copy, ckpt_path)
        preprocessor = make_preprocessor(args_copy)
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

        print_scalar_results(ckpt_path, ckpt_step, ckpt_results)
        rows = [
            metric_row(metrics, step=ckpt_step, dataset=Path(p).stem, n_samples=n)
            for p, (metrics, n) in ckpt_results.items()
        ]
        write_csv_rows(output_csv, CSV_HEADER, rows)
        print(f"Rows written to {output_csv}")

        del model
        torch.cuda.empty_cache()

    print(f"\nAll results written to {output_csv}")


if __name__ == "__main__":
    main()
