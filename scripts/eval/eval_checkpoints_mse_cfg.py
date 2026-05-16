"""Evaluate checkpoints on multiple zarr data paths across classifier-free guidance scales.

Usage (single GPU, no torchrun needed):
    python scripts/eval/eval_checkpoints_mse_cfg.py \
        checkpoints=/path/to/ckpt.pth \
        data_paths=/path/to/val1.zarr,/path/to/val2.zarr \
        val_instructions=instructions/peract/instructions.json \
        bimanual=false \
        cfg_scale=4 \
        output_csv=results/checkpoint_mse_cfg/cfg_4.csv \
        num_batches=100

Identical to eval_checkpoints_mse.py but plumbs `cfg_scale` into the model's
inference call so classifier-free guidance is applied. The CSV gains a
`cfg_scale` column. cfg_scale=null/none/<empty> runs vanilla conditional
inference (no uncond pass) for a baseline.

cfg_scale=0 yields pure unconditional generation (the CFG formula collapses to
`out_uncond` when scale=0). It still runs the cond forward pass per timestep
which is wasted, but for offline eval that's fine — no separate script needed.

When running as a SLURM array job (one task per cfg value), pass a per-task
output_csv so parallel tasks never write the same file.

Dataset type, model architecture, and all other config are auto-detected from
each checkpoint's saved config dict — no need to pass arch flags on the CLI.
"""
import sys
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from scripts.eval.eval_utils import (
    BASE_SCRIPT_KEYS, METRIC_KEYS, cfg_label,
    eval_scalar, extract_script_args, load_args, load_model,
    make_loader, make_preprocessor, make_tokenizer, metric_row,
    parse_cfg_scale, parse_csv_list, pick_amp_dtype,
    print_scalar_results, write_csv_rows,
)


SCRIPT_KEYS = BASE_SCRIPT_KEYS | {"cfg_scale"}
CSV_HEADER = ["step", "cfg_scale", "dataset", "n_samples"] + METRIC_KEYS


def main():
    custom, hydra_argv = extract_script_args(sys.argv[1:], SCRIPT_KEYS)

    checkpoints = parse_csv_list(custom.get("checkpoints"))
    data_paths = parse_csv_list(custom.get("data_paths"))
    output_csv = Path(custom.get("output_csv", "results/checkpoint_mse_cfg/results.csv"))
    num_batches = int(custom.get("num_batches", 100))
    cfg_scale = parse_cfg_scale(custom.get("cfg_scale"))

    if not checkpoints:
        raise ValueError("Pass at least one checkpoint path via checkpoints=path.pth")
    if not data_paths:
        raise ValueError("Pass at least one data path via data_paths=path1.zarr,path2.zarr")

    args = load_args(hydra_argv)

    print(f"Checkpoints ({len(checkpoints)}):")
    for p in checkpoints:
        print(f"  {p}")
    print(f"Data paths ({len(data_paths)}):")
    for p in data_paths:
        print(f"  {p}")
    print(f"cfg_scale={cfg_label(cfg_scale)}  num_batches={num_batches}  output_csv={output_csv}")
    print("-" * 80)

    amp_dtype = pick_amp_dtype()
    print(f"AMP dtype: {amp_dtype}")

    for ckpt_path in checkpoints:
        if output_csv.exists():
            print(f"Skipping {ckpt_path} — output CSV already exists: {output_csv}")
            continue

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
                cfg_scale=cfg_scale,
            )
            ckpt_results[data_path] = (metrics, n)

        print_scalar_results(ckpt_path, ckpt_step, ckpt_results,
                             extra=f"cfg_scale={cfg_label(cfg_scale)}")
        rows = [
            metric_row(metrics,
                       step=ckpt_step, cfg_scale=cfg_label(cfg_scale),
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
