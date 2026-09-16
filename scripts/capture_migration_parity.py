#!/usr/bin/env python3
# Kept under scripts/ (rather than scripts/testing/) so it is tracked.
"""Capture a reviewed golden output for the opt-in migration parity test.

Run this only on a known-good revision.  It never overwrites a case or golden
output unless ``--force`` is explicit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path

import torch

from evaluation.checkpoints import load_model_for_evaluation
from utils.hydra_utils import get_config, get_config_path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def deterministic_torch():
    old = {
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
    }
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(old["deterministic"])
        torch.backends.cudnn.deterministic = old["cudnn_deterministic"]
        torch.backends.cudnn.benchmark = old["cudnn_benchmark"]
        torch.backends.cuda.matmul.allow_tf32 = old["matmul_tf32"]
        torch.backends.cudnn.allow_tf32 = old["cudnn_tf32"]


def to_device(value, device):
    if hasattr(value, "to"):
        return value.to(device)
    if isinstance(value, dict):
        return {key: to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(to_device(item, device) for item in value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True, help="torch-save dict of model.forward arguments")
    parser.add_argument("--case", type=Path, required=True, help="new JSON parity-case path")
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("capture requires CUDA so it matches the GPU parity contract")
    for path in (args.checkpoint, args.input):
        if not path.is_file():
            raise FileNotFoundError(path)
    output = args.case.with_suffix(".expected_output.pt")
    if not args.force and (args.case.exists() or output.exists()):
        raise FileExistsError("case or output already exists; use --force only after reviewed intent")

    batch = torch.load(args.input, map_location="cpu", weights_only=False)
    if not isinstance(batch, dict):
        raise TypeError("--input must contain a dictionary of model.forward keyword arguments")
    cfg = get_config(overrides=[], config_name="config", config_path=get_config_path())
    cfg.checkpoint = args.checkpoint.resolve()
    with deterministic_torch():
        model = load_model_for_evaluation(cfg)
        if hasattr(model, "_orig_mod"):
            raise RuntimeError("capture must use eager mode, not torch.compile")
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        with torch.inference_mode():
            output_tensor = model(**to_device(batch, torch.device("cuda")), run_inference=True).cpu()

    args.case.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_tensor, output)
    case = {
        "schema_version": 1,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "input": str(args.input.resolve()),
        "expected_output": str(output.resolve()),
        "seed": args.seed,
        "rtol": 0.0,
        "atol": 0.0,
    }
    args.case.write_text(json.dumps(case, indent=2, sort_keys=True) + "\n")
    print(f"Wrote case: {args.case}")
    print(f"Wrote golden output: {output}")


if __name__ == "__main__":
    main()
