"""Canonical command-line entry points for evaluation."""

from __future__ import annotations

import sys
from pathlib import Path

from evaluation.checkpoints import load_model_for_evaluation
from evaluation.online.runner import resolve_runtime_paths, run_online_evaluation
from utils.hydra_utils import get_config, get_config_path


def online_rlbench_main() -> None:
    """Run online RLBench evaluation with the legacy Hydra override surface."""
    args = get_config(overrides=sys.argv[1:], config_name="config", config_path=get_config_path())
    # Existing commands resolve relative eval paths from the historical script.
    legacy_base = Path(__file__).resolve().parents[1] / "online_evaluation_rlbench"
    resolve_runtime_paths(args, legacy_base)
    print("Arguments:")
    for key, value in sorted(vars(args).items()):
        print(f"  {key}: {value}")
    print("-" * 100)
    run_online_evaluation(args, model_loader=load_model_for_evaluation)


if __name__ == "__main__":
    online_rlbench_main()
