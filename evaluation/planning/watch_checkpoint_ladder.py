#!/usr/bin/env python3
"""Watch one or more training directories and submit new checkpoint-ladder evals."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from .checkpoint_ladder import load_config, process_directory, repo_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", nargs=2, action="append", metavar=("METHOD_ID", "CHECKPOINT_DIR"), required=True,
        help="Repeat for each arm, e.g. --run base train_logs/..._base_200k",
    )
    parser.add_argument("--config", default="instructions/eval_plans/sbrs_checkpoint_ladder_v01.json")
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--stop-step", type=int, default=200000)
    parser.add_argument("--once", action="store_true", help="Scan once instead of polling.")
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        raise ValueError("--poll-seconds must be positive")

    config = load_config(args.config)
    runs = [(method_id, repo_path(directory)) for method_id, directory in args.run]
    while True:
        all_finished = True
        for method_id, directory in runs:
            if not directory.is_dir():
                # A watcher can be submitted before its paired training job has
                # created the new run directory.  Keep polling rather than
                # treating that normal startup ordering as a failure.
                print(f"[watch] {method_id}: waiting for {directory}", flush=True)
                all_finished = False
                continue
            submitted = process_directory(
                config, directory, method_id, min_step=config["min_step"], max_step=args.stop_step,
                submit=True, force=False, ledger_path=directory / "checkpoint_ladder_submissions.json",
            )
            if submitted:
                print(f"[watch] {method_id}: submitted steps {submitted}", flush=True)
            expected = directory / f"interm_step_{args.stop_step}.pth"
            all_finished = all_finished and expected.exists()
        if args.once:
            print("[watch] scan complete", flush=True)
            return
        if all_finished:
            print("[watch] reached stop step for every arm", flush=True)
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
