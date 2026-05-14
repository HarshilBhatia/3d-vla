"""
Strip interm_step_*.pth files down to weights only (removes optimizer, scaler, config, etc).

Usage:
    python scripts/strip_interm_checkpoints.py <log_dir> [--dry-run]
"""
import argparse
from pathlib import Path
import torch


KEEP_KEYS = {"weight", "ema_weight"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.log_dir.is_dir():
        raise SystemExit(f"Not a directory: {args.log_dir}")

    ckpts = sorted(args.log_dir.rglob("interm_step_*.pth"))
    if not ckpts:
        print("No interm_step_*.pth files found.")
        return

    for p in ckpts:
        before = p.stat().st_size
        state = torch.load(p, map_location="cpu", weights_only=False)
        stripped = {k: v for k, v in state.items() if k in KEEP_KEYS}
        removed = [k for k in state if k not in KEEP_KEYS]
        after_est = before  # placeholder for dry-run display

        if args.dry_run:
            print(f"[DRY RUN] {p.name} ({before / 1e9:.2f} GB) — would remove keys: {removed}")
        else:
            torch.save(stripped, p)
            after = p.stat().st_size
            print(f"{p.name}: {before / 1e9:.2f} GB → {after / 1e9:.2f} GB (removed: {removed})")


if __name__ == "__main__":
    main()
