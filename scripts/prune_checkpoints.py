"""
Delete step_*.pth and interm_step_*.pth files from a log directory (recursive).

Usage:
    python scripts/prune_checkpoints.py <log_dir> [--dry-run]

    --dry-run   Print what would be deleted without actually deleting
"""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.log_dir.is_dir():
        raise SystemExit(f"Not a directory: {args.log_dir}")

    to_delete = sorted([
        *args.log_dir.rglob("step_*.pth"),
        *args.log_dir.rglob("interm_step_*.pth"),
    ])

    if not to_delete:
        print("Nothing to delete.")
        return

    total_bytes = sum(p.stat().st_size for p in to_delete if p.exists())
    print(f"{'[DRY RUN] ' if args.dry_run else ''}Deleting {len(to_delete)} file(s) "
          f"({total_bytes / 1e9:.2f} GB):")
    for p in to_delete:
        print(f"  {p}")

    if not args.dry_run:
        for p in to_delete:
            p.unlink(missing_ok=True)
        print("Done.")


if __name__ == "__main__":
    main()
