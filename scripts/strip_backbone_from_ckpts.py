"""
Strip frozen backbone weights (encoder.backbone / encoder.text_encoder) from checkpoints.
Works on any .pth file: last.pth, best.pth, interm_step_*.pth, etc.

Usage:
    python scripts/strip_backbone_from_ckpts.py <log_dir> [--dry-run]

    --dry-run   Print what would change without writing anything
"""
import argparse
from pathlib import Path
import torch


BACKBONE_SUBSTRINGS = ("encoder.backbone", "encoder.text_encoder")


def is_backbone_key(k: str) -> bool:
    return any(s in k for s in BACKBONE_SUBSTRINGS)


def strip_state_dict(sd: dict) -> tuple[dict, int, float]:
    """Return (stripped_dict, n_removed, bytes_removed)."""
    keep, removed_bytes = {}, 0
    for k, v in sd.items():
        if is_backbone_key(k):
            removed_bytes += v.nbytes
        else:
            keep[k] = v
    return keep, len(sd) - len(keep), removed_bytes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.log_dir.is_dir():
        raise SystemExit(f"Not a directory: {args.log_dir}")

    ckpts = sorted(args.log_dir.rglob("*.pth"))
    if not ckpts:
        print("No .pth files found.")
        return

    for p in ckpts:
        state = torch.load(p, map_location="cpu", weights_only=False)

        total_removed_bytes = 0
        for key in ("weight", "ema_weight"):
            if not isinstance(state.get(key), dict):
                continue
            stripped, n_removed, removed_bytes = strip_state_dict(state[key])
            if n_removed == 0:
                continue
            total_removed_bytes += removed_bytes
            if not args.dry_run:
                state[key] = stripped

        if total_removed_bytes == 0:
            print(f"  {p.name}: already clean, skipping")
            continue

        before = p.stat().st_size
        tag = "[DRY RUN] " if args.dry_run else ""
        if args.dry_run:
            print(f"{tag}{p}: would free ~{total_removed_bytes / 1e9:.2f} GB")
        else:
            torch.save(state, p)
            after = p.stat().st_size
            print(f"{p}: {before / 1e9:.2f} GB → {after / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
