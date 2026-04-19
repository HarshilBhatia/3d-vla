"""
Build peract_perturbations/ from peract_packaged_perturbed/ with exactly
100 train + 25 val episodes per task (matching peract_packaged layout).

Episodes are selected by sorting all .dat files in all variation folders,
taking the first N, then hard-linking them into the output tree with
sequential ep numbers (ep0.dat, ep1.dat, ...) within each variation folder.
"""
import argparse
import os
import shutil
from collections import defaultdict
from pathlib import Path

TASKS = [
    "place_cups", "close_jar", "insert_onto_square_peg", "light_bulb_in",
    "meat_off_grill", "open_drawer", "place_shape_in_shape_sorter",
    "place_wine_at_rack_location", "push_buttons", "put_groceries_in_cupboard",
    "put_item_in_drawer", "put_money_in_safe", "reach_and_drag",
    "slide_block_to_color_target", "stack_blocks", "stack_cups",
    "sweep_to_dustpan_of_size", "turn_tap",
]


def _ep_sort_key(path: Path):
    """Sort .dat files by variation number then episode number."""
    var = path.parent.name  # e.g. "close_jar+3"
    try:
        var_num = int(var.split("+")[-1])
    except ValueError:
        var_num = 0
    ep_str = path.stem  # e.g. "ep7042"
    try:
        ep_num = int(ep_str.replace("ep", ""))
    except ValueError:
        ep_num = 0
    return (var_num, ep_num)


def link_or_copy(src: Path, dst: Path, use_symlink: bool):
    if use_symlink:
        dst.symlink_to(src.resolve())
    else:
        os.link(src, dst)  # hard link (same filesystem, no extra disk space)


def build(src_root: Path, dst_root: Path, n_train: int, n_val: int,
          use_symlink: bool, dry_run: bool):
    summary = []
    for task in TASKS:
        for split, n_target in [("train", n_train), ("val", n_val)]:
            # Collect all .dat files for this task/split
            all_files = sorted(
                (src_root / split).glob(f"{task}+*/ep*.dat"),
                key=_ep_sort_key,
            )

            if len(all_files) < n_target:
                print(f"  [WARN] {split}/{task}: only {len(all_files)} eps "
                      f"(need {n_target}) – skipping")
                summary.append((task, split, len(all_files), n_target, "SKIP"))
                continue

            selected = all_files[:n_target]

            # Group selected files by variation folder name
            by_var: dict[str, list[Path]] = defaultdict(list)
            for f in selected:
                by_var[f.parent.name].append(f)

            # Write into dst with sequential ep numbers per variation folder
            global_idx = 0
            for var_name in sorted(by_var, key=lambda v: int(v.split("+")[-1])):
                var_dir = dst_root / split / var_name
                if not dry_run:
                    var_dir.mkdir(parents=True, exist_ok=True)
                for src_file in sorted(by_var[var_name], key=_ep_sort_key):
                    dst_file = var_dir / f"ep{global_idx}.dat"
                    global_idx += 1
                    if dry_run:
                        print(f"  [DRY] {dst_file} <- {src_file}")
                    else:
                        if dst_file.exists():
                            dst_file.unlink()
                        link_or_copy(src_file, dst_file, use_symlink)

            print(f"  [OK] {split}/{task}: wrote {global_idx}/{n_target} eps")
            summary.append((task, split, global_idx, n_target, "OK"))

    print("\n=== Summary ===")
    print(f"  {'Task':<42}  {'Split':<5}  {'Written':>7}  {'Target':>6}  Status")
    for row in summary:
        print(f"  {row[0]:<42}  {row[1]:<5}  {row[2]:>7}  {row[3]:>6}  {row[4]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default="Peract_packaged_perturbed",
                    help="Source root (default: Peract_packaged_perturbed)")
    ap.add_argument("--dst", default="peract_perturbations",
                    help="Destination root (default: peract_perturbations)")
    ap.add_argument("--n_train", type=int, default=100)
    ap.add_argument("--n_val", type=int, default=25)
    ap.add_argument("--symlink", action="store_true",
                    help="Use symlinks instead of hard links (cross-filesystem safe)")
    ap.add_argument("--dry_run", action="store_true",
                    help="Print what would be done without writing")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    assert src.exists(), f"Source not found: {src}"

    if not args.dry_run:
        if dst.exists():
            print(f"[INFO] Removing existing {dst} ...")
            shutil.rmtree(dst)
        dst.mkdir(parents=True)

    print(f"Building {dst}/ from {src}/")
    print(f"  {args.n_train} train + {args.n_val} val per task")
    print(f"  Mode: {'symlink' if args.symlink else 'hard link'}")
    print()
    build(src, dst, args.n_train, args.n_val, args.symlink, args.dry_run)


if __name__ == "__main__":
    main()
