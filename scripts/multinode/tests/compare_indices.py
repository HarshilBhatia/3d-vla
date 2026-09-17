#!/usr/bin/env python3
"""Compare the exact index sequences and RNG bytes of paired runs.

Why not compare weights: this stack is nondeterministic at ~1e-2 even with a
fixed seed (torch.compile, cudnn autotune, scatter atomics), so two identical
uninterrupted runs already differ by more than a broken resume would. Weight
comparison therefore cannot decide the question.

Both quantities here are discrete and so immune to that:

* index sequence -- which samples each batch contained. This is the thing that
  breaks silently when a resume restores only the epoch and restarts at batch 0.
* RNG state bytes -- advance by draw *count*, not draw values, so an extra or
  missing random draw shows up exactly.

The index log is written only when RESILIENCE_INDEX_LOG is set.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import torch

L = Path("train_logs/ResumeTest")
PAIRS = [
    ("phaseA", "phaseA2", "determinism control (same seed, two runs)"),
    ("phaseA", "phaseB", "1 GPU: uninterrupted vs resumed"),
    ("phaseC", "phaseD", "2 GPU: uninterrupted vs resumed"),
    ("phaseE", "phaseF", "diverse sampler: uninterrupted vs resumed"),
]


def read_markers(run: str, rank: int = 0) -> list[int]:
    """The skip= value of every epoch_start line, in order.

    A correctly resumed pair writes two: skip=0 from the first half, then a
    non-zero skip from the second. Two zeros means the resume restarted at
    batch 0 -- the exact bug this whole exercise exists to catch.
    """
    f = L / run / f"indices.rank{rank}.txt"
    if not f.exists():
        return []
    out = []
    for line in f.read_text().splitlines():
        if line.startswith("# epoch_start"):
            for tok in line.split():
                if tok.startswith("skip="):
                    out.append(int(tok.split("=", 1)[1]))
    return out


def coverage(idx: dict[int, str], steps: int) -> tuple[bool, str]:
    """Did the run actually advance through `steps` distinct batches?

    This is the assertion that discriminates. Comparing only the *overlap* of
    two logs is not enough: a resume that restarts at batch 0 re-yields the
    same head indices, so the overlap matches and the check passes while the
    run has silently trained the first 20 batches twice and never reached 20-39.
    A broken resume shows up here as max batch ~= steps/2 instead of ~= steps.
    """
    if not idx:
        return False, "no entries"
    nums = sorted(idx)
    gaps = [n for n in range(nums[0], nums[-1]) if n not in idx]
    if gaps:
        return False, f"gaps in coverage at {gaps[:5]}"
    if nums[-1] < steps - 1:
        return False, (f"only reached batch {nums[-1]} in {steps} steps "
                       f"(expected >= {steps - 1}) -- the resume re-consumed "
                       f"the start of the epoch instead of continuing")
    return True, f"batches {nums[0]}..{nums[-1]}, {len(nums)} distinct"


def read_indices(run: str, rank: int = 0) -> dict[int, str]:
    """batch number -> its index list. Later entries win.

    Prefetch means a run's sampler is consumed a few batches beyond the last
    step it actually trained, and the two halves of a resumed pair append to
    one file, so the same batch number can appear twice. Identical values are
    fine; conflicting ones are reported by the caller.
    """
    f = L / run / f"indices.rank{rank}.txt"
    if not f.exists():
        return {}
    out: dict[int, str] = {}
    for line in f.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        n, _, rest = line.partition(" ")
        try:
            out[int(n)] = rest
        except ValueError:
            continue
    return out


def rng_digest(st: dict) -> str:
    h = hashlib.sha256()
    h.update(torch.as_tensor(st["torch"]).cpu().numpy().tobytes())
    if st.get("cuda") is not None:
        h.update(torch.as_tensor(st["cuda"]).cpu().numpy().tobytes())
    npst = st["numpy"]
    h.update(np.asarray(npst[1]).tobytes())
    h.update(str(npst[2]).encode())
    h.update(repr(st["python"]).encode())
    return h.hexdigest()[:16]


def compare(a: str, b: str, label: str) -> bool | None:
    pa, pb = L / a / "last.pth", L / b / "last.pth"
    if not (pa.exists() and pb.exists()):
        print(f"{label:<44} SKIP (no checkpoint)")
        return None
    da = torch.load(pa, map_location="cpu", weights_only=False)
    db = torch.load(pb, map_location="cpu", weights_only=False)
    ok = True

    if da.get("iter") != db.get("iter"):
        print(f"{label:<44} FAIL step {da.get('iter')} vs {db.get('iter')}")
        return False
    step = da["iter"]

    # -- RNG bytes, every rank
    ra, rb = da.get("rng"), db.get("rng")
    if not ra or not rb:
        print(f"{label:<44} WARN no RNG in checkpoint")
    else:
        ha, hb = [rng_digest(s) for s in ra], [rng_digest(s) for s in rb]
        if ha != hb:
            ok = False
            print(f"{label:<44} FAIL rng differs: {ha} vs {hb}")

    # -- index sequence, every rank
    nranks = len(ra or [1])
    checked = 0
    for r in range(nranks):
        ia, ib = read_indices(a, r), read_indices(b, r)
        if not ia or not ib:
            print(f"{label:<44} WARN rank {r}: no index log "
                  f"(set RESILIENCE_INDEX_LOG)")
            continue
        common = sorted(set(ia) & set(ib))
        if not common:
            print(f"{label:<44} WARN rank {r}: no overlapping batches")
            continue
        bad = [n for n in common if ia[n] != ib[n]]
        checked += len(common)
        if bad:
            ok = False
            n = bad[0]
            print(f"{label:<44} FAIL rank {r}: {len(bad)}/{len(common)} batches "
                  f"differ, first at batch {n}")
            print(f"{'':<44}   A: {ia[n][:70]}")
            print(f"{'':<44}   B: {ib[n][:70]}")

        # Both runs must have advanced through `step` distinct batches.
        for tag, idx in ((a, ia), (b, ib)):
            good, why = coverage(idx, step)
            if not good:
                ok = False
                print(f"{label:<44} FAIL {tag} rank {r}: {why}")

        # And the resumed half must actually have skipped.
        marks = read_markers(b, r)
        if len(marks) >= 2 and not any(m > 0 for m in marks[1:]):
            ok = False
            print(f"{label:<44} FAIL {b} rank {r}: epoch_start skips are "
                  f"{marks} -- the second half restarted at batch 0")

    if ok:
        marks = read_markers(b, 0)
        print(f"{label:<44} PASS step {step}, rng identical across "
              f"{nranks} rank(s), {checked} batches index-identical, "
              f"skips={marks}")
    return ok


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", help="directory holding the run dirs "
                                   "(default: train_logs/ResumeTest)")
    ap.add_argument("--pair", action="append", default=[], metavar="A:B:LABEL",
                    help="compare run A against run B; repeatable. "
                         "Without any, the default ResumeTest pairs are used.")
    args = ap.parse_args()

    global L
    if args.base:
        L = Path(args.base)
    pairs = PAIRS
    if args.pair:
        pairs = []
        for spec in args.pair:
            parts = spec.split(":", 2)
            if len(parts) < 2:
                raise SystemExit(f"--pair needs A:B[:LABEL], got {spec!r}")
            a, b = parts[0], parts[1]
            pairs.append((a, b, parts[2] if len(parts) > 2 else f"{a} vs {b}"))

    res = [compare(a, b, lbl) for a, b, lbl in pairs]
    print()
    print(f"{res.count(True)} pass, {res.count(False)} fail, {res.count(None)} skipped")
    return 1 if False in res else 0


if __name__ == "__main__":
    raise SystemExit(main())
