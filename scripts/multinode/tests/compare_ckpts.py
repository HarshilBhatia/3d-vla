"""Compare two last.pth files tensor by tensor.

Bit-identity is the assertion that matters here: if a resume restores the data
order and every RNG stream exactly, the weights after N steps must be the same
bits as an uninterrupted run's. Anything else means some state was missed.
"""
import os
import sys
from pathlib import Path
import torch


def load(p):
    d = torch.load(p, map_location="cpu", weights_only=False)
    return d["weight"], d.get("iter"), d


def cmp(a_path, b_path, label):
    if not (Path(a_path).exists() and Path(b_path).exists()):
        print(f"{label:<34} SKIP (missing: "
              f"{[p for p in (a_path, b_path) if not Path(p).exists()]})")
        return None
    wa, ia, da = load(a_path)
    wb, ib, db = load(b_path)
    ka, kb = set(wa), set(wb)
    if ka != kb:
        print(f"{label:<34} FAIL key mismatch ({len(ka - kb)} only-A, {len(kb - ka)} only-B)")
        return False
    if ia != ib:
        print(f"{label:<34} FAIL step mismatch: {ia} vs {ib}")
        return False

    identical, worst, worst_k = 0, 0.0, None
    for k in sorted(ka):
        x, y = wa[k].float(), wb[k].float()
        if torch.equal(wa[k], wb[k]):
            identical += 1
            continue
        d = (x - y).abs().max().item()
        if d > worst:
            worst, worst_k = d, k

    n = len(ka)
    has_rng = "rng" in da or "rng" in db
    if identical == n:
        print(f"{label:<34} PASS bit-identical  ({n}/{n} tensors, step {ia})")
        return True
    print(f"{label:<34} DIFF {identical}/{n} identical, "
          f"max|Δ|={worst:.3e} on {worst_k} (step {ia}, rng_in_ckpt={has_rng})")
    return False


if __name__ == "__main__":
    L = Path("train_logs/ResumeTest")
    # main.py resolves base_log_dir relative to its own file, so the pre-patch
    # tree writes into its own root, not this one.
    OLD_L = Path(os.environ.get("OLD_TREE", Path.home() / "3dfa_bitcheck")) / "train_logs/ResumeTest"
    checks = [
        # Two identical seeded runs: calibrates how deterministic the stack is.
        # If this is not bit-identical, nothing downstream can be either, and
        # the resume checks must be read against this tolerance.
        (L / "phaseA/last.pth", L / "phaseA2/last.pth", "determinism: same seed, two runs"),
        (L / "phaseA/last.pth", L / "phaseB/last.pth", "1 GPU: uninterrupted vs resumed"),
        (L / "phaseC/last.pth", L / "phaseD/last.pth", "2 GPU: uninterrupted vs resumed"),
        (L / "phaseE/last.pth", L / "phaseF/last.pth", "diverse sampler: uninterr vs resumed"),
    ]
    results = [cmp(a, b, lbl) for a, b, lbl in checks]
    print()
    ok = [r for r in results if r is True]
    bad = [r for r in results if r is False]
    print(f"{len(ok)} pass, {len(bad)} fail, {results.count(None)} skipped")
    sys.exit(1 if bad else 0)
