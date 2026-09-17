#!/usr/bin/env python3
"""Scaling harness: same config at N=1,2,3,4.

Run this before touching gradient accumulation, NCCL channel counts, or
anything else. Without a measured efficiency curve you cannot tell whether a
slow run is the fabric, the dataloader, or the model -- and the answer on this
cluster is usually the dataloader.

Submits one job per N through submit.py (--detach), then reports step time and
scaling efficiency from each run's step-time log. Efficiency is relative to
N=1, so N=1 must be in the list.

    python scripts/multinode/bench/scaling.py config/example.yaml --n 1 2 3 4
    python scripts/multinode/bench/scaling.py config/example.yaml --report
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
SUBMIT = HERE.parent / "submit.py"


def submit_one(cfg_path: Path, n: int, out_dir: Path, extra: list[str]) -> str:
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["nodes"] = n
    # Separate checkpoint + log dirs per N, or the runs resume each other's
    # state and the comparison is meaningless.
    cfg["ckpt_dir"] = str(out_dir / f"n{n}" / "ckpt")
    cfg["runlog"] = str(out_dir / "runs.jsonl")
    cfg["log_dir"] = str(out_dir / "slurm")
    steps_log = out_dir / f"n{n}" / "steps.jsonl"
    cfg["train_cmd"] = f"{cfg['train_cmd']} --step-log {steps_log}"
    Path(cfg["ckpt_dir"]).mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile("w", suffix=f"-n{n}.yaml", delete=False) as f:
        yaml.safe_dump(cfg, f)
        tmp = f.name
    r = subprocess.run([sys.executable, str(SUBMIT), tmp, "--detach", *extra],
                       capture_output=True, text=True)
    print(r.stdout.strip())
    if r.returncode != 0:
        print(r.stderr.strip(), file=sys.stderr)
        return ""
    for line in r.stdout.splitlines():
        if "submitted with constraint" in line:
            return line.split()[2]
    return ""


def report(out_dir: Path, ns: list[int]) -> int:
    rows = []
    for n in ns:
        log = out_dir / f"n{n}" / "steps.jsonl"
        if not log.exists():
            rows.append((n, None, None))
            continue
        recs = [json.loads(l) for l in log.read_text().splitlines() if l.strip()]
        # Drop the first 10% as warmup: cudnn autotune, NCCL channel setup and
        # the first dataloader fill all land there and are not steady state.
        warm = recs[max(1, len(recs) // 10):]
        if not warm:
            rows.append((n, None, None))
            continue
        step_s = sum(r["step_time"] for r in warm) / len(warm)
        samples = sum(r.get("global_batch", 0) for r in warm) / len(warm)
        rows.append((n, step_s, samples / step_s if step_s else None))

    base = next((t for n, t, _ in rows if n == 1 and t), None)
    print(f"{'N':>3} {'step (s)':>10} {'samples/s':>11} {'speedup':>8} {'efficiency':>11}")
    print("-" * 48)
    for n, step_s, thru in rows:
        if step_s is None:
            print(f"{n:>3} {'no data':>10}")
            continue
        sp = base / step_s if base else float("nan")
        eff = sp / n if base else float("nan")
        print(f"{n:>3} {step_s:>10.3f} {thru or 0:>11.1f} {sp:>8.2f} {eff * 100:>10.0f}%")
    if base is None:
        print("\nno N=1 baseline — speedup and efficiency are undefined")
    else:
        print("\nefficiency below ~70% at small N usually means the dataloader, "
              "not the fabric:\n  re-run with more --num-workers before tuning NCCL.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--n", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--out", default="logs/multinode/scaling")
    ap.add_argument("--report", action="store_true", help="only summarise existing runs")
    args, extra = ap.parse_known_args()

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.report:
        return report(out, sorted(args.n))

    if 1 not in args.n:
        print("[scaling] warning: N=1 missing, efficiency cannot be computed")
    for n in sorted(args.n):
        jid = submit_one(Path(args.config), n, out, extra)
        print(f"[scaling] N={n} -> job {jid or 'FAILED'}")
    print(f"\n[scaling] when the jobs finish:\n"
          f"  python {Path(__file__).relative_to(Path.cwd())} {args.config} "
          f"--report --n {' '.join(map(str, sorted(args.n)))} --out {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
