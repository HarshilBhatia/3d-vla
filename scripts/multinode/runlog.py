#!/usr/bin/env python3
"""Run log and live pool census.

A snapshot of which GPU pools are usable goes stale within hours -- nodes drain,
labs fill up, features get relabelled. So rather than baking a table into the
docs, re-derive it:

    python scripts/multinode/runlog.py pools -p all
    python scripts/multinode/runlog.py pools -p all --nodes 3
    python scripts/multinode/runlog.py summary

``summary`` aggregates what actually happened across runs: which constraint you
landed on, measured busbw, whether you got preempted and when. After a dozen
runs that tells you which pools work, which a one-off census cannot.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from submit import NodeInfo, node_table, pool_for  # noqa: E402


def cmd_pools(args: argparse.Namespace) -> int:
    nodes = node_table()
    part = args.partition
    by_feat: dict[str, list[NodeInfo]] = defaultdict(list)
    for n in pool_for(nodes, part, None):
        by_feat[n.feature].append(n)

    # A node in no lab partition cannot be preempted: on grogu `all` is tier 1
    # and only a *higher* tier evicts you, so a node whose only other partition
    # is root-only is structurally safe rather than merely quiet.
    def preempt_safe(n: NodeInfo) -> bool:
        others = [p for p in n.partitions if p != part]
        return all(p in ("facunix",) for p in others)

    print(f"partition={part}   (usable nodes only; DOWN/DRAIN excluded)\n")
    print(f"{'constraint':<16} {'nodes':>5} {'w/free':>7} {'free gpu':>9} "
          f"{'safe':>5}  nodes")
    print("-" * 92)
    rows = sorted(by_feat.items(), key=lambda kv: -sum(n.gpus_free for n in kv[1]))
    for feat, ns in rows:
        free_nodes = [n for n in ns if n.gpus_free > 0]
        safe = [n for n in ns if preempt_safe(n)]
        flag = ""
        if args.nodes and len(ns) < args.nodes:
            flag = f"  <-- too small for N={args.nodes}"
        print(f"{feat:<16} {len(ns):>5} {len(free_nodes):>7} "
              f"{sum(n.gpus_free for n in ns):>9} {len(safe):>5}  "
              f"{','.join(n.name for n in sorted(ns, key=lambda x: x.name))}{flag}")
    print()
    print("safe = node belongs to no partition that can preempt this one")
    if args.nodes:
        ok = [f for f, ns in rows if len(ns) >= args.nodes]
        print(f"\nconstraints that can satisfy N={args.nodes}: {ok or 'NONE'}")
    return 0


def cmd_summary(args: argparse.Namespace) -> int:
    p = Path(args.runlog)
    if not p.exists():
        print(f"no run log at {p}")
        return 1
    recs = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]

    jobs: dict[str, dict] = defaultdict(dict)
    for r in recs:
        jid = r.get("job_id")
        if not jid:
            continue
        j = jobs[jid]
        kind = r.get("kind")
        if kind == "submit":
            j.update(constraint=r.get("constraint"), nodes=r.get("nodes"),
                     partition=r.get("partition"), submitted=r["ts"])
        elif kind == "running":
            j["nodelist"] = r.get("nodelist")
            j["started"] = r["ts"]
        elif kind == "preflight":
            j["gpu"] = r.get("gpu")
            j["busbw"] = r.get("busbw_gbs")
            j["nodelist"] = r.get("nodelist") or j.get("nodelist")
            j["pf_fail"] = len(r.get("failures") or [])
        elif kind == "preempted":
            j["preempts"] = r.get("requeue", j.get("preempts", 0))
            j["preempt_at"] = r["ts"]
        elif kind == "fallback":
            j.setdefault("fallbacks", []).append(f"{r.get('from')}->{r.get('to')}")
        elif kind == "final":
            j["state"] = r.get("state")

    print(f"{'job':>9} {'constraint':<14} {'N':>2} {'gpu':<22} {'busbw':>6} "
          f"{'wait':>6} {'ran':>7} {'pmt':>4} {'state':<10} nodes")
    print("-" * 118)
    for jid, j in sorted(jobs.items()):
        wait = (j.get("started", 0) - j.get("submitted", 0)) if j.get("started") else 0
        ran = 0.0
        if j.get("started"):
            end = j.get("preempt_at") or j.get("submitted", 0)
            ran = max(0.0, (j.get("preempt_at") or 0) - j["started"]) if j.get("preempt_at") else 0.0
        bw = f"{j['busbw']:.1f}" if j.get("busbw") else "-"
        waited = f"{wait / 60:.0f}m" if wait else "-"
        elapsed = f"{ran / 3600:.1f}h" if ran else "-"
        print(f"{jid:>9} {str(j.get('constraint') or 'any'):<14} "
              f"{str(j.get('nodes') or '?'):>2} {str(j.get('gpu') or '?')[:22]:<22} "
              f"{bw:>6} {waited:>6} {elapsed:>7} "
              f"{str(j.get('preempts') or 0):>4} {str(j.get('state') or 'running'):<10} "
              f"{j.get('nodelist') or '-'}")

    pre = sum(1 for j in jobs.values() if j.get("preempts"))
    pf = sum(1 for j in jobs.values() if j.get("pf_fail"))
    print(f"\n{len(jobs)} run(s): {pre} hit preemption, {pf} failed preflight")
    by_c: dict[str, list[float]] = defaultdict(list)
    for j in jobs.values():
        if j.get("busbw"):
            by_c[str(j.get("constraint") or "any")].append(j["busbw"])
    if by_c:
        print("\nmeasured busbw by constraint:")
        for c, vs in sorted(by_c.items()):
            print(f"  {c:<14} n={len(vs):<3} mean={sum(vs)/len(vs):.2f} GB/s "
                  f"min={min(vs):.2f} max={max(vs):.2f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("pools", help="live homogeneous-pool census for a partition")
    p1.add_argument("-p", "--partition", default="all")
    p1.add_argument("--nodes", type=int, default=0, help="flag pools too small for N")
    p1.set_defaults(fn=cmd_pools)

    p2 = sub.add_parser("summary", help="aggregate the run log")
    p2.add_argument("--runlog", default="logs/multinode/runs.jsonl")
    p2.set_defaults(fn=cmd_summary)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
