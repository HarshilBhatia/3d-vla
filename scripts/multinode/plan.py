#!/usr/bin/env python3
"""Decide how to place a training run, from live cluster state.

Answers "what should I request right now, and what will I get?" using measured
performance rather than guesses, then writes a ready-to-submit config.

Why this exists: `runlog.py pools` counts free *GPUs*, which is not enough.
A request sized reasonably by GPU count sat with a two-day start estimate
because the node had 6 free GPUs and 53 GB of free host memory; and every node
with plenty of free RAM and >=4 GPUs was a 2080Ti, which this model cannot run
on at all. Placement has to consider GPUs, CPUs, memory, GPU model and HCA
together.

    python scripts/multinode/plan.py --gpus 4 --per-gpu-batch 8
    python scripts/multinode/plan.py --gpus 4 --walltime 2-00:00:00 \
        --partition shubhamlong --emit config/my_run.yaml
    python scripts/multinode/plan.py --gpus 8 --show-all
"""

from __future__ import annotations

import argparse
import math
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from submit import NodeInfo, parse_walltime, partition_maxtime  # noqa: E402

# ── measured performance, per GPU model ──────────────────────────────────────
# samp/s for ONE rank at per-GPU batch 8, on an otherwise quiet node.
# Measured 2026-09-16/17 on PerAct2 orbital 2d (2.55M trainable params).
# Re-measure with bench/scaling.py if the model or data pipeline changes.
RANK_THROUGHPUT_B8 = {
    "A6000": 71.9,
    "A5000": 63.6,
    "rtx3090": 53.1,
    "rtx6000": 16.9,      # Turing: no tf32/bf16, 91% of the step in forward
}
# Cards this model cannot run on at all, regardless of what Slurm advertises.
UNUSABLE = {
    "rtx2080ti": "10.6 GiB: too small for this model's per-GPU training batch (eval fits)",
}
# Never measured for this workload, so refuse to predict rather than guess.
UNMEASURED = {"H200", "6000Blackwell", "A6000Ada"}

# Per-rank efficiency vs a single GPU, all ranks on ONE node. Measured on
# A5000 at per-GPU batch 8. This is DDP's collective overhead: it appears with
# no network involved at all (bwd 13.0 -> 15.6 -> 23.6 -> 31.4 ms for
# 1 -> 2 -> 3 -> 8 ranks inside one node).
INTRA_NODE_EFF = {1: 1.000, 2: 0.960, 3: 0.896, 4: 0.875, 8: 0.852}

# Extra multiplier for spreading the same ranks across nodes. Small: at 3 ranks
# the backward time is identical over PCIe and over InfiniBand.
INTER_NODE_EFF = {1: 1.000, 2: 0.971, 3: 0.951, 4: 0.940}

# Larger per-GPU batch amortises the fixed collective cost. Measured at 3 ranks:
# batch 8 -> 87.3% efficiency, batch 16 -> 92.2%; and single-GPU throughput rose
# 63.6 -> 75.1 samp/s (+18%).
BATCH_SCALE = {8: 1.00, 16: 1.18}


def _interp(table: dict[int, float], n: int) -> float:
    if n in table:
        return table[n]
    keys = sorted(table)
    if n < keys[0]:
        return table[keys[0]]
    if n > keys[-1]:
        return table[keys[-1]]
    lo = max(k for k in keys if k < n)
    hi = min(k for k in keys if k > n)
    f = (n - lo) / (hi - lo)
    return table[lo] + f * (table[hi] - table[lo])


def batch_factor(per_gpu_batch: int) -> tuple[float, bool]:
    """Throughput multiplier for a per-GPU batch, and whether it is measured."""
    if per_gpu_batch in BATCH_SCALE:
        return BATCH_SCALE[per_gpu_batch], True
    keys = sorted(BATCH_SCALE)
    if per_gpu_batch < keys[0] or per_gpu_batch > keys[-1]:
        return BATCH_SCALE[min(keys, key=lambda k: abs(k - per_gpu_batch))], False
    return _interp({k: v for k, v in BATCH_SCALE.items()}, per_gpu_batch), False


# ── live cluster state, including the resources that actually bind ───────────

@dataclass
class Node:
    name: str
    feature: str
    state: str
    partitions: list[str]
    gpus_total: int = 0
    gpus_used: int = 0
    cpus_total: int = 0
    cpus_used: int = 0
    mem_total_mb: int = 0
    mem_used_mb: int = 0
    hca: str = ""

    @property
    def gpus_free(self) -> int:
        return max(0, self.gpus_total - self.gpus_used)

    @property
    def cpus_free(self) -> int:
        return max(0, self.cpus_total - self.cpus_used)

    @property
    def mem_free_gb(self) -> float:
        return max(0.0, (self.mem_total_mb - self.mem_used_mb) / 1024)

    @property
    def usable(self) -> bool:
        return not any(k in self.state.upper()
                       for k in ("DOWN", "DRAIN", "FAIL", "INVAL", "MAINT"))

    @property
    def load_frac(self) -> float:
        return self.gpus_used / self.gpus_total if self.gpus_total else 1.0


def _mem_mb(tres: str) -> int:
    m = re.search(r"mem=(\d+)([KMGT]?)", tres)
    if not m:
        return 0
    val, unit = int(m.group(1)), m.group(2)
    return {"": val // 1024, "K": val // 1024 // 1024, "M": val,
            "G": val * 1024, "T": val * 1024 * 1024}.get(unit, val)


def read_nodes(partition: str) -> list[Node]:
    out = subprocess.run(["scontrol", "show", "node", "-o"],
                         capture_output=True, text=True, check=True).stdout
    nodes = []
    for line in out.splitlines():
        if "NodeName=" not in line:
            continue

        def g(pat: str, default: str = "") -> str:
            m = re.search(pat, line)
            return m.group(1) if m else default

        name = g(r"NodeName=(\S+)")
        parts = [p for p in g(r"Partitions=(\S*)").split(",") if p]
        if partition not in parts:
            continue
        cfg, alloc = g(r"CfgTRES=(\S*)"), g(r"AllocTRES=(\S*)")

        def gpus(t: str) -> int:
            m = re.search(r"gres/gpu=(\d+)", t)
            return int(m.group(1)) if m else 0

        def cpus(t: str) -> int:
            m = re.search(r"cpu=(\d+)", t)
            return int(m.group(1)) if m else 0

        nodes.append(Node(
            name=name, feature=g(r"ActiveFeatures=(\S*)"), state=g(r" State=(\S+)"),
            partitions=parts, gpus_total=gpus(cfg), gpus_used=gpus(alloc),
            cpus_total=cpus(cfg), cpus_used=cpus(alloc),
            mem_total_mb=_mem_mb(cfg), mem_used_mb=_mem_mb(alloc),
        ))
    return nodes


def read_exclusions(paths: list[str]) -> dict[str, str]:
    """Nodes previously found broken or mislabelled, with the reason."""
    bad: dict[str, str] = {}
    for p in paths:
        f = Path(p)
        if not f.exists():
            continue
        for line in f.read_text().splitlines():
            parts = line.split(None, 1)
            if parts:
                bad.setdefault(parts[0], (parts[1] if len(parts) > 1 else "recorded").strip())
    return bad


# ── candidate placements ─────────────────────────────────────────────────────

@dataclass
class Plan:
    feature: str
    gpus: int
    nodes: list[str]
    gpus_per_node: int
    cpus_per_task: int
    mem_per_cpu_gb: int
    per_gpu_batch: int
    est_rank_samp_s: float
    starts_now: bool
    blockers: list[str] = field(default_factory=list)
    max_other_load: float = 0.0

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def est_total_samp_s(self) -> float:
        return self.est_rank_samp_s * self.gpus

    @property
    def global_batch(self) -> int:
        return self.per_gpu_batch * self.gpus


def estimate(feature: str, gpus: int, n_nodes: int, per_gpu_batch: int) -> Optional[float]:
    base = RANK_THROUGHPUT_B8.get(feature)
    if base is None:
        return None
    gpus_per_node = math.ceil(gpus / n_nodes)
    # Rank-count overhead is what it is regardless of placement; spreading adds
    # a small extra penalty on top.
    eff = _interp(INTRA_NODE_EFF, gpus) * _interp(INTER_NODE_EFF, n_nodes)
    bf, _ = batch_factor(per_gpu_batch)
    del gpus_per_node
    return base * eff * bf


def candidates(nodes: list[Node], want_gpus: int, per_gpu_batch: int,
               cpus_per_task: int, mem_per_cpu_gb: int, bad: dict[str, str],
               max_load: float) -> list[Plan]:
    by_feat: dict[str, list[Node]] = {}
    for n in nodes:
        if not n.usable or n.name in bad or not n.feature or n.feature == "(null)":
            continue
        by_feat.setdefault(n.feature, []).append(n)

    plans: list[Plan] = []
    for feat, group in by_feat.items():
        if feat in UNUSABLE or feat in UNMEASURED:
            continue
        # Prefer fewest nodes (measured: fewer nodes is faster at equal GPUs),
        # and within that the quietest nodes.
        group = sorted(group, key=lambda n: (-n.gpus_free, n.load_frac, n.name))
        for n_nodes in range(1, min(len(group), want_gpus) + 1):
            if want_gpus % n_nodes:
                continue  # keep ranks even across nodes
            per_node = want_gpus // n_nodes
            need_cpu = per_node * cpus_per_task
            need_mem = per_node * cpus_per_task * mem_per_cpu_gb
            picked, blockers = [], []
            for n in group:
                if len(picked) == n_nodes:
                    break
                why = []
                if n.gpus_free < per_node:
                    why.append(f"{n.name}: {n.gpus_free} free GPU < {per_node}")
                if n.cpus_free < need_cpu:
                    why.append(f"{n.name}: {n.cpus_free} free CPU < {need_cpu}")
                if n.mem_free_gb < need_mem:
                    why.append(f"{n.name}: {n.mem_free_gb:.0f}G free < {need_mem}G")
                if n.load_frac > max_load:
                    why.append(f"{n.name}: {n.load_frac:.0%} busy > {max_load:.0%}")
                if why:
                    blockers.extend(why)
                else:
                    picked.append(n)
            est = estimate(feat, want_gpus, n_nodes, per_gpu_batch)
            if est is None:
                continue
            starts = len(picked) == n_nodes
            chosen = picked if starts else [n.name for n in group[:n_nodes]]
            plans.append(Plan(
                feature=feat, gpus=want_gpus,
                nodes=[n.name if isinstance(n, Node) else n for n in chosen],
                gpus_per_node=per_node, cpus_per_task=cpus_per_task,
                mem_per_cpu_gb=mem_per_cpu_gb, per_gpu_batch=per_gpu_batch,
                est_rank_samp_s=est, starts_now=starts,
                blockers=blockers[:4],
                max_other_load=max((n.load_frac for n in picked), default=0.0),
            ))
    # Objective: throughput per wall-clock. A placement that can start now
    # beats a faster one that cannot, because queue time is dead time.
    return sorted(plans, key=lambda p: (not p.starts_now, -p.est_total_samp_s, p.n_nodes))


# ── reporting and config emission ────────────────────────────────────────────

def report(plans: list[Plan], nodes: list[Node], bad: dict[str, str],
           want_gpus: int, show_all: bool) -> Optional[Plan]:
    runnable = [p for p in plans if p.starts_now]
    print(f"{'GPU model':<15} {'GPUs':>4} {'nodes':>5} {'/node':>5} "
          f"{'samp/s/rank':>11} {'total':>8} {'start':>7}  placement")
    print("-" * 96)
    shown = plans if show_all else (runnable[:5] or plans[:5])
    for p in shown:
        print(f"{p.feature:<15} {p.gpus:>4} {p.n_nodes:>5} {p.gpus_per_node:>5} "
              f"{p.est_rank_samp_s:>11.1f} {p.est_total_samp_s:>8.1f} "
              f"{'now' if p.starts_now else 'waits':>7}  "
              f"{','.join(p.nodes) if p.starts_now else '(' + str(p.n_nodes) + ' nodes short)'}")
        if not p.starts_now and p.blockers:
            for b in p.blockers:
                print(f"{'':<49} blocked: {b}")

    skipped = sorted({n.feature for n in nodes
                      if n.usable and (n.feature in UNUSABLE or n.feature in UNMEASURED)})
    if skipped:
        print()
        for f in skipped:
            why = UNUSABLE.get(f) or "never benchmarked for this workload — refusing to guess"
            print(f"  skipped {f}: {why}")
    if bad:
        print(f"  excluded {len(bad)} recorded bad node(s): {', '.join(sorted(bad))}")

    if not runnable:
        print(f"\nNothing can start now at {want_gpus} GPU(s). Options: lower --gpus, "
              f"raise --max-node-load, or wait.")
        return None
    return runnable[0]


def describe(p: Plan, walltime: str, partition: str, bad: dict[str, str]) -> None:
    measured = p.per_gpu_batch in BATCH_SCALE
    print(f"\nRecommended: {p.gpus}x {p.feature} on {p.n_nodes} node(s) "
          f"-> ~{p.est_total_samp_s:.0f} samp/s "
          f"({p.est_rank_samp_s:.1f}/rank), global batch {p.global_batch}")
    if p.n_nodes > 1:
        one = estimate(p.feature, p.gpus, 1, p.per_gpu_batch)
        print(f"  spreading over {p.n_nodes} nodes costs ~"
              f"{(1 - p.est_rank_samp_s / one) * 100:.1f}% vs the same GPUs in one box; "
              f"it is what is free now")
    if p.max_other_load >= 0.5:
        print(f"  note: busiest chosen node is {p.max_other_load:.0%} held by other jobs — "
              f"a DDP step runs at its slowest rank, so expect ~5-10% under this estimate")
    if not measured:
        print(f"  note: per-GPU batch {p.per_gpu_batch} is interpolated; "
              f"measured points are {sorted(BATCH_SCALE)}")
    print("\nsbatch flags:")
    print(f"  --partition={partition} --nodes={p.n_nodes} "
          f"--ntasks-per-node={p.gpus_per_node} --gres=gpu:{p.gpus_per_node}")
    print(f"  --cpus-per-task={p.cpus_per_task} --mem-per-cpu={p.mem_per_cpu_gb}G "
          f"--time={walltime} --constraint={p.feature}")
    if bad:
        print(f"  --exclude={','.join(sorted(bad))}")


def emit(p: Plan, path: Path, walltime: str, partition: str, ckpt_dir: str,
         train_cmd: str, fallback: list[str]) -> None:
    # Dedupe: there is one candidate plan per (model, node count), so the same
    # GPU model appears several times in the runnable list.
    order: list[str] = [p.feature]
    for f in fallback:
        if f not in order:
            order.append(f)
    body = f"""# Generated by scripts/multinode/plan.py from live cluster state.
# Placement chosen for throughput per wall-clock: ~{p.est_total_samp_s:.0f} samp/s
# ({p.est_rank_samp_s:.1f}/rank) on {p.gpus}x {p.feature} over {p.n_nodes} node(s),
# which could start immediately when this was written.
#
# Re-run plan.py before submitting if the cluster has moved on; free capacity
# here changes within minutes and the node list below is not pinned.

partition: {partition}
nodes: {p.n_nodes}
gpus_per_node: {p.gpus_per_node}
cpus_per_task: {p.cpus_per_task}
mem_per_cpu: {p.mem_per_cpu_gb}G
walltime: "{walltime}"

# First entry is what plan.py picked; the rest are fallbacks submit.py rotates
# to if this pool fills up before the job lands.
constraints:
{chr(10).join(f'  - {f}' for f in order)}

pend_timeout_seconds: 600
max_requeues_per_constraint: 2
max_preflight_retries: 4

ckpt_dir: {ckpt_dir}
ckpt_interval_seconds: 300
signal_lead_seconds: 120

# 7.3-9.7 GB/s busbw is healthy here, ~4 GB/s is contended-but-fine, ~1.3 GB/s
# means NCCL fell back to sockets. 3.5 discriminates without false positives.
min_busbw: 3.5
min_free_gb: 20.0
allow_heterogeneous: false
# A DDP step runs at the speed of its slowest rank, so reject an allocation
# where a node is mostly someone else's work.
max_node_load: 0.75

train_cmd: >-
  {train_cmd}

log_dir: logs/multinode
runlog: logs/multinode/runs.jsonl
env:
  PYTHONUNBUFFERED: "1"
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    print(f"\nwrote {path}")
    print(f"  validate: python scripts/multinode/submit.py {path} --dry-run")
    if "checkpoint=" not in train_cmd and "--ckpt-dir" not in train_cmd:
        print(f"  WARNING: train_cmd has no resume path, so submit.py will refuse it. "
              f"Add checkpoint=<log_dir>/last.pth — a requeue without it restarts "
              f"from step 0.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpus", type=int, required=True, help="total GPUs wanted")
    ap.add_argument("--per-gpu-batch", type=int, default=8,
                    help="your per-GPU batch; placement respects it (default 8). "
                         "It sets the global batch, so it is yours to choose, not "
                         "the scheduler's")
    ap.add_argument("--partition", default="all")
    ap.add_argument("--walltime", default="1-00:00:00")
    ap.add_argument("--cpus-per-task", type=int, default=6)
    ap.add_argument("--mem-per-cpu-gb", type=int, default=4)
    ap.add_argument("--max-node-load", type=float, default=0.75,
                    help="skip nodes with more than this fraction of GPUs held by others")
    ap.add_argument("--show-all", action="store_true")
    ap.add_argument("--emit", help="write a submit.py config here")
    ap.add_argument("--ckpt-dir", default="/home/harshilb/3dfa_unified/train_logs/MultiNode")
    ap.add_argument("--train-cmd", default="<your train command>")
    ap.add_argument("--bad-nodes", action="append", default=[
        "train_logs/MultiNode/_bad_nodes.txt",
        "/grogu/user/harshilb/ckpt/mn-real/_bad_nodes.txt",
    ])
    args = ap.parse_args()

    want = parse_walltime(args.walltime)
    cap = partition_maxtime(args.partition)
    if want > cap:
        raise SystemExit(
            f"[plan] walltime {args.walltime} exceeds {args.partition} MaxTime "
            f"({cap // 3600}h). EnforcePartLimits=NO here, so Slurm would accept "
            f"the job and leave it PENDING forever.")

    nodes = read_nodes(args.partition)
    bad = read_exclusions(args.bad_nodes)
    plans = candidates(nodes, args.gpus, args.per_gpu_batch, args.cpus_per_task,
                       args.mem_per_cpu_gb, bad, args.max_node_load)
    if not plans:
        raise SystemExit(f"[plan] no GPU model in {args.partition} can run this "
                         f"workload at {args.gpus} GPU(s)")

    print(f"partition={args.partition}  want={args.gpus} GPU  "
          f"per-GPU batch={args.per_gpu_batch}  "
          f"{args.cpus_per_task} cpu + {args.cpus_per_task * args.mem_per_cpu_gb}G per rank\n")
    best = report(plans, nodes, bad, args.gpus, args.show_all)
    if best is None:
        return 1
    describe(best, args.walltime, args.partition, bad)
    if args.emit:
        fallback = [p.feature for p in plans if p.starts_now]
        emit(best, Path(args.emit), args.walltime, args.partition,
             args.ckpt_dir, args.train_cmd, fallback)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
