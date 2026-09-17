#!/usr/bin/env python3
"""Submit loop with constraint fallback.

The single highest-value piece for running multi-node in a scavenger queue.
A pinned constraint like `A6000` gives you a homogeneous allocation but a tiny
pool -- on grogu there are 7 A6000 nodes in `all` and at times only 2 have a
free GPU -- so a fixed constraint can pend indefinitely while an equally usable
pool sits idle. This cycles an ordered list instead.

Two things it refuses to do, both learned from how this cluster is configured:

* It validates the walltime against the partition's MaxTime *before* submitting.
  `EnforcePartLimits=NO` means Slurm accepts an over-limit job and leaves it
  PENDING with reason `PartitionTimeLimit` forever. A naive fallback loop would
  cycle through every constraint, blame capacity, and never report the real
  cause.
* It validates N against the usable pool size for each constraint and drops the
  ones that cannot possibly satisfy it, rather than discovering that by timeout.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

# Pend reasons that will never clear on their own. Waiting them out is the
# default behaviour of every fallback loop and it is always wrong.
FATAL_PEND = {
    "PartitionTimeLimit", "PartitionNodeLimit", "PartitionConfig",
    "BadConstraints", "InvalidQOS", "InvalidAccount", "QOSNotAllowed",
    "PartitionDown", "PartitionInactive", "JobHeldAdmin",
    "QOSMaxWallDurationPerJobLimit", "AssocMaxWallDurationPerJobLimit",
}
# Reasons that mean "keep waiting, capacity will come".
TRANSIENT_PEND = {
    "Resources", "Priority", "None", "", "Nodes required for job are DOWN,"
    " DRAINED or reserved for jobs in higher priority partitions",
    "BeginTime", "Dependency", "JobArrayTaskLimit",
}


def sh(cmd: list[str], check: bool = True) -> str:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise SystemExit(f"[submit] command failed: {shlex.join(cmd)}\n{r.stderr.strip()}")
    return r.stdout


# ── cluster introspection ─────────────────────────────────────────────────────

def parse_walltime(s: str) -> int:
    """Slurm walltime -> seconds. Accepts D-HH:MM:SS, HH:MM:SS, MM:SS, MM."""
    s = s.strip()
    if s.upper() in ("UNLIMITED", "INFINITE"):
        return sys.maxsize
    days = 0
    if "-" in s:
        d, s = s.split("-", 1)
        days = int(d)
    parts = [int(p) for p in s.split(":")] if ":" in s else [int(s)]
    if len(parts) == 3:
        h, m, sec = parts
    elif len(parts) == 2:
        h, m, sec = 0, parts[0], parts[1]
    else:
        h, m, sec = 0, parts[0], 0
    return days * 86400 + h * 3600 + m * 60 + sec


def partition_maxtime(part: str) -> int:
    out = sh(["scontrol", "show", "partition", part])
    m = re.search(r"MaxTime=(\S+)", out)
    if not m:
        raise SystemExit(f"[submit] could not read MaxTime for partition {part}")
    return parse_walltime(m.group(1))


@dataclass
class NodeInfo:
    name: str
    feature: str
    partitions: list[str]
    gpus_total: int
    gpus_used: int
    state: str

    @property
    def usable(self) -> bool:
        return not any(k in self.state.upper() for k in ("DOWN", "DRAIN", "FAIL", "INVAL"))

    @property
    def gpus_free(self) -> int:
        return max(0, self.gpus_total - self.gpus_used)


def node_table() -> list[NodeInfo]:
    out = sh(["scontrol", "show", "node", "-o"])
    nodes: list[NodeInfo] = []
    for line in out.splitlines():
        if "NodeName=" not in line:
            continue
        def g(pat: str, default: str = "") -> str:
            m = re.search(pat, line)
            return m.group(1) if m else default
        name = g(r"NodeName=(\S+)")
        feat = g(r"ActiveFeatures=(\S*)")
        parts = [p for p in g(r"Partitions=(\S*)").split(",") if p]
        cfg = g(r"CfgTRES=(\S*)")
        alloc = g(r"AllocTRES=(\S*)")
        def gpus(tres: str) -> int:
            m = re.search(r"gres/gpu=(\d+)", tres)
            return int(m.group(1)) if m else 0
        if not parts or feat in ("", "(null)"):
            continue
        nodes.append(NodeInfo(name, feat, parts, gpus(cfg), gpus(alloc), g(r" State=(\S+)")))
    return nodes


def pool_for(nodes: list[NodeInfo], partition: str, constraint: Optional[str]) -> list[NodeInfo]:
    return [n for n in nodes
            if partition in n.partitions
            and n.usable
            and (constraint is None or n.feature == constraint)]


# ── job state ─────────────────────────────────────────────────────────────────

@dataclass
class JobState:
    job_id: str
    state: str = "UNKNOWN"
    reason: str = ""
    nodelist: str = ""
    restarts: int = 0

    @property
    def pending(self) -> bool:
        return self.state in ("PENDING", "CONFIGURING")

    @property
    def running(self) -> bool:
        return self.state == "RUNNING"

    @property
    def gone(self) -> bool:
        return self.state in ("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT",
                              "NODE_FAIL", "OUT_OF_MEMORY", "UNKNOWN_GONE")


def job_state(job_id: str) -> JobState:
    out = sh(["squeue", "-j", job_id, "-h", "-o", "%T|%r|%N|%k"], check=False)
    line = out.strip().splitlines()
    if not line:
        # Left the queue; ask accounting what happened to it.
        # NOTE: no "Restarts" field -- it does not exist in Slurm 20.11 and asking
        # for it makes sacct fail, which previously turned every finished job into
        # UNKNOWN_GONE and made the supervisor exit the moment a job ended.
        acct = sh(["sacct", "-j", job_id, "-X", "-n", "-P",
                   "-o", "State,ExitCode"], check=False)
        first = [l for l in acct.strip().splitlines() if l.strip()]
        if not first:
            return JobState(job_id, "UNKNOWN_GONE")
        fields = first[0].split("|")
        state = fields[0].split()[0] if fields[0] else "UNKNOWN_GONE"
        exit_code = fields[1] if len(fields) > 1 else ""
        return JobState(job_id, state, reason=exit_code)
    t, reason, nodes = (line[0].split("|") + ["", "", ""])[:3]
    return JobState(job_id, t, reason, nodes)


def classify(reason: str) -> str:
    r = reason.strip()
    if r in FATAL_PEND:
        return "fatal"
    if r in TRANSIENT_PEND or r.startswith("Nodes required"):
        return "transient"
    if r.startswith("ReqNodeNotAvail"):
        # Names specific unavailable nodes: usually a drained node in a narrow
        # constraint pool. Transient in principle, fatal in practice for hours.
        return "slow"
    return "transient"


# ── config ────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    partition: str
    nodes: int
    walltime: str
    train_cmd: str
    ckpt_dir: str
    constraints: list[str] = field(default_factory=list)
    gpus_per_node: int = 1
    cpus_per_task: int = 8
    mem_per_cpu: str = "8G"
    ckpt_interval_seconds: int = 600
    pend_timeout_seconds: int = 900
    max_requeues_per_constraint: int = 3
    max_preflight_retries: int = 4
    min_busbw: float = 4.0
    max_node_load: float = 1.0
    min_free_gb: float = 50.0
    allow_heterogeneous: bool = False
    log_dir: str = "logs/multinode"
    runlog: str = "logs/multinode/runs.jsonl"
    signal_lead_seconds: int = 120
    env: dict = field(default_factory=dict)

    @staticmethod
    def load(path: str | Path) -> "Config":
        raw = yaml.safe_load(Path(path).read_text()) or {}
        known = {f for f in Config.__dataclass_fields__}  # type: ignore[attr-defined]
        unknown = set(raw) - known
        if unknown:
            raise SystemExit(f"[submit] unknown config keys: {sorted(unknown)}")
        missing = [k for k in ("partition", "nodes", "walltime", "train_cmd", "ckpt_dir")
                   if k not in raw]
        if missing:
            raise SystemExit(f"[submit] config is missing required keys: {missing}")
        return Config(**raw)


# ── validation ────────────────────────────────────────────────────────────────

# Any of these in train_cmd means the trainer has been told where to resume from.
_RESUME_HINTS = ("checkpoint=", "--checkpoint", "--ckpt-dir", "--resume",
                 "RESILIENCE_CKPT_DIR")


def validate(cfg: Config, nodes: list[NodeInfo]) -> list[Optional[str]]:
    """Fail fast on anything that would pend forever. Returns the usable
    constraint list, in the configured order, with impossible ones dropped."""
    # A requeue that cannot resume is worse than no requeue: the job saves a
    # checkpoint on preemption, comes back, and silently restarts from step 0 --
    # losing everything while looking like it recovered. This bit us for real:
    # `checkpoint: null` is 3DFA's default, so a train_cmd that omits it trains
    # forever without ever making progress past the first preemption.
    if not any(h in cfg.train_cmd for h in _RESUME_HINTS):
        raise SystemExit(
            "[submit] FATAL: train_cmd contains no resume path — none of "
            f"{list(_RESUME_HINTS)} appear in it.\n"
            "          The launcher requeues preempted jobs, so without this the "
            "run restarts from step 0 every\n"
            "          time it is preempted and never finishes. For 3DFA add:\n"
            "            checkpoint=<base_log_dir>/<exp_log_dir>/<run_log_dir>/last.pth\n"
            "          It is safe on a first run: base.py starts from scratch when "
            "the file does not exist yet."
        )
    want = parse_walltime(cfg.walltime)
    cap = partition_maxtime(cfg.partition)
    if want > cap:
        raise SystemExit(
            f"[submit] FATAL: walltime {cfg.walltime} exceeds partition "
            f"{cfg.partition} MaxTime ({cap // 3600}h{cap % 3600 // 60:02d}m). "
            f"EnforcePartLimits=NO on this cluster, so Slurm would accept this job "
            f"and leave it PENDING with reason=PartitionTimeLimit indefinitely. "
            f"Lower walltime or pick a partition with a longer limit."
        )

    if not cfg.constraints:
        unpinned = pool_for(nodes, cfg.partition, None)
        if len(unpinned) < cfg.nodes:
            raise SystemExit(f"[submit] FATAL: partition {cfg.partition} has "
                             f"{len(unpinned)} usable nodes, need {cfg.nodes}")
        print(f"[submit] no constraints configured — unpinned. "
              f"WARNING: {cfg.partition} spans "
              f"{sorted({n.feature for n in unpinned})}; a mixed allocation runs at "
              f"the speed of its slowest GPU.")
        return [None]

    excluded = set(bad_nodes(cfg))
    if excluded:
        print(f"[submit] excluding nodes preflight found mislabelled: {sorted(excluded)}")

    usable: list[Optional[str]] = []
    ready: list[Optional[str]] = []
    for c in cfg.constraints:
        pool = [n for n in pool_for(nodes, cfg.partition, c) if n.name not in excluded]
        free = [n for n in pool if n.gpus_free >= cfg.gpus_per_node]
        if len(pool) < cfg.nodes:
            print(f"[submit] dropping constraint {c}: pool has {len(pool)} usable node(s) "
                  f"in {cfg.partition}, need {cfg.nodes} — it can never satisfy this job")
            continue
        usable.append(c)
        if len(free) >= cfg.nodes:
            ready.append(c)
        print(f"[submit] constraint {c}: {len(pool)} usable node(s), "
              f"{len(free)} with >={cfg.gpus_per_node} free GPU now"
              f"{'  <- can start now' if len(free) >= cfg.nodes else ''}")

    # Lead with a pool that can start immediately. Submitting to the preferred
    # pool when it is full means waiting out the whole pend timeout before
    # rotating, while an equally usable pool sits idle -- pure queue time for
    # nothing. Configured order is preserved within each group, so preference
    # still decides between two pools that are both free.
    if ready and usable[0] not in ready:
        reordered = ready + [c for c in usable if c not in ready]
        print(f"[submit] reordering to start on available capacity: "
              f"{[c or 'any' for c in reordered]} "
              f"(configured: {[c or 'any' for c in usable]})")
        usable = reordered
    if not usable:
        raise SystemExit(
            f"[submit] FATAL: no configured constraint has {cfg.nodes} usable nodes in "
            f"{cfg.partition}. Pools: "
            + ", ".join(f"{c}={len(pool_for(nodes, cfg.partition, c))}" for c in cfg.constraints)
        )
    return usable


# ── submission ────────────────────────────────────────────────────────────────

def job_env(cfg: Config, constraint: Optional[str]) -> dict[str, str]:
    """Everything job.sbatch needs, so the batch script holds no settings."""
    return {
        "RESILIENCE_CKPT_DIR": cfg.ckpt_dir,
        "RESILIENCE_TRAIN_CMD": cfg.train_cmd,
        "RESILIENCE_CONSTRAINT": constraint or "",
        "RESILIENCE_MIN_BUSBW": str(cfg.min_busbw),
        "RESILIENCE_MAX_NODE_LOAD": str(cfg.max_node_load),
        "RESILIENCE_MIN_FREE_GB": str(cfg.min_free_gb),
        "RESILIENCE_ALLOW_HET": "1" if cfg.allow_heterogeneous else "0",
        "RESILIENCE_RUNLOG": str(Path(cfg.runlog).resolve()),
        "RESILIENCE_CKPT_INTERVAL": str(cfg.ckpt_interval_seconds),
        **{str(k): str(v) for k, v in cfg.env.items()},
    }


def bad_nodes(cfg: Config) -> list[str]:
    """Nodes preflight found to be mislabelled, accumulated across runs.

    Slurm node features are hand-maintained and do go stale: grogu-4-13
    advertises ActiveFeatures=A6000 while physically holding an RTX 3080 Ti. So
    --constraint does not guarantee a homogeneous allocation, and without this
    the launcher keeps landing on the same bad node and failing preflight
    forever.
    """
    f = Path(cfg.ckpt_dir) / "_bad_nodes.txt"
    if not f.exists():
        return []
    seen = []
    for line in f.read_text().splitlines():
        host = line.split()[0] if line.split() else ""
        if host and host not in seen:
            seen.append(host)
    return seen


def build_sbatch(cfg: Config, constraint: Optional[str], script: Path) -> list[str]:
    log_dir = Path(cfg.log_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    tag = constraint or "any"
    cmd = [
        "sbatch",
        "--export=ALL," + ",".join(f"{k}={v}" for k, v in job_env(cfg, constraint).items()),
        f"--partition={cfg.partition}",
        f"--nodes={cfg.nodes}",
        "--ntasks-per-node=1",
        f"--gres=gpu:{cfg.gpus_per_node}",
        f"--cpus-per-task={cfg.cpus_per_task}",
        f"--mem-per-cpu={cfg.mem_per_cpu}",
        f"--time={cfg.walltime}",
        f"--job-name=mn-{tag}",
        f"--output={log_dir}/%x-%j.out",
        f"--error={log_dir}/%x-%j.err",
        "--requeue",
        f"--signal=B:USR1@{cfg.signal_lead_seconds}",
        # Stop the login node's 64 KB memlock limit being copied into the job.
        # The batch script still runs `ulimit -l unlimited` because the node's
        # own default is no better.
        "--propagate=NONE",
    ]
    if constraint:
        cmd.append(f"--constraint={constraint}")
    bad = bad_nodes(cfg)
    if bad:
        cmd.append(f"--exclude={','.join(bad)}")
    cmd.append(str(script))
    return cmd


def submit(cfg: Config, constraint: Optional[str], script: Path) -> str:
    cmd = build_sbatch(cfg, constraint, script)
    print(f"[submit] {shlex.join(cmd)}")
    out = sh(cmd)
    m = re.search(r"(\d+)", out)
    if not m:
        raise SystemExit(f"[submit] could not parse a job id from: {out!r}")
    return m.group(1)


def record(cfg: Config, rec: dict) -> None:
    p = Path(cfg.runlog)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps({"ts": time.time(), **rec}) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--dry-run", action="store_true",
                    help="validate and print the plan without submitting")
    ap.add_argument("--poll", type=int, default=30, help="seconds between squeue polls")
    ap.add_argument("--detach", action="store_true",
                    help="submit the first attempt and exit instead of supervising")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    script = Path(__file__).with_name("job.sbatch")
    if not script.exists():
        raise SystemExit(f"[submit] missing {script}")

    nodes = node_table()
    order = validate(cfg, nodes)
    print(f"[submit] plan: partition={cfg.partition} N={cfg.nodes} "
          f"walltime={cfg.walltime} constraints={[c or 'any' for c in order]}")

    if args.dry_run:
        print("[submit] dry run — nothing submitted")
        for c in order:
            print("  " + shlex.join(build_sbatch(cfg, c, script)))
        return 0

    idx = 0
    requeues = 0
    attempts = 0
    while True:
        constraint = order[idx % len(order)]
        job_id = submit(cfg, constraint, script)
        record(cfg, {"kind": "submit", "job_id": job_id, "constraint": constraint,
                     "nodes": cfg.nodes, "partition": cfg.partition})
        print(f"[submit] job {job_id} submitted with constraint={constraint or 'any'}")
        if args.detach:
            print(f"[submit] --detach: not supervising. Watch with: squeue -j {job_id}")
            return 0

        pend_since: Optional[float] = time.time()
        while True:
            time.sleep(args.poll)
            st = job_state(job_id)

            if st.running:
                if pend_since is not None:
                    print(f"[submit] job {job_id} RUNNING on {st.nodelist} "
                          f"after {time.time() - pend_since:.0f}s pending")
                    record(cfg, {"kind": "running", "job_id": job_id,
                                 "constraint": constraint, "nodelist": st.nodelist})
                    pend_since = None
                continue

            if st.pending:
                if pend_since is None:
                    # Was running, now pending again: Slurm requeued it, which
                    # only happens on preemption here. Note that a requeue keeps
                    # the ORIGINAL --constraint, so it returns to the same pool.
                    requeues += 1
                    pend_since = time.time()
                    print(f"[submit] job {job_id} was requeued (preempted) — "
                          f"requeue {requeues}/{cfg.max_requeues_per_constraint} "
                          f"on constraint={constraint or 'any'}")
                    record(cfg, {"kind": "preempted", "job_id": job_id,
                                 "constraint": constraint, "requeue": requeues})
                    if requeues >= cfg.max_requeues_per_constraint and len(order) > 1:
                        print(f"[submit] this pool keeps evicting us; cancelling and "
                              f"moving to the next constraint")
                        sh(["scancel", job_id], check=False)
                        idx += 1
                        requeues = 0
                        break
                    continue

                kind = classify(st.reason)
                waited = time.time() - pend_since
                if kind == "fatal":
                    sh(["scancel", job_id], check=False)
                    raise SystemExit(
                        f"[submit] FATAL: job {job_id} is PENDING with reason "
                        f"'{st.reason}', which never clears. Cancelled. This is a "
                        f"configuration error, not a capacity problem — cycling "
                        f"constraints would not help."
                    )
                if waited > cfg.pend_timeout_seconds and len(order) > 1:
                    nxt = order[(idx + 1) % len(order)]
                    print(f"[submit] job {job_id} pending {waited:.0f}s "
                          f"(reason={st.reason}) > {cfg.pend_timeout_seconds}s; "
                          f"cancelling and retrying with constraint={nxt or 'any'}")
                    record(cfg, {"kind": "fallback", "job_id": job_id,
                                 "from": constraint, "to": nxt, "reason": st.reason,
                                 "waited_s": round(waited)})
                    sh(["scancel", job_id], check=False)
                    idx += 1
                    break
                continue

            if st.gone:
                # 78 is job.sbatch's EX_CONFIG: preflight refused the allocation.
                # That is a property of the nodes we were given, not of the run,
                # so retry rather than give up -- preflight has just recorded any
                # mislabelled node, and build_sbatch will now exclude it.
                if st.reason.startswith("78:"):
                    attempts += 1
                    bad = bad_nodes(cfg)
                    print(f"[submit] job {job_id} failed preflight (exit 78) — "
                          f"the allocation was unusable, not the run. "
                          f"Excluding {bad or 'nothing new'}; retrying "
                          f"({attempts}/{cfg.max_preflight_retries})")
                    record(cfg, {"kind": "preflight_reject", "job_id": job_id,
                                 "constraint": constraint, "excluding": bad})
                    if attempts >= cfg.max_preflight_retries:
                        raise SystemExit(
                            f"[submit] giving up after {attempts} preflight "
                            f"rejections. Read the job output: the allocations "
                            f"keep failing a check that excluding nodes did not fix."
                        )
                    idx += 1  # try the next pool too
                    break
                print(f"[submit] job {job_id} finished with state {st.state} "
                      f"(exit {st.reason or 'n/a'})")
                record(cfg, {"kind": "final", "job_id": job_id, "exit": st.reason,
                             "constraint": constraint, "state": st.state})
                return 0 if st.state == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
