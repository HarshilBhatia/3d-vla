"""In-allocation preflight. Fails closed.

Runs as an srun step across every rank *before* train.py starts. You do not
know what you actually got until you are inside the allocation: which nodes,
which GPU model, whether memlock was propagated, whether NCCL will pick the
InfiniBand verbs path or quietly fall back to sockets.

Every check either passes or exits nonzero. There is no degraded mode: a run
that silently trains on IPoIB at 18% of fabric bandwidth, or on a mixed
A5000/2080Ti pool where every step waits on the slowest card, is worse than a
run that refuses to start.

Usage (inside a job):
    srun --ntasks-per-node=1 python scripts/multinode/preflight.py --min-busbw 4.0
"""

from __future__ import annotations

import argparse
import json
import os
import re
import resource
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

FAIL: list[str] = []
WARN: list[str] = []

# Every rank writes to the same job stdout, so without a host tag the lines
# interleave into something unreadable at N>2.
_HOST = socket.gethostname().split(".")[0]


def _say(level: str, msg: str) -> None:
    print(f"[preflight {_HOST}] {level} {msg}", flush=True)


def _fail(msg: str) -> None:
    FAIL.append(msg)
    _say("FAIL", msg)


def _warn(msg: str) -> None:
    WARN.append(msg)
    _say("WARN", msg)


def _ok(msg: str) -> None:
    _say("ok  ", msg)


# ── individual checks ─────────────────────────────────────────────────────────

def check_memlock() -> None:
    """RDMA registers pinned memory; a 64 KB soft limit makes ibv_create_cq fail.

    On grogu the login node's hard limit is 64 KB and PropagateResourceLimits=ALL
    copies it into the job, while the compute nodes' own hard limit is unlimited.
    So this is always fixable in-job with `ulimit -l unlimited`, and the failure
    mode without it is a hard NCCL crash, not slow training.
    """
    soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    if soft != resource.RLIM_INFINITY:
        _fail(f"memlock soft limit is {soft} bytes, not unlimited. Add "
              f"'ulimit -l unlimited' to the batch script before srun "
              f"(hard limit here is {'unlimited' if hard == resource.RLIM_INFINITY else hard}).")
    else:
        _ok("memlock unlimited")


def _record_bad(outfile: str | None, reason: str) -> None:
    """Append this host to the launcher's exclude list."""
    if not outfile:
        return
    try:
        Path(outfile).parent.mkdir(parents=True, exist_ok=True)
        with open(outfile, "a") as f:
            f.write(f"{_HOST} {reason}\n")
        _say("--  ", f"recorded {_HOST} in {outfile}")
    except OSError as e:
        _warn(f"could not write {outfile}: {e}")


def check_hca(expect: str = "mlx5_0", outfile: str | None = None) -> None:
    """The expected HCA must exist and be an Active InfiniBand port.

    Row-2 nodes also expose Intel `irdma0/irdma1` (RoCE on the ethernet NIC).
    Unpinned, NCCL can enumerate one of those instead of the 100 Gb fabric and
    the same script gets different bandwidth depending on which nodes it landed
    on -- so we assert the specific device, not merely "some RDMA device".
    """
    base = Path("/sys/class/infiniband")
    if not base.is_dir():
        _fail("no /sys/class/infiniband — this node has no RDMA device at all")
        return
    present = sorted(p.name for p in base.iterdir())
    if expect not in present:
        # grogu-2-35 carries mlx4_0 (an older ConnectX generation) where every
        # other node has mlx5_0. NCCL_IB_HCA is one value for the whole job, so
        # no single setting works across a mixed-fabric allocation: on the odd
        # node NCCL cannot find the named device and dies with
        # "socketFinalizeAccept: wrong type 3 != 4". Such a node simply cannot
        # participate, so record it for exclusion rather than just failing.
        _fail(f"HCA {expect} absent; found {present or 'none'}. NCCL_IB_HCA is "
              f"job-wide, so this node cannot join a job pinned to {expect}.")
        _record_bad(outfile, f"HCA-MISMATCH has {present} not {expect}")
        return
    states = list((base / expect / "ports").glob("*/state"))
    link = list((base / expect / "ports").glob("*/link_layer"))
    state_txt = " ".join(p.read_text().strip() for p in states)
    link_txt = " ".join(p.read_text().strip() for p in link)
    if "ACTIVE" not in state_txt.upper():
        _fail(f"{expect} port not ACTIVE (state={state_txt!r})")
    elif "INFINIBAND" not in link_txt.upper():
        _fail(f"{expect} link_layer is {link_txt!r}, expected InfiniBand")
    else:
        _ok(f"{expect} ACTIVE InfiniBand (other devices present: "
            f"{[d for d in present if d != expect] or 'none'})")


def gpu_name() -> str:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30, check=True)
        names = sorted({l.strip() for l in out.stdout.splitlines() if l.strip()})
        return "/".join(names) if names else "none"
    except Exception as e:  # noqa: BLE001
        return f"unknown({type(e).__name__})"


def check_gpu_health() -> None:
    """Every allocated GPU must exist and accept work.

    Slurm's view of a node can be wrong in both directions, and partially so.
    On grogu-3-20 exactly one card of eight (PCI 0000:1B:00.0) is faulty:
    nvidia-smi prints "Unable to determine the device handle for GPU0 ...
    Unknown Error" and omits it, while /proc/driver/nvidia still lists all
    eight and Slurm keeps the node in State=MIXED with Gres=gpu:8. So whether a
    job gets a dead GPU is luck of the draw.

    Checking only that *a* name came back is not enough -- with several GPUs
    allocated nvidia-smi still returns the healthy ones and the dead one simply
    goes missing. Hence the count comparison and the per-device probe.
    """
    expected = 0
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd:
        expected = len([x for x in cvd.split(",") if x != ""])
    elif os.environ.get("SLURM_GPUS_ON_NODE"):
        expected = int(os.environ["SLURM_GPUS_ON_NODE"])

    name = gpu_name()
    if "No devices" in name or name in ("", "none"):
        _fail(f"nvidia-smi reports no usable GPU on this node (got {name!r}) "
              f"though Slurm allocated {expected or 'some'} — the device is "
              f"broken; exclude this node")
        return

    try:
        import torch
    except ImportError:
        _warn("torch unavailable; skipped the GPU allocation probe")
        return

    count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if expected and count != expected:
        _fail(f"Slurm allocated {expected} GPU(s) but torch sees {count} "
              f"({name}) — a card in this allocation is faulty (nvidia-smi "
              f"silently omits a GPU whose handle it cannot read); exclude this node")
        return
    if count < 1:
        _fail(f"torch sees no CUDA device though nvidia-smi says {name}")
        return

    # A handle can exist while the GPU refuses work, so touch every one of them.
    for i in range(count):
        try:
            t = torch.ones(1024, device=f"cuda:{i}")
            _ = (t * 2).sum().item()
            del t
        except Exception as e:  # noqa: BLE001
            _fail(f"cuda:{i} ({name}) rejected a trivial allocation: "
                  f"{type(e).__name__}: {str(e)[:80]}")
            return
    torch.cuda.empty_cache()
    _ok(f"GPU healthy: {count}x {name}")


def check_neighbours(max_load: float = 1.0) -> None:
    """Reject (or warn about) a node most of whose GPUs belong to other jobs.

    Measured, and it matters more than the fabric does. The same config on a
    quiet A5000 ran at 61.8 samp/s; on one with 5 of 8 GPUs busy, 55.4 -- a 10%
    loss. Exposed allreduce cost over the same span was +1.2 ms at N=2 and
    +6.6 ms at N=4, i.e. 1-4% of a step. So neighbour contention costs several
    times more than going multi-node.

    It compounds, because a DDP step runs at the speed of its slowest rank: the
    N=4 run that happened to include the busy node fell to 53.6 samp/s/rank,
    *below* the single-GPU run on that same node. One noisy neighbour taxes
    every rank in the job.

    Load is transient, so a failure here is not recorded against the node --
    the launcher simply retries and gets a different allocation.
    """
    host = socket.gethostname().split(".")[0]
    try:
        out = subprocess.run(["scontrol", "show", "node", host],
                             capture_output=True, text=True, timeout=30, check=True).stdout
    except Exception as e:  # noqa: BLE001
        _warn(f"could not query node allocation: {type(e).__name__}")
        return

    def grab(pat: str) -> int:
        m = re.search(pat, out)
        return int(m.group(1)) if m else 0

    total = grab(r"CfgTRES=.*?gres/gpu=(\d+)")
    used = grab(r"AllocTRES=.*?gres/gpu=(\d+)")
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES") or ""
    mine = len([x for x in cvd.split(",") if x]) if cvd else 0
    others = max(0, used - mine)
    if not total:
        _warn(f"could not determine GPU count on {host}")
        return

    frac = others / total
    detail = f"{others}/{total} GPUs on {host} held by other jobs ({frac:.0%})"
    if frac > max_load:
        _fail(f"{detail}, over the {max_load:.0%} limit — a DDP step runs at the "
              f"speed of its slowest rank, so this node would tax every rank. "
              f"Load is transient; retry for a different allocation.")
    elif frac >= 0.5:
        _warn(f"{detail} — expect ~10% slower steps than on a quiet node")
    else:
        _ok(f"node load: {detail}")


def check_ckpt_dir(path: str | None, min_free_gb: float) -> None:
    """Checkpoint dir must be shared and have room.

    /tmp on grogu is node-local (verified: 63 GB local disk, not the NAS), so a
    checkpoint written there is simply gone when the job requeues elsewhere.
    """
    if not path:
        _warn("no --ckpt-dir given; skipping checkpoint filesystem check")
        return
    p = Path(path)
    for bad in ("/tmp", "/var/tmp", "/dev/shm"):
        if str(p) == bad or str(p).startswith(bad + "/"):
            _fail(f"ckpt_dir {p} is node-local on grogu; a requeue would lose it. "
                  f"Use /home/... or /grogu/user/...")
            return
    try:
        p.mkdir(parents=True, exist_ok=True)
        probe = p / f".preflight_{os.environ.get('SLURM_JOB_ID', 'x')}_{socket.gethostname()}"
        probe.write_text("ok")
        probe.unlink()
    except OSError as e:
        _fail(f"ckpt_dir {p} is not writable from this node: {e}")
        return
    free_gb = shutil.disk_usage(p).free / 1e9
    if free_gb < min_free_gb:
        _fail(f"ckpt_dir {p} has {free_gb:.0f} GB free, below the {min_free_gb:.0f} GB floor")
    else:
        _ok(f"ckpt_dir {p} writable, {free_gb:.0f} GB free")


def check_nccl(min_busbw: float, expect_net: str = "IB") -> float | None:
    """Run one real allreduce across every rank and assert transport + bandwidth.

    Done in-process rather than by spawning a probe per task: a subprocess per
    rank means each task builds its own process group, the groups contend for
    the same fabric, and the measured bandwidth comes out roughly halved while
    only one rank ever sees the result.

    Parsing NCCL's own debug log is the only way to learn which path was
    actually taken. NCCL_SOCKET_IFNAME scopes just the bootstrap socket, so a
    job can report ib0 everywhere and still move gradients over TCP.
    """
    world = int(os.environ.get("SLURM_NTASKS") or os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("SLURM_PROCID") or os.environ.get("RANK", 0))
    if world < 2:
        _warn("world size < 2; skipping NCCL transport + bandwidth check")
        return None

    share = os.environ.get("PREFLIGHT_SHARE", ".")
    dbg = Path(share) / f"nccl.rank{rank}.log"
    dbg.parent.mkdir(parents=True, exist_ok=True)
    # Must be set before the first NCCL call, which happens inside
    # init_process_group -- not at import time.
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_DEBUG_SUBSYS"] = "INIT,NET"
    os.environ["NCCL_DEBUG_FILE"] = str(dbg)

    import torch
    import torch.distributed as dist

    if not torch.cuda.is_available():
        _fail("torch reports no CUDA device")
        return None

    # Offset the port so a lingering TCPStore from the training step (or a
    # previous attempt on a requeued job) cannot be mistaken for ours.
    port = int(os.environ.get("MASTER_PORT", 29500)) + 1
    try:
        torch.cuda.set_device(0)  # cgroup-confined: the job's GPU is device 0
        dist.init_process_group(
            "nccl", rank=rank, world_size=world,
            init_method=f"tcp://{os.environ['MASTER_ADDR']}:{port}",
            timeout=__import__("datetime").timedelta(seconds=180),
        )
    except Exception as e:  # noqa: BLE001
        blob = dbg.read_text() if dbg.exists() else ""
        if "ibv_create_cq" in blob or "Cannot allocate memory" in blob:
            _fail("NCCL init crashed in ibv_create_cq (Cannot allocate memory) — "
                  "this is the memlock limit, add 'ulimit -l unlimited'")
        else:
            _fail(f"NCCL init failed: {type(e).__name__}: {e}")
        return None

    try:
        t = torch.ones(64 << 20, device="cuda")  # 256 MiB fp32
        for _ in range(3):
            dist.all_reduce(t)
        torch.cuda.synchronize()
        iters = 10
        t0 = time.time()
        for _ in range(iters):
            dist.all_reduce(t)
        torch.cuda.synchronize()
        el = (time.time() - t0) / iters
    except Exception as e:  # noqa: BLE001
        _fail(f"NCCL allreduce failed: {type(e).__name__}: {e}")
        dist.destroy_process_group()
        return None

    gb = 0.25 * 1.073741824  # 256 MiB -> GB
    algbw = gb / el
    busbw = algbw * 2 * (world - 1) / world

    blob = dbg.read_text() if dbg.exists() else ""
    via = set(re.findall(r"via (NET/\w+)/\d+", blob))
    if not via:
        _warn(f"could not parse a transport from {dbg}")
    elif any(v != f"NET/{expect_net}" for v in via):
        _fail(f"NCCL selected {sorted(via)}, expected NET/{expect_net} only. "
              f"NET/Socket means TCP-over-IPoIB (~18% of RDMA here) with no RDMA. "
              f"Check NCCL_IB_HCA and NCCL_IB_DISABLE=0.")
    else:
        _ok(f"NCCL transport {sorted(via)}")

    if busbw < min_busbw:
        _fail(f"busbw {busbw:.2f} GB/s is below the {min_busbw:.2f} GB/s floor "
              f"(algbw {algbw:.2f}, {el * 1e3:.1f} ms/allreduce, world {world}). "
              f"Expect ~7.3 GB/s for 2 ranks on this fabric; ~1.3 GB/s means sockets.")
    else:
        _ok(f"busbw {busbw:.2f} GB/s (floor {min_busbw:.2f}, {el * 1e3:.1f} ms/allreduce)")

    dist.destroy_process_group()
    return busbw


def check_homogeneous(strict: bool, outfile: str | None = None) -> str:
    """Every rank must hold the same GPU model.

    `--constraint` narrows the pool but does not guarantee it: node features are
    hand-maintained labels and `all` spans H200 down to 2080Ti. In a mixed
    allocation every allreduce syncs to the slowest card, so a job that "works"
    can be quietly running at the speed of its worst node.
    """
    mine = gpu_name()
    host = socket.gethostname()
    names = {host: mine}

    # Ranks are separate processes; exchange through a file in the shared
    # checkpoint-adjacent dir that the launcher gives us.
    share = os.environ.get("PREFLIGHT_SHARE")
    if share:
        d = Path(share)
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{host}.json").write_text(json.dumps({"host": host, "gpu": mine}))
        expect = int(os.environ.get("SLURM_JOB_NUM_NODES", 1))
        deadline = time.time() + 120
        while time.time() < deadline:
            found = sorted(d.glob("*.json"))
            if len(found) >= expect:
                break
            time.sleep(1)
        names = {}
        for f in sorted(d.glob("*.json")):
            try:
                rec = json.loads(f.read_text())
                names[rec["host"]] = rec["gpu"]
            except (OSError, ValueError):
                continue

    distinct = sorted(set(names.values()))
    if len(distinct) > 1:
        detail = ", ".join(f"{h}={g}" for h, g in sorted(names.items()))
        msg = (f"heterogeneous allocation: {distinct}. Every step would sync to the "
               f"slowest card. Nodes: {detail}")
        _fail(msg) if strict else _warn(msg)
        # Slurm node features are hand-maintained and can be wrong -- grogu-4-13
        # advertises ActiveFeatures=A6000 while physically holding an RTX 3080 Ti.
        # A constraint therefore cannot guarantee homogeneity, and re-submitting
        # with the same constraint will keep landing on the same bad node. Record
        # the odd ones out so the launcher can exclude them from here on.
        if outfile:
            counts: dict[str, int] = {}
            for g in names.values():
                counts[g] = counts.get(g, 0) + 1
            majority = max(counts, key=lambda g: counts[g])
            odd = sorted(h for h, g in names.items() if g != majority)
            try:
                Path(outfile).parent.mkdir(parents=True, exist_ok=True)
                with open(outfile, "a") as f:
                    for h in odd:
                        f.write(f"{h} {names[h]} (declared {os.environ.get('RESILIENCE_CONSTRAINT', '?')})\n")
                _say("--  ", f"recorded mislabelled node(s) {odd} in {outfile}")
            except OSError as e:
                _warn(f"could not write {outfile}: {e}")
    else:
        _ok(f"homogeneous GPUs: {distinct[0] if distinct else 'unknown'} "
            f"across {len(names)} node(s)")
    return distinct[0] if len(distinct) == 1 else "/".join(distinct)


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir")
    ap.add_argument("--min-busbw", type=float, default=4.0,
                    help="GB/s floor; ~7.3 is normal, ~1.3 means socket fallback")
    ap.add_argument("--min-free-gb", type=float, default=50.0)
    ap.add_argument("--hca", default=os.environ.get("NCCL_IB_HCA", "mlx5_0"))
    ap.add_argument("--allow-heterogeneous", action="store_true",
                    help="downgrade the GPU-identity check to a warning")
    ap.add_argument("--skip-nccl", action="store_true")
    ap.add_argument("--max-node-load", type=float, default=1.0,
                    help="fail if more than this fraction of a node's GPUs belong to "
                         "other jobs (default 1.0 = never fail, only warn)")
    ap.add_argument("--runlog", help="append a JSON record of what we found")
    ap.add_argument("--bad-nodes-out",
                    help="append nodes whose GPU disagrees with the majority, so the "
                         "launcher can exclude them (Slurm features can be stale)")
    args = ap.parse_args()

    _say("--  ", f"job={os.environ.get('SLURM_JOB_ID')} "
                 f"nodes={os.environ.get('SLURM_JOB_NODELIST')} "
                 f"ntasks={os.environ.get('SLURM_NTASKS')}")

    check_memlock()
    check_gpu_health()
    check_neighbours(max_load=args.max_node_load)
    check_hca(args.hca, outfile=args.bad_nodes_out)
    gpu = check_homogeneous(strict=not args.allow_heterogeneous,
                            outfile=args.bad_nodes_out)
    check_ckpt_dir(args.ckpt_dir, args.min_free_gb)
    busbw = None if args.skip_nccl else check_nccl(args.min_busbw)

    if args.runlog and os.environ.get("SLURM_PROCID", "0") == "0":
        rec = {
            "kind": "preflight",
            "ts": time.time(),
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "nodelist": os.environ.get("SLURM_JOB_NODELIST"),
            "num_nodes": os.environ.get("SLURM_JOB_NUM_NODES"),
            "gpu": gpu,
            "busbw_gbs": busbw,
            "constraint": os.environ.get("RESILIENCE_CONSTRAINT"),
            "failures": FAIL,
            "warnings": WARN,
        }
        Path(args.runlog).parent.mkdir(parents=True, exist_ok=True)
        with open(args.runlog, "a") as f:
            f.write(json.dumps(rec) + "\n")

    _say("--  ", f"{len(FAIL)} failure(s), {len(WARN)} warning(s)")
    if FAIL:
        _say("--  ", "refusing to start training:")
        for m in FAIL:
            _say("--  ", f"  - {m}")
        return 1
    _say("--  ", "all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
