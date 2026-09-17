"""Atomic, rank-aware, time-driven checkpointing.

Design constraints that come from the cluster, not from taste:

* ``GraceTime=120`` is set per-partition by the admins. On preemption a job is
  signalled and then killed 120 s later, and ``--signal=...@N`` does **not**
  extend that window (it only moves the *time-limit* warning). So a checkpoint
  that takes longer than ~120 s cannot be written on the way out, and the
  periodic timer -- not the signal handler -- is the actual safety net.
* Checkpoints must land on NFS. ``/tmp`` on grogu is node-local, so a
  checkpoint written there is invisible to the next allocation.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Optional

import torch
import torch.distributed as dist

from .state import capture_rng, gather_rng, restore_rng

_NODE_LOCAL_PREFIXES = ("/tmp", "/var/tmp", "/dev/shm", "/scratch/local")


def _atomic_save(obj: Any, path: Path) -> float:
    """Write via temp file + rename. Returns seconds taken.

    The rename is atomic within a filesystem, so a kill mid-write leaves the
    previous checkpoint intact rather than a truncated one.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        os.close(fd)
        with open(tmp, "wb") as f:
            torch.save(obj, f)
            f.flush()
            os.fsync(f.fileno())  # NFS: without this the rename can win the race
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return time.time() - t0


class Checkpointer:
    """The only two touchpoints ``train.py`` needs.

    ``maybe_save(step, build_state)`` is called every step and is cheap when it
    is not time to save. ``resume(apply_state)`` is called once at startup.

    ``build_state`` returns the model-specific half (weights, optimizer, ...);
    this class owns the model-agnostic half (step, RNG, data position,
    world size, provenance) and never inspects the former.
    """

    def __init__(
        self,
        ckpt_dir: str | Path,
        interval_seconds: float = 600.0,
        keep_last: int = 2,
        keep_every_steps: Optional[int] = None,
        rank: int = 0,
        world_size: int = 1,
        require_shared_fs: bool = True,
    ) -> None:
        self.dir = Path(ckpt_dir).resolve()
        self.interval = interval_seconds
        self.keep_last = keep_last
        self.keep_every_steps = keep_every_steps
        self.rank = rank
        self.world_size = world_size
        self._last_save = time.time()
        self._last_duration: Optional[float] = None
        self._forced = False

        if require_shared_fs:
            self._assert_shared_fs()
        if self.rank == 0:
            self.dir.mkdir(parents=True, exist_ok=True)

    # -- filesystem ----------------------------------------------------------

    def _assert_shared_fs(self) -> None:
        p = str(self.dir)
        for bad in _NODE_LOCAL_PREFIXES:
            if p == bad or p.startswith(bad + "/"):
                raise SystemExit(
                    f"[checkpointer] FATAL: ckpt_dir={p} is node-local. On grogu /tmp is "
                    f"per-node, so the next allocation would start from scratch. "
                    f"Use a path under /home or /grogu/user."
                )

    # -- paths ---------------------------------------------------------------

    @property
    def latest_path(self) -> Path:
        return self.dir / "latest.pt"

    def _step_path(self, step: int) -> Path:
        return self.dir / f"step_{step:09d}.pt"

    def find_resume(self) -> Optional[Path]:
        if self.latest_path.exists():
            return self.latest_path
        steps = sorted(self.dir.glob("step_*.pt"))
        return steps[-1] if steps else None

    # -- saving --------------------------------------------------------------

    def request_save(self) -> None:
        """Ask for a save at the next step boundary (used by the signal handler)."""
        self._forced = True

    def due(self) -> bool:
        return self._forced or (time.time() - self._last_save) >= self.interval

    def maybe_save(self, step: int, build_state: Callable[[], dict], *, force: bool = False) -> bool:
        """Save iff due. Every rank must call this: rank 0 writes, all barrier.

        Returns True if a checkpoint was written.
        """
        want = force or self.due()
        # A time-based trigger fires at slightly different moments on different
        # ranks; without agreeing first, some ranks enter the barrier and others
        # don't, and the job deadlocks.
        if self.world_size > 1 and dist.is_initialized():
            flag = torch.tensor([1 if want else 0], dtype=torch.int32)
            if torch.cuda.is_available():
                flag = flag.cuda()
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            want = bool(flag.item())
        if not want:
            return False

        rng_all = gather_rng(capture_rng(), self.world_size)

        wrote = False
        if self.rank == 0:
            payload = dict(build_state())
            payload["_resilience"] = {
                "step": step,
                "rng": rng_all,
                "world_size": self.world_size,
                "saved_at": time.time(),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "nodelist": os.environ.get("SLURM_JOB_NODELIST"),
            }
            dur = _atomic_save(payload, self.latest_path)
            self._last_duration = dur
            if self.keep_every_steps and step % self.keep_every_steps == 0:
                shutil.copyfile(self.latest_path, self._step_path(step))
            self._prune()
            self._warn_if_slow(dur, step)
            wrote = True

        self._last_save = time.time()
        self._forced = False
        if self.world_size > 1 and dist.is_initialized():
            dist.barrier()
        return wrote

    def _warn_if_slow(self, dur: float, step: int) -> None:
        size_mb = self.latest_path.stat().st_size / 1e6
        size = f"{size_mb:.1f} MB" if size_mb >= 1 else f"{size_mb * 1000:.0f} KB"
        rate = f", {size_mb / dur:.0f} MB/s" if dur > 0.05 else ""
        print(f"[checkpointer] step {step}: wrote {size} in {dur:.2f}s{rate} "
              f"-> {self.latest_path}", flush=True)
        if dur > 90:
            print(
                f"[checkpointer] WARNING: {dur:.0f}s exceeds the 90s budget. GraceTime on this "
                f"cluster is 120s and is NOT extendable via --signal, so a preemption may kill "
                f"this job mid-write. Lower interval_seconds or shrink the payload; raising the "
                f"--signal lead time will not help.",
                flush=True,
            )

    def _prune(self) -> None:
        keeps = sorted(self.dir.glob("step_*.pt"))
        if self.keep_last > 0 and len(keeps) > self.keep_last:
            for old in keeps[: -self.keep_last]:
                try:
                    old.unlink()
                except OSError:
                    pass

    # -- resuming ------------------------------------------------------------

    def resume(self, apply_state: Callable[[dict], None]) -> int:
        """Load the newest checkpoint and return the step to continue from.

        Returns 0 when there is nothing to resume.
        """
        path = self.find_resume()
        if path is None:
            if self.rank == 0:
                print(f"[checkpointer] no checkpoint in {self.dir} — starting from scratch", flush=True)
            return 0

        # map_location='cpu' keeps the checkpoint portable across GPU counts and
        # models; the caller moves tensors to its own device.
        payload = torch.load(path, map_location="cpu", weights_only=False)
        meta = payload.pop("_resilience")
        step = int(meta["step"])

        if meta["world_size"] != self.world_size:
            if self.rank == 0:
                print(
                    f"[checkpointer] WARNING: resuming a world_size={meta['world_size']} run at "
                    f"world_size={self.world_size}. Data order is still deterministic but NOT "
                    f"bit-identical, and per-rank RNG cannot be restored. Loss curves will "
                    f"diverge from the original run.",
                    flush=True,
                )
        else:
            restore_rng(meta["rng"][self.rank])

        apply_state(payload)
        if self.rank == 0:
            print(f"[checkpointer] resumed {path} at step {step} "
                  f"(saved by job {meta.get('slurm_job_id')} on {meta.get('nodelist')})", flush=True)
        return step
