"""Preemption signal handling.

Slurm's ``preempt_send_user_signal`` is set cluster-wide on grogu, so a
preempted job receives its own ``--signal`` immediately at preempt time, then
SIGTERM, then SIGKILL after ``GraceTime`` (120 s, admin-set, not extendable).

The handler deliberately does almost nothing: it flips a flag. Writing a
checkpoint from inside a signal handler risks tearing a collective, so the
actual save happens at the next step boundary where every rank agrees.
"""

from __future__ import annotations

import os
import signal
import time
from pathlib import Path
from typing import Optional


class PreemptionGuard:
    """Flag-flipping signal handler plus a filesystem sentinel.

    The sentinel exists because only the batch-script shell reliably receives
    the Slurm signal; ``srun`` children may not. The batch script touches the
    sentinel, and every rank polls it. Both paths set the same flag.
    """

    def __init__(self, sentinel: Optional[str | Path] = None, poll_every: int = 20,
                 done_marker: Optional[str | Path] = None) -> None:
        self.preempted = False
        self.at: Optional[float] = None
        self.poll_every = poll_every
        self._n = 0
        env_sentinel = os.environ.get("RESILIENCE_SENTINEL")
        self.sentinel = Path(sentinel or env_sentinel) if (sentinel or env_sentinel) else None
        env_done = os.environ.get("RESILIENCE_DONE_MARKER")
        self.done_marker = Path(done_marker or env_done) if (done_marker or env_done) else None

        for sig in (signal.SIGUSR1, signal.SIGTERM):
            try:
                signal.signal(sig, self._handle)
            except (ValueError, OSError):
                pass  # not on the main thread (e.g. a dataloader worker)

    def _handle(self, signum, frame):  # noqa: ARG002
        if not self.preempted:
            self.preempted = True
            self.at = time.time()
            print(f"[preempt] caught {signal.Signals(signum).name}; "
                  f"checkpointing at next step boundary", flush=True)

    def confirm(self) -> None:
        """Tell the batch script the checkpoint is on disk, so it can requeue now.

        Without this the trap has no way to distinguish "saved" from "still
        saving" and must wait out its whole timeout -- which eats the 120 s
        preemption grace window that the save itself needs.
        """
        if self.done_marker is None:
            return
        try:
            self.done_marker.parent.mkdir(parents=True, exist_ok=True)
            self.done_marker.touch()
            print(f"[preempt] confirmed checkpoint via {self.done_marker}", flush=True)
        except OSError as e:
            print(f"[preempt] WARNING: could not write {self.done_marker}: {e}", flush=True)

    def check(self) -> bool:
        """Call once per step. Cheap: the sentinel is stat'ed every N steps."""
        self._n += 1
        if not self.preempted and self.sentinel is not None and self._n % self.poll_every == 0:
            if self.sentinel.exists():
                self._handle(signal.SIGUSR1, None)
        return self.preempted
