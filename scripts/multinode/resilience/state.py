"""Exactly-resumable RNG and data-order state.

This is the only part of the resilience layer with a real correctness
criterion: after a kill and resume, the trainer must see the *same* sample
order and the *same* random draws it would have seen uninterrupted.
``tests/test_kill_resume.py`` asserts exactly that.

Nothing here knows about models. It deals in indices and RNG bytes.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any, Iterator, Optional, Sequence

import numpy as np
import torch
import torch.distributed as dist


# ── RNG ───────────────────────────────────────────────────────────────────────

def capture_rng() -> dict:
    """Snapshot every RNG stream a training step can consume.

    CUDA state is captured for the *current* device only. Under Slurm each rank
    is cgroup-confined to one GPU that always presents as device 0, so a
    per-device list would be both redundant and unrestorable on a node with a
    different device count.
    """
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": None,
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state()
    return state


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    # torch wants a ByteTensor on CPU; torch.load(map_location="cpu") gives that.
    torch.set_rng_state(state["torch"].cpu() if torch.is_tensor(state["torch"]) else state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(state["cuda"].cpu() if torch.is_tensor(state["cuda"]) else state["cuda"])


def seed_everything(base_seed: int, rank: int) -> None:
    """Seed all streams deterministically from (base_seed, rank).

    Ranks must differ or every rank draws identical dropout masks; they must be
    derived from the base seed or a resume cannot reproduce them.
    """
    seed = (base_seed * 1_000_003 + rank) % (2**31 - 1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def gather_rng(state: dict, world_size: int) -> list[dict]:
    """Collect every rank's RNG state so one rank can write one checkpoint.

    RNG state is per-rank, but the checkpoint is written by rank 0 only. Without
    this gather, a resume would restore rank 0's stream onto every rank.
    """
    if world_size == 1 or not dist.is_initialized():
        return [state]
    out: list[Optional[dict]] = [None] * world_size
    dist.all_gather_object(out, state)
    return [s for s in out]  # type: ignore[misc]


# ── Data order ────────────────────────────────────────────────────────────────

@dataclass
class _Position:
    epoch: int = 0
    consumed: int = 0  # global samples consumed within the current epoch


class ResumableDistributedSampler(torch.utils.data.Sampler[int]):
    """A distributed sampler whose position is a checkpointable integer.

    ``DistributedSampler`` can only be rewound to an epoch boundary, which is
    useless when a job dies 40% through a 100k-step run. This derives the epoch
    permutation from ``(seed, epoch)`` and slices from ``consumed``, so restore
    is O(1) and does not depend on replaying the loader.

    Resharding: position is stored as a *global* sample count, so the same
    position is well-defined under a different world size. The data order is
    then still deterministic but not bit-identical to the original run --
    ``Checkpointer`` warns when it sees a world-size change.
    """

    def __init__(
        self,
        dataset_len: int,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = True,
    ) -> None:
        if num_replicas is None:
            num_replicas = dist.get_world_size() if dist.is_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_initialized() else 0
        self.dataset_len = dataset_len
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.pos = _Position()

        if drop_last:
            self.total_size = (dataset_len // num_replicas) * num_replicas
        else:
            self.total_size = dataset_len
        self.num_samples = self.total_size // num_replicas

    # -- position ------------------------------------------------------------

    def state_dict(self) -> dict:
        return {"epoch": self.pos.epoch, "consumed": self.pos.consumed}

    def load_state_dict(self, sd: dict) -> None:
        self.pos = _Position(epoch=int(sd["epoch"]), consumed=int(sd["consumed"]))

    def advance(self, n_global_samples: int) -> None:
        """Record that ``n_global_samples`` were consumed (call once per step)."""
        self.pos.consumed += n_global_samples
        if self.pos.consumed >= self.total_size:
            self.pos.consumed = 0
            self.pos.epoch += 1

    # -- iteration -----------------------------------------------------------

    def _epoch_order(self, epoch: int) -> torch.Tensor:
        if not self.shuffle:
            return torch.arange(self.dataset_len)
        g = torch.Generator()
        # Mixing epoch into the seed is what makes epoch N's order reproducible
        # from the checkpoint alone, with no loader replay.
        g.manual_seed(self.seed * 6_364_136 + epoch)
        return torch.randperm(self.dataset_len, generator=g)

    def __iter__(self) -> Iterator[int]:
        order = self._epoch_order(self.pos.epoch)[: self.total_size]
        # Skip whole global samples, then shard. Skipping before sharding keeps
        # the rank->index mapping identical to an uninterrupted run.
        remaining = order[self.pos.consumed :]
        mine = remaining[self.rank :: self.num_replicas]
        return iter(mine.tolist())

    def __len__(self) -> int:
        return max(0, (self.total_size - self.pos.consumed) // self.num_replicas)


class StatefulLoader:
    """Wraps a DataLoader, keeping both the sampler position and the loader's
    own RNG stream checkpointable.

    The second half is not optional and is easy to miss: every call to
    ``iter(DataLoader)`` draws one int64 from the *global* torch generator to
    derive ``_base_seed`` for worker seeding. A resumed run creates an extra
    iterator that the uninterrupted run never created, so the global stream
    ends up one draw out of step and every subsequent dropout mask differs --
    weights and data order both restored correctly, losses still wrong. Giving
    the loader a dedicated generator isolates it from the global stream, and
    checkpointing that generator keeps worker seeding reproducible when
    ``num_workers > 0``.
    """

    def __init__(self, loader: torch.utils.data.DataLoader, sampler: ResumableDistributedSampler,
                 global_batch_size: int, seed: int = 0) -> None:
        self.loader = loader
        self.sampler = sampler
        self.global_batch_size = global_batch_size
        # Own the generator so DataLoader never touches the global RNG.
        self.generator = torch.Generator()
        self.generator.manual_seed(seed * 2_654_435_761 % (2**63 - 1))
        self.loader.generator = self.generator

    def state_dict(self) -> dict:
        sd = self.sampler.state_dict()
        sd["loader_generator"] = self.generator.get_state()
        return sd

    def load_state_dict(self, sd: dict) -> None:
        self.sampler.load_state_dict(sd)
        gen = sd.get("loader_generator")
        if gen is not None:
            self.generator.set_state(gen.cpu() if torch.is_tensor(gen) else gen)

    def __iter__(self):
        while True:  # infinite stream: step-based training, not epoch-based
            for batch in self.loader:
                self.sampler.advance(self.global_batch_size)
                yield batch
            # Epoch exhausted without hitting the advance() rollover (drop_last
            # rounding): force the boundary so the next order differs.
            if self.sampler.pos.consumed != 0:
                self.sampler.pos.consumed = 0
                self.sampler.pos.epoch += 1


class SkipAheadSampler(torch.utils.data.Sampler):
    """Wrap an existing sampler so a resume can skip what it already consumed.

    Why a wrapper rather than replacing the sampler: an existing run's data
    order must not change. This preserves the inner sampler's order exactly and
    only discards the prefix the interrupted run had already seen, so a fresh
    run stays bit-for-bit identical while a resumed one lands on the batch it
    would have reached uninterrupted.

    ``mode="batch"`` wraps a ``batch_sampler`` (yields lists of indices);
    ``mode="index"`` wraps a plain sampler (yields single indices) and converts
    the batch skip using ``batch_size``.

    Discarding happens at the sampler level, so skipped batches cost only the
    index arithmetic -- no collation and no data loading.
    """

    def __init__(self, inner, mode: str = "batch", batch_size: Optional[int] = None) -> None:
        if mode not in ("batch", "index"):
            raise ValueError(f"mode must be 'batch' or 'index', got {mode!r}")
        if mode == "index" and not batch_size:
            raise ValueError("mode='index' requires batch_size to convert the batch skip")
        self.inner = inner
        self.mode = mode
        self.batch_size = batch_size
        self._skip = 0
        self._epoch = None

    # -- position ------------------------------------------------------------

    def set_skip(self, n_batches: int) -> None:
        """Discard this many batches for the duration of the current epoch.

        The skip is *sticky* until the epoch advances, deliberately. It must not
        be consumed by the first ``__iter__``: ``_MultiProcessingDataLoaderIter``
        builds the sampler iterator twice -- once in
        ``_BaseDataLoaderIter.__init__`` and again in ``_reset`` -- so a
        one-shot skip is eaten by the throwaway iterator and the real one starts
        from batch 0. That failure is silent and only appears with
        ``num_workers > 0``, and only for a ``batch_sampler`` (a plain sampler
        gets wrapped in a generator-based ``BatchSampler``, which defers and so
        happens to survive).
        """
        self._skip = max(0, int(n_batches))

    @property
    def skip(self) -> int:
        return self._skip

    def state_dict(self) -> dict:
        inner_sd = self.inner.state_dict() if hasattr(self.inner, "state_dict") else None
        return {"skip": self._skip, "inner": inner_sd}

    def load_state_dict(self, sd: dict) -> None:
        self._skip = int(sd.get("skip", 0))
        inner_sd = sd.get("inner")
        if inner_sd is not None and hasattr(self.inner, "load_state_dict"):
            self.inner.load_state_dict(inner_sd)

    # -- delegation ----------------------------------------------------------

    def set_epoch(self, epoch: int) -> None:
        # Advancing the epoch is what clears the skip: the resume prefix applies
        # to exactly one epoch, and the trainer bumps the epoch on StopIteration.
        if self._epoch is not None and epoch != self._epoch:
            self._skip = 0
        self._epoch = epoch
        if hasattr(self.inner, "set_epoch"):
            self.inner.set_epoch(epoch)

    def __len__(self) -> int:
        # Deliberately the *full* epoch length, not the post-skip length: the
        # trainer derives samples_per_epoch from len(loader), and that must stay
        # constant across a resume. The shortened epoch shows up as an early
        # StopIteration, which the loop already handles by bumping the epoch.
        return len(self.inner)

    def __iter__(self):
        it = iter(self.inner)
        skip = self._skip  # sticky until set_epoch advances; see set_skip
        if skip:
            if self.mode == "batch":
                to_drop = skip
            else:
                to_drop = skip * int(self.batch_size)
            dropped = 0
            for _ in it:
                dropped += 1
                if dropped >= to_drop:
                    break
        # Optional audit trail of the exact index sequence. Off unless
        # RESILIENCE_INDEX_LOG is set, so the normal path is untouched.
        log = os.environ.get("RESILIENCE_INDEX_LOG")
        if log:
            return self._logging_iter(it, log, skip)
        return it

    def _logging_iter(self, it, path: str, skip: int):
        """Record every batch of indices this sampler hands out.

        Weight comparison cannot validate a resume on a stack whose kernels are
        nondeterministic at ~1e-2. Indices are integers: comparing the sequence
        a resumed run consumes against an uninterrupted one is exact, and it is
        the specific thing that breaks silently when only the epoch is restored.
        """
        rank = os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0"))
        f = open(f"{path}.rank{rank}.txt", "a", buffering=1)
        f.write(f"# epoch_start skip={skip} mode={self.mode}\n")
        n = skip
        try:
            if self.mode == "batch":
                for batch in it:
                    f.write(f"{n} {' '.join(map(str, batch))}\n")
                    n += 1
                    yield batch
            else:
                bs = int(self.batch_size)
                buf = []
                for idx in it:
                    buf.append(idx)
                    if len(buf) == bs:
                        f.write(f"{n} {' '.join(map(str, buf))}\n")
                        n += 1
                        buf = []
                    yield idx
        finally:
            f.close()
