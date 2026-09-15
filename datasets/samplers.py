"""Batch samplers for temporal, multi-camera training.

The normal ``DistributedSampler`` maximises global shuffle, but it also makes a
worker jump to unrelated zarr chunks for every sample.  The sampler below keeps
a *small* worker-local working set of episodes for a few batches.  Each batch
still contains one independently drawn time point from distinct episodes, so
the examples within a gradient update are not correlated.
"""

from collections import defaultdict
import math

import numpy as np
from torch.utils.data import Sampler


class DiverseChunkBatchSampler(Sampler):
    """DDP-aware, episode-diverse batches with bounded temporal locality.

    A lane is assigned to each loader worker.  For ``cache_batches`` passes a
    lane reuses its set of ``batch_size`` *different* demos, while drawing a
    fresh target time for every demo and batch.  Targets are constrained to a
    short logical-index span around a newly sampled anchor, allowing the
    worker's zarr LRU cache to reuse history frames.  Lanes are emitted round
    robin, matching DataLoader's round-robin worker dispatch.

    This deliberately optimises locality *across* batches, never by placing
    multiple samples from one recording in the same batch.
    """

    def __init__(
        self,
        dataset,
        batch_size,
        num_replicas=1,
        rank=0,
        drop_last=True,
        seed=0,
        cache_batches=1,
        cache_span=8,
        num_workers=1,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if cache_batches <= 0 or cache_span <= 0 or num_workers <= 0:
            raise ValueError("cache_batches, cache_span, and num_workers must be positive")
        if "demo_id" not in dataset.annos:
            raise ValueError("DiverseChunkBatchSampler requires a demo_id annotation")

        self.dataset = dataset
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.drop_last = drop_last
        self.seed = seed
        self.cache_batches = cache_batches
        self.cache_span = cache_span
        self.num_workers = num_workers
        self.epoch = 0

        # Dataset indices are logical chunks; map them to the physical zarr row
        # used by __getitem__ before reading demo_id.
        physical_rows = np.arange(len(dataset) // dataset.copies) * dataset.chunk_size
        demo_ids = np.asarray(dataset.annos["demo_id"][physical_rows])
        self._by_demo = defaultdict(list)
        for logical_idx, demo_id in enumerate(demo_ids.tolist()):
            self._by_demo[int(demo_id)].append(logical_idx)
        self._demos = np.asarray(sorted(self._by_demo), dtype=np.int64)
        if len(self._demos) < batch_size:
            raise ValueError(
                f"batch_size={batch_size} requires at least that many demos for "
                f"within-batch diversity; dataset has {len(self._demos)} demos"
            )

        # Match DistributedSampler's per-rank sample count.  Keeping this
        # fixed makes the trainer's epoch and resume bookkeeping unchanged.
        global_samples = len(dataset)
        if drop_last:
            self.num_samples = global_samples // num_replicas
        else:
            self.num_samples = int(math.ceil(global_samples / num_replicas))
        self.num_batches = self.num_samples // batch_size if drop_last else int(
            math.ceil(self.num_samples / batch_size)
        )

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.num_batches

    def _sample_lane(self, rng):
        """Choose unique demos and a cache-friendly range for each one."""
        demos = rng.choice(self._demos, size=self.batch_size, replace=False)
        ranges = []
        for demo in demos:
            candidates = self._by_demo[int(demo)]
            anchor = candidates[int(rng.integers(len(candidates)))]
            local = [i for i in candidates if anchor <= i < anchor + self.cache_span]
            # Near the end of an episode, use the preceding span instead.
            if not local:
                local = [i for i in candidates if anchor - self.cache_span < i <= anchor]
            ranges.append(local or candidates)
        return ranges

    def __iter__(self):
        # rank is intentionally part of the seed: distinct DDP ranks should
        # not warm identical episode caches or form identical local batches.
        rng = np.random.default_rng(self.seed + 1_000_003 * self.epoch + 10_007 * self.rank)
        emitted = 0
        while emitted < self.num_batches:
            lanes = [self._sample_lane(rng) for _ in range(self.num_workers)]
            for _ in range(self.cache_batches):
                for lane in lanes:
                    if emitted >= self.num_batches:
                        return
                    yield [choices[int(rng.integers(len(choices)))] for choices in lane]
                    emitted += 1
