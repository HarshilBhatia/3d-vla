import numpy as np

from datasets.samplers import DiverseChunkBatchSampler


class _Dataset:
    """Minimal BaseDataset-shaped object for sampler tests."""

    def __init__(self):
        # Four contiguous recordings, six logical timesteps each.
        self.copies = 1
        self.chunk_size = 1
        self.annos = {
            "action": np.zeros((24, 1)),
            "demo_id": np.repeat(np.arange(4), 6),
        }

    def __len__(self):
        return 24


def _demo_ids(dataset, batch):
    return [int(dataset.annos["demo_id"][i]) for i in batch]


def test_batches_have_one_context_per_demo():
    dataset = _Dataset()
    sampler = DiverseChunkBatchSampler(
        dataset, batch_size=4, cache_batches=2, cache_span=3, num_workers=2
    )
    batches = list(sampler)
    assert len(batches) == len(sampler)
    assert all(len(set(_demo_ids(dataset, batch))) == 4 for batch in batches)


def test_epoch_changes_draws_but_is_deterministic():
    dataset = _Dataset()
    sampler = DiverseChunkBatchSampler(dataset, batch_size=4, seed=5, num_workers=1)
    first = list(sampler)
    assert first == list(sampler)
    sampler.set_epoch(1)
    assert first != list(sampler)
