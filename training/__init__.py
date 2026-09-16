"""Training orchestration and trainer implementations."""

from .peract import PeractTrainTester
from .rlbench import RLBenchTrainTester


def fetch_train_tester(dataset_name):
    """Return the trainer implementation for a dataset identifier."""
    dataset_name = dataset_name.lower()
    if "peract2" in dataset_name or "peractcollected" in dataset_name:
        return RLBenchTrainTester
    if "peract" in dataset_name:
        return PeractTrainTester
    if "rlbench" in dataset_name or "orbital" in dataset_name:
        return RLBenchTrainTester
    return None
