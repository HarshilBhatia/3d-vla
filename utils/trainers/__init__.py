"""Compatibility forwarding package; use :mod:`training`."""

from training import (  # noqa: F401
    PeractTrainTester,
    RLBenchTrainTester,
    fetch_train_tester,
)
