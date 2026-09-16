"""Compatibility contracts for the trainer package extraction."""

import importlib

import pytest

pytest.importorskip("torch")
pytest.importorskip("hydra")
pytest.importorskip("wandb")


def test_training_is_canonical_and_legacy_registry_forwards():
    canonical = importlib.import_module("training")
    legacy = importlib.import_module("utils.trainers")
    assert canonical.fetch_train_tester is not None
    assert legacy.fetch_train_tester is canonical.fetch_train_tester
    assert legacy.RLBenchTrainTester is canonical.RLBenchTrainTester
    assert legacy.PeractTrainTester is canonical.PeractTrainTester
    assert canonical.fetch_train_tester("OrbitalPeract2") is canonical.RLBenchTrainTester
    assert canonical.fetch_train_tester("PeractCollected") is canonical.RLBenchTrainTester
    assert canonical.fetch_train_tester("Peract") is canonical.PeractTrainTester


def test_metrics_are_owned_by_common_not_training():
    metrics = importlib.import_module("common.metrics")
    legacy = importlib.import_module("utils.trainers.utils")
    assert metrics.compute_metrics.__module__ == "common.metrics"
    assert legacy.compute_metrics is metrics.compute_metrics
