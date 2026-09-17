"""Checkpoints are written off the training thread, without racing it."""

import pathlib
import time

import pytest
import torch

from training.base import _AsyncSaver, _cpu_snapshot


def test_snapshot_isolates_from_later_mutation():
    src = {"p": torch.ones(4), "nest": [torch.zeros(2), {"x": torch.full((2,), 3.0)}]}
    snap = _cpu_snapshot(src)
    src["p"].add_(99)
    src["nest"][0].add_(99)
    src["nest"][1]["x"].add_(99)
    assert snap["p"].tolist() == [1, 1, 1, 1]
    assert snap["nest"][0].tolist() == [0, 0]
    assert snap["nest"][1]["x"].tolist() == [3, 3]


def test_snapshot_passes_non_tensors_through():
    assert _cpu_snapshot({"a": 5, "b": "s", "c": None}) == {"a": 5, "b": "s", "c": None}


def test_queued_writes_land_in_order(tmp_path):
    saver = _AsyncSaver()
    for i in range(6):
        saver.save({"i": i}, tmp_path / "last.pth")
    saver.close()
    assert torch.load(tmp_path / "last.pth", weights_only=False)["i"] == 5


def test_background_error_reaches_the_training_thread(tmp_path):
    saver = _AsyncSaver()
    saver.save({"x": torch.ones(2)}, tmp_path / "missing" / "f.pth")
    time.sleep(0.5)
    with pytest.raises(Exception):
        saver.save({"x": torch.ones(2)}, tmp_path / "ok.pth")
        saver.flush()


def test_close_drains_and_leaves_no_partial_files(tmp_path):
    saver = _AsyncSaver()
    saver.save({"v": torch.arange(1000)}, tmp_path / "big.pth")
    saver.close()
    assert (tmp_path / "big.pth").exists()
    assert not list(tmp_path.glob("*.tmp"))
