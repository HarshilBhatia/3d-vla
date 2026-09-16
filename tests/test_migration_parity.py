"""Opt-in checkpoint parity contract for behavior-preserving refactors.

This is deliberately *not* part of the normal test suite.  Run it immediately
before and after a mechanical migration (for example, moving a utility module):

    THREEDFA_MIGRATION_PARITY_CASE=/abs/path/case.json \\
      pytest -m migration_parity -q

The case names an immutable checkpoint, one already-preprocessed model input,
and its golden inference output.  It therefore catches changes in module
construction, checkpoint loading, tensor wiring, and numerical behavior without
requiring a simulator or a dataset mount.
"""

from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest


CASE_ENV = "THREEDFA_MIGRATION_PARITY_CASE"
_REQUIRED_BATCH_KEYS = frozenset({
    "gt_trajectory",
    "trajectory_mask",
    "rgb3d",
    "rgb2d",
    "pcd",
    "instruction",
    "proprio",
})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _case_path() -> Path:
    value = os.environ.get(CASE_ENV)
    if not value:
        pytest.skip(f"set {CASE_ENV} to run the manual migration-parity contract")
    path = Path(value)
    if not path.is_file():
        pytest.fail(f"migration parity case does not exist: {path}")
    return path


@pytest.fixture(scope="module")
def parity_case() -> dict[str, Any]:
    path = _case_path()
    case = json.loads(path.read_text())
    required = {"schema_version", "checkpoint", "checkpoint_sha256", "input", "expected_output", "seed"}
    missing = required - set(case)
    assert not missing, f"{path} is missing required keys: {sorted(missing)}"
    assert case["schema_version"] == 1
    case["_path"] = path
    for key in ("checkpoint", "input", "expected_output"):
        value = Path(case[key])
        if not value.is_absolute():
            value = path.parent / value
        assert value.is_file(), f"case {key} does not exist: {value}"
        case[key] = value
    return case


@contextmanager
def _deterministic_torch(torch):
    """Make capture and comparison use the same conservative CUDA settings."""
    old = {
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
    }
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(old["deterministic"])
        torch.backends.cudnn.deterministic = old["cudnn_deterministic"]
        torch.backends.cudnn.benchmark = old["cudnn_benchmark"]
        torch.backends.cuda.matmul.allow_tf32 = old["matmul_tf32"]
        torch.backends.cudnn.allow_tf32 = old["cudnn_tf32"]


def _to_device(value: Any, device):
    if hasattr(value, "to"):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_device(item, device) for item in value)
    return value


@pytest.mark.migration_parity
def test_parity_case_pins_the_exact_checkpoint(parity_case):
    """Prevent an accidental checkpoint substitution from making a green result meaningless."""
    actual = _sha256(parity_case["checkpoint"])
    assert actual == parity_case["checkpoint_sha256"], (
        "checkpoint digest differs from the recorded parity case; capture a new "
        "case only after deliberately selecting a new checkpoint"
    )


@pytest.fixture(scope="module")
def loaded_model(parity_case):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("migration parity is a GPU checkpoint contract")

    # This is the canonical evaluation loader: the test exercises the same
    # checkpoint-config overlay and constructor path used for online rollout.
    from evaluation.checkpoints import load_model_for_evaluation
    from utils.hydra_utils import get_config, get_config_path

    args = get_config(overrides=[], config_name="config", config_path=get_config_path())
    args.checkpoint = parity_case["checkpoint"]
    with _deterministic_torch(torch):
        model = load_model_for_evaluation(args)
    assert not hasattr(model, "_orig_mod"), "parity must use eager mode; do not torch.compile this contract"
    return model


@pytest.mark.migration_parity
def test_checkpoint_loads_eagerly_for_migration_parity(loaded_model):
    """Smoke-test the checkpoint/config/state-dict compatibility path."""
    assert loaded_model.training is False
    assert next(loaded_model.parameters()).is_cuda


@pytest.mark.migration_parity
def test_deterministic_forward_matches_frozen_golden(parity_case, loaded_model):
    """The behavior-preservation assertion used before and after a code move."""
    torch = pytest.importorskip("torch")
    batch = torch.load(parity_case["input"], map_location="cpu", weights_only=False)
    assert isinstance(batch, dict), "input artifact must be a dict of model.forward keyword arguments"
    missing = _REQUIRED_BATCH_KEYS - set(batch)
    assert not missing, f"input artifact is missing model inputs: {sorted(missing)}"

    expected = torch.load(parity_case["expected_output"], map_location="cpu", weights_only=False)
    assert isinstance(expected, torch.Tensor), "expected output artifact must be a tensor"
    device_batch = _to_device(batch, torch.device("cuda"))

    with _deterministic_torch(torch), torch.inference_mode():
        torch.manual_seed(int(parity_case["seed"]))
        torch.cuda.manual_seed_all(int(parity_case["seed"]))
        actual = loaded_model(**device_batch, run_inference=True).cpu()

    # Exact comparison is the default.  A case may explicitly record a small
    # tolerance only when a reviewed backend/library change makes it necessary.
    torch.testing.assert_close(
        actual,
        expected,
        rtol=float(parity_case.get("rtol", 0.0)),
        atol=float(parity_case.get("atol", 0.0)),
        msg="migration changed deterministic policy output",
    )
