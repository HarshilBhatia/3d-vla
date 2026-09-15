"""RLBench backend selection during the evaluator migration.

This module is the single, testable place that selects an adapter. It uses lazy
imports so ordinary tooling does not require simulator dependencies. Historical
modules under ``online_evaluation_rlbench`` now forward here for compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from importlib import import_module
from types import ModuleType


class RLBenchBackend(str, Enum):
    """Supported RLBench environment/action-adapter combinations."""

    PERACT = "peract"
    HIVEFORMER = "hiveformer"
    ORBITAL = "orbital"
    BIMANUAL = "bimanual"
    ORBITAL_BIMANUAL = "orbital_bimanual"


@dataclass(frozen=True)
class RLBenchRequest:
    """The capability pair that determines the simulator harness."""

    dataset_name: str
    bimanual: bool

    @property
    def normalized_dataset_name(self) -> str:
        return self.dataset_name.strip().lower()


_LEGACY_MODULES: dict[RLBenchBackend, str] = {
    RLBenchBackend.PERACT: "evaluation.online.rlbench.backends.peract",
    RLBenchBackend.HIVEFORMER: "evaluation.online.rlbench.backends.hiveformer",
    RLBenchBackend.ORBITAL: "evaluation.online.rlbench.backends.orbital",
    RLBenchBackend.BIMANUAL: "evaluation.online.rlbench.backends.bimanual",
    RLBenchBackend.ORBITAL_BIMANUAL: "evaluation.online.rlbench.backends.orbital_bimanual",
}


def resolve_backend(request: RLBenchRequest) -> RLBenchBackend:
    """Select a supported backend or fail with an actionable error.

    Orbital+bimanual is checked first because it has a dedicated composition
    harness. Unknown datasets no longer silently select the HiveFormer path.
    """
    dataset = request.normalized_dataset_name
    is_orbital = "orbital" in dataset
    is_peract = "peract" in dataset

    if request.bimanual and is_orbital:
        return RLBenchBackend.ORBITAL_BIMANUAL
    if request.bimanual:
        return RLBenchBackend.BIMANUAL
    if is_orbital:
        return RLBenchBackend.ORBITAL
    if is_peract:
        return RLBenchBackend.PERACT
    if "hiveformer" in dataset:
        return RLBenchBackend.HIVEFORMER
    raise ValueError(
        "No RLBench evaluation backend is registered for "
        f"dataset={request.dataset_name!r}, bimanual={request.bimanual}."
    )


def load_backend(request: RLBenchRequest) -> ModuleType:
    """Load the selected simulator harness on demand."""
    return import_module(_LEGACY_MODULES[resolve_backend(request)])


# Compatibility alias while callers migrate to the canonical name.
load_legacy_backend = load_backend
