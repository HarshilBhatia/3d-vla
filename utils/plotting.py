"""Shared plotting theme and result-loading helpers for 3DFA experiments.

This module deliberately keeps experiment semantics out of the visual layer.
Individual figure scripts declare their arms, conditions, and protocol notes;
they reuse the same strict result validation and presentation conventions here.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


# Semantic, accessible colors. Keep the same arm colors across all figures.
COLORS = {
    "baseline": "#606B75",
    "r1a": "#426A8C",
    "r1b": "#0F8B8D",
    "r1c": "#155C8D",
    "deltam": "#087E8B",
    "reference": "#A65E2E",
    "text_muted": "#4E5965",
    "grid": "#D9DEE3",
    "panel": "#FAFBFC",
    "annotation": "#1F2933",
}

DEFAULT_TASK_COUNT = 13
DEFAULT_DPI = 240


def configure_theme() -> None:
    """Apply the shared publication/presentation Matplotlib theme."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "hatch.linewidth": 1.0,
        }
    )


def style_axis(
    ax: Axes,
    *,
    y_grid: bool = True,
    success_axis: bool = False,
    y_lim: tuple[float, float] | None = None,
    y_ticks: Sequence[float] | None = None,
) -> None:
    """Apply common panel styling and optional success-rate axis settings."""
    ax.set_facecolor(COLORS["panel"])
    ax.set_axisbelow(True)
    if y_grid:
        ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
    if y_lim is not None:
        ax.set_ylim(*y_lim)
    if y_ticks is not None:
        ax.set_yticks(y_ticks)
        if success_axis:
            ax.set_yticklabels([f"{tick:g}%" for tick in y_ticks])


def add_protocol_note(
    ax: Axes,
    text: str,
    *,
    x: float = 0.985,
    y: float = 0.965,
    ha: str = "right",
    va: str = "top",
    fontsize: float = 8.6,
) -> None:
    """Add a consistent boxed protocol note in axes-relative coordinates."""
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=fontsize,
        color=COLORS["text_muted"],
        linespacing=1.35,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 2.5},
    )


def load_task_means(directory: Path, *, expected_tasks: int = DEFAULT_TASK_COUNT) -> dict[str, float]:
    """Read and validate one directory of per-task online-evaluation JSONs.

    Progress files are ignored; duplicated task names, invalid finite means, and
    missing tasks fail loudly. This makes a figure fail rather than silently
    averaging partial or collided outputs.
    """
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    result: dict[str, float] = {}
    for path in sorted(directory.glob("results_*.json")):
        # Evaluation directories may include provenance sidecars alongside the
        # task-result JSONs.  They are not result payloads.
        if path.name.endswith((".progress.json", ".manifest.json")):
            continue
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError as error:
            raise RuntimeError(f"Invalid JSON: {path}") from error
        for task, values in payload.items():
            if task in result:
                raise RuntimeError(f"Duplicate task {task}: {path}")
            mean = values.get("mean") if isinstance(values, dict) else None
            if not isinstance(mean, (int, float)) or not math.isfinite(mean):
                raise RuntimeError(f"Invalid mean for {task}: {path}")
            result[task] = float(mean)
    if len(result) != expected_tasks:
        raise RuntimeError(f"Expected {expected_tasks} tasks in {directory}; found {len(result)}")
    return result


def assert_shared_tasks(named_results: Mapping[str, Mapping[str, float]]) -> tuple[str, ...]:
    """Ensure every named result matrix covers exactly the same task set."""
    iterator = iter(named_results.items())
    try:
        reference_name, reference = next(iterator)
    except StopIteration as error:
        raise ValueError("No result matrices supplied") from error
    task_set = set(reference)
    for name, values in iterator:
        if set(values) != task_set:
            missing = sorted(task_set - set(values))
            extra = sorted(set(values) - task_set)
            raise RuntimeError(f"Task set mismatch: {name}; missing={missing}, extra={extra}")
    return tuple(sorted(task_set))


def mean_success(task_means: Mapping[str, float]) -> float:
    """Return the equally task-weighted mean success rate."""
    if not task_means:
        raise ValueError("Cannot aggregate an empty task matrix")
    return float(np.mean(list(task_means.values())))


def rope_phase_bound(
    rotation_degrees: np.ndarray | Sequence[float],
    translation_m: np.ndarray | Sequence[float],
    *,
    rho_m: float,
    omega_max: float = 1.0,
) -> np.ndarray:
    """Evaluate ``omega_max * (2 sin(alpha/2) rho + tau)`` in radians.

    ``rho_m`` is a conservative scene/workspace radius in metres and
    ``omega_max`` is the largest coordinate-RoPE spatial frequency in rad/m.
    The result is a phase-error upper bound, not a task-success prediction.
    """
    rotation = np.asarray(rotation_degrees, dtype=float)
    translation = np.asarray(translation_m, dtype=float)
    if rotation.shape != translation.shape:
        raise ValueError("rotation_degrees and translation_m must have the same shape")
    if rho_m < 0 or omega_max < 0:
        raise ValueError("rho_m and omega_max must be non-negative")
    return omega_max * (2 * np.sin(np.deg2rad(rotation) / 2) * rho_m + translation)


def fit_clean_anchored_phase_trend(
    phase_bound: np.ndarray | Sequence[float], success: np.ndarray | Sequence[float]
) -> tuple[np.ndarray, float]:
    """Fit a non-negative linear degradation from a clean, zero-bound point.

    Fits ``success[0] - scale * phase_bound`` by least squares with the clean
    condition fixed at ``success[0]``. The returned scale has units of success
    fraction per radian and is a visualization calibration only.
    """
    bound = np.asarray(phase_bound, dtype=float)
    values = np.asarray(success, dtype=float)
    if bound.ndim != 1 or values.ndim != 1 or bound.shape != values.shape:
        raise ValueError("phase_bound and success must be matching 1D arrays")
    if len(bound) < 2 or not math.isclose(float(bound[0]), 0.0, abs_tol=1e-12):
        raise ValueError("the first condition must be the clean, zero-bound condition")
    denominator = float(np.dot(bound[1:], bound[1:]))
    if denominator == 0:
        raise ValueError("at least one non-clean phase bound must be positive")
    scale = max(0.0, float(np.dot(bound[1:], values[0] - values[1:]) / denominator))
    return values[0] - scale * bound, scale


def save_figure(fig: Figure, stem: Path, *, dpi: int = DEFAULT_DPI) -> tuple[Path, Path]:
    """Save the standard high-resolution PNG and editable SVG outputs."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = stem.with_suffix(".png")
    svg_path = stem.with_suffix(".svg")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    return png_path, svg_path


def write_markdown_table(path: Path, lines: Iterable[str]) -> Path:
    """Write a reproducible compact Markdown summary next to a figure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n")
    return path
