"""Compatibility forwarding package; use :mod:`data.preprocessing`."""

from data.preprocessing import (  # noqa: F401
    PeractDataPreprocessor,
    RLBenchDataPreprocessor,
    fetch_data_preprocessor,
)
