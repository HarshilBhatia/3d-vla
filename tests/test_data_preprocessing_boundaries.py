"""Import and registry contracts for the migrated data preprocessing domain."""

import importlib

import pytest

pytest.importorskip("torch")
pytest.importorskip("kornia")


def test_data_preprocessing_public_modules_are_canonical():
    preprocessing = importlib.import_module("data.preprocessing")
    geometry = importlib.import_module("data.geometry")
    assert preprocessing.fetch_data_preprocessor.__module__ == "data.preprocessing"
    assert geometry.fetch_depth2cloud.__module__ == "data.geometry"
    assert preprocessing.RLBenchDataPreprocessor.__module__ == "data.preprocessing.rlbench"
    assert geometry.RLBenchDepth2Cloud.__module__ == "data.geometry.rlbench"


def test_legacy_preprocessing_paths_forward_to_canonical_objects():
    old_pre = importlib.import_module("utils.data_preprocessors")
    new_pre = importlib.import_module("data.preprocessing")
    old_geo = importlib.import_module("utils.depth2cloud")
    new_geo = importlib.import_module("data.geometry")
    assert old_pre.fetch_data_preprocessor is new_pre.fetch_data_preprocessor
    assert old_pre.RLBenchDataPreprocessor is new_pre.RLBenchDataPreprocessor
    assert old_geo.fetch_depth2cloud is new_geo.fetch_depth2cloud
    assert old_geo.RLBenchDepth2Cloud is new_geo.RLBenchDepth2Cloud
