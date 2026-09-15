import pytest

from evaluation.online.rlbench import RLBenchBackend, RLBenchRequest, resolve_backend


@pytest.mark.parametrize(
    ("dataset_name", "bimanual", "expected"),
    [
        ("Peract2_3dfront_3dwrist", False, RLBenchBackend.PERACT),
        ("PeractCollected", False, RLBenchBackend.PERACT),
        ("HiveformerRLBench", False, RLBenchBackend.HIVEFORMER),
        ("OrbitalWrist", False, RLBenchBackend.ORBITAL),
        ("OrbitalPeract2", True, RLBenchBackend.ORBITAL_BIMANUAL),
        ("Peract2_3dfront_3dwrist", True, RLBenchBackend.BIMANUAL),
    ],
)
def test_resolve_backend(dataset_name, bimanual, expected):
    assert resolve_backend(RLBenchRequest(dataset_name, bimanual)) is expected


def test_unknown_dataset_is_rejected():
    with pytest.raises(ValueError, match="No RLBench evaluation backend"):
        resolve_backend(RLBenchRequest("OtherBenchmark", False))
