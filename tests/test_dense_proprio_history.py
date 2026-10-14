import numpy as np
import pytest
import zarr

from datasets.base import BaseDataset

ROOT = "/grogu/datasets/hbhatia/3dfa_dense/handover_chunk8/val.zarr"


@pytest.mark.skipif(not __import__("os").path.exists(ROOT), reason="dense handover zarr not mounted")
@pytest.mark.parametrize("k", [3, 5])
def test_chunk_zarr_proprio_is_the_stored_stride_window(k):
    ds = BaseDataset.__new__(BaseDataset)
    ds.annos = zarr.open(ROOT, "r")
    ds.chunk_size, ds.visual_num_history, ds._demo_ids = 1, k, None
    i = 500
    got = ds._get_proprioception_hist(i).numpy()
    np.testing.assert_array_equal(got[0], np.asarray(ds.annos["proprioception"][i]))
