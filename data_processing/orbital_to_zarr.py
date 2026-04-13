# Re-export from data_generation.orbital.to_zarr for backward compatibility.
# Run directly with: python -m data_generation.orbital.to_zarr  (or via this file)
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data_generation.orbital.to_zarr import (
    process_episode,
    load_rgb,
    load_depth_metres,
    load_extrinsics_from_misc,
    load_intrinsics_from_misc,
    load_orbital_extrinsics,
    get_group_id,
    main,
)

__all__ = [
    "process_episode",
    "load_rgb",
    "load_depth_metres",
    "load_extrinsics_from_misc",
    "load_intrinsics_from_misc",
    "load_orbital_extrinsics",
    "get_group_id",
    "main",
]

if __name__ == "__main__":
    main()
