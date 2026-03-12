"""
Extract depth frames from ZED SVO files for RAIL episodes.
Saves per-episode depth as {output_dir}/{canonical_id}/{serial}/depth.blosc (float32, T x 180 x 320)
Compressed with Blosc zstd+bitshuffle. Also saves {serial}/intrinsics.npy

Usage:
    python data_processing/extract_svo_depth.py \
        --raw-dir /work/nvme/bgkz/droid_rail_raw \
        --output-dir /work/nvme/bgkz/droid_rail_depths \
        [--cameras wrist ext1]   # default: all
"""


import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple

os.environ.setdefault("ZED_SETTINGS_PATH",
    str(Path.home() / ".local" / "share" / "stereolabs" / "settings"))

import blosc
import cv2
import numpy as np
import pyzed.sl as sl

TARGET_H, TARGET_W = 180, 320



def extract_episode(ep_dir: Path, output_dir: Path, cameras: List[str]) -> Tuple[bool, str]:
    ep_id = ep_dir.name

    # Load metadata to get serial -> camera mapping
    meta_files = list(ep_dir.glob("metadata_*.json"))
    if not meta_files:
        return False, "no metadata file"
    meta = json.load(open(meta_files[0]))

    cam_map = {
        "wrist": meta["wrist_cam_serial"],
        "ext1":  meta["ext1_cam_serial"],
        "ext2":  meta["ext2_cam_serial"],
    }

    any_success = False
    for cam_name in cameras:
        serial = cam_map[cam_name]
        svo_path = ep_dir / "recordings" / "SVO" / f"{serial}.svo"
        if not svo_path.exists():
            print(f"  [{ep_id}] SVO not found: {svo_path}", file=sys.stderr)
            continue

        out_dir = output_dir / ep_id / serial
        done_file = out_dir / ".done"
        if done_file.exists():
            continue

        out_dir.mkdir(parents=True, exist_ok=True)

        init = sl.InitParameters()

        print(svo_path)
        init.set_from_svo_file(str(svo_path))
        init.svo_real_time_mode = False
        init.depth_mode = sl.DEPTH_MODE.ULTRA
        init.coordinate_units = sl.UNIT.METER
        init.depth_minimum_distance = 0.1
        init.depth_maximum_distance = 5.0

        cam = sl.Camera()

        status = cam.open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            print(f"SKIP [{ep_id}] {cam_name} ({serial}): {status} (code: {repr(status)})", file=sys.stderr)
            continue

        # Save intrinsics scaled to TARGET resolution
        cam_info = cam.get_camera_information()
        native_h = cam_info.camera_configuration.resolution.height
        native_w = cam_info.camera_configuration.resolution.width
        calib = cam_info.camera_configuration.calibration_parameters
        left = calib.left_cam
        scale_x = TARGET_W / native_w
        scale_y = TARGET_H / native_h
        intrinsics = np.array([
            left.fx * scale_x, left.fy * scale_y,
            left.cx * scale_x, left.cy * scale_y,
        ], dtype=np.float32)
        np.save(out_dir / "intrinsics.npy", intrinsics)  # [fx, fy, cx, cy] at 180x320

        # Extract depth frames into a single (T, H, W) array
        depth_mat = sl.Mat()
        runtime = sl.RuntimeParameters()
        frames = []
        while True:
            err = cam.grab(runtime)
            if err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                break
            if err != sl.ERROR_CODE.SUCCESS:
                print(f"  [{ep_id}] grab error at frame {len(frames)}: {err}", file=sys.stderr)
                break
            cam.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
            depth = depth_mat.get_data()  # float32 HxW, meters, nan for invalid
            frames.append(cv2.resize(depth, (TARGET_W, TARGET_H), interpolation=cv2.INTER_NEAREST))

        cam.close()

        all_depth = np.stack(frames)  # (T, H, W) float32
        compressed = blosc.compress_ptr(
            all_depth.ctypes.data, all_depth.size, all_depth.dtype.itemsize,
            clevel=5, cname='zstd', shuffle=blosc.BITSHUFFLE,
        )
        (out_dir / "depth.blosc").write_bytes(compressed)
        np.save(out_dir / "shape.npy", np.array(all_depth.shape, dtype=np.int32))  # (T, H, W)

        done_file.touch()
        print(f"  [{ep_id}] {cam_name} ({serial}): {len(frames)} frames saved")
        any_success = True

    return any_success, "ok" if any_success else "all cameras skipped"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--cameras", nargs="+", default=["wrist", "ext1", "ext2"],
                        choices=["wrist", "ext1", "ext2"])
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    args = parser.parse_args()

    ep_dirs = sorted([d for d in args.raw_dir.iterdir() if d.is_dir()])
    my_eps = ep_dirs[args.rank::args.world_size]
    print(f"Rank {args.rank}/{args.world_size}: processing {len(my_eps)}/{len(ep_dirs)} episodes")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    done, failed = 0, 0
    for ep_dir in my_eps:
        ok, msg = extract_episode(ep_dir, args.output_dir, args.cameras)
        if ok:
            done += 1
        else:
            failed += 1
            print(f"FAILED {ep_dir.name}: {msg}", file=sys.stderr)

    print(f"\nDone: {done} ok, {failed} failed")


if __name__ == "__main__":
    main()
