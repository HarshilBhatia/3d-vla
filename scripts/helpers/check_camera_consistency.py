import os
import pickle
import numpy as np
import sys

class Stub:
    def __init__(self, *args, **kwargs):
        pass
    def __setstate__(self, state):
        self.__dict__.update(state)
    def __getattr__(self, name):
        return self.__dict__.get(name, Stub())
    def __getitem__(self, key):
        # Handle demo[k] where demo might have _observations
        if hasattr(self, '_observations'):
            return self._observations[key]
        if isinstance(key, int):
             # If it's acting like a list but data is elsewhere
             return Stub()
        return self.__dict__.get(key, Stub())

class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'list': return list
        if name == 'dict': return dict
        try:
            return super().find_class(module, name)
        except (AttributeError, ModuleNotFoundError, ImportError):
            return Stub

RAW_ROOT = "peract2_raw"

TASK = "bimanual_lift_tray"
ROOT = RAW_ROOT
SPLIT = "train"

CAMERAS = ["front", "wrist_left", "wrist_right"]
TOL = 1e-5

# Try different path structures
possible_task_folders = [
    f"{ROOT}/{SPLIT}/{TASK}/all_variations/episodes",
    f"{ROOT}/{TASK}/all_variations/episodes",
    f"{ROOT}/{TASK}/variation0/episodes",
]

task_folder = None
for folder in possible_task_folders:
    if os.path.exists(folder):
        task_folder = folder
        break

if task_folder is None:
    raise FileNotFoundError(f"Could not find episodes for task {TASK} in {ROOT}")

episodes = sorted(os.listdir(task_folder))

def load_camera_params(ep):
    with open(f"{task_folder}/{ep}/low_dim_obs.pkl", "rb") as f:
        demo = CustomUnpickler(f).load()
    
    # take first frame only
    k = 0
    obs = demo[k]
    return {
        cam: {
            "extrinsics": obs.misc[f"{cam}_camera_extrinsics"],
            "intrinsics": obs.misc[f"{cam}_camera_intrinsics"]
        }
        for cam in CAMERAS
    }

ref = load_camera_params(episodes[0])

for ep in episodes[1:]:
    cur = load_camera_params(ep)
    for cam in CAMERAS:
        # Check Extrinsics
        ext_diff = np.abs(ref[cam]["extrinsics"] - cur[cam]["extrinsics"]).max()
        if ext_diff > TOL:
            if cam == "front":
                raise ValueError(
                    f"[FAIL] Static camera {cam} EXTRINSICS mismatch in episode {ep}, diff={ext_diff}"
                )

        # Check Intrinsics (Should be static for ALL cameras)
        int_diff = np.abs(ref[cam]["intrinsics"] - cur[cam]["intrinsics"]).max()
        if int_diff > TOL:
            raise ValueError(
                f"[FAIL] Camera {cam} INTRINSICS mismatch in episode {ep}, diff={int_diff}"
            )

print("[PASS] All camera intrinsics and static extrinsics are consistent across episodes.")

