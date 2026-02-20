import os
import sys

# Add RLBench and PyRep to path so they are available globally
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.join(ROOT_DIR, 'RLBench') not in sys.path:
    sys.path.append(os.path.join(ROOT_DIR, 'RLBench'))
if os.path.join(ROOT_DIR, 'PyRep') not in sys.path:
    sys.path.append(os.path.join(ROOT_DIR, 'PyRep'))

# ==============================================================================
# CONFIGURATION TOGGLE
# ==============================================================================
# Set this to the user name
ENV = os.getenv("USER_NAME") 



CONFIGS = {
    "LUQMAN": {
        "RAW_ROOT": "/home/lzaceria/mscv/3dvla/3d-vla/peract2_test",
        "ZARR_ROOT": "/home/lzaceria/mscv/3dvla/3d-vla/Peract2_zarr",
        "USER_DATA": "/home/lzaceria/mscv/3dvla/3d-vla",
    },
    "HB": {
        "RAW_ROOT": "peract2_raw",
        "ZARR_ROOT": "Peract2_zarr",
        "USER_DATA": "",
    },
    "ORIGINAL": {
        "RAW_ROOT": "/data/group_data/katefgroup/VLA/peract2_raw_squash",
        "ZARR_ROOT": "/data/user_data/ngkanats/zarr_datasets/Peract2_zarr",
        "USER_DATA": "/data/user_data/ngkanats",
    }
}

# ==============================================================================
# RESOLVED PATHS
# ==============================================================================
RAW_ROOT = CONFIGS[ENV]["RAW_ROOT"]
ZARR_ROOT = CONFIGS[ENV]["ZARR_ROOT"]
USER_DATA = CONFIGS[ENV]["USER_DATA"]

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        # Allow shell scripts to query paths: python paths.py RAW_ROOT
        print(globals().get(sys.argv[1], ""))

