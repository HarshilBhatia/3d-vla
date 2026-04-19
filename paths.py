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
# Set this to the user name (default LUQMAN when unset, e.g. in sbatch)
ENV = os.getenv("USER_NAME") or "LUQMAN"



CONFIGS = {
    "LUQMAN": {
        "RAW_ROOT": "/home/lzaceria/mscv/3dvla/3d-vla/peract2_raw",
        "ZARR_ROOT": "/home/lzaceria/mscv/3dvla/3d-vla/Peract2_zarr",
        "USER_DATA": "/home/lzaceria/mscv/3dvla/3d-vla",
    },
    "HB": {
        "RAW_ROOT": "peract2_raw",
        "ZARR_ROOT": "Peract2_zarr",
        "USER_DATA": "",
    }
}

# ==============================================================================
# RESOLVED PATHS
# ==============================================================================
_env = ENV if ENV in CONFIGS else "LUQMAN"
RAW_ROOT = CONFIGS[_env]["RAW_ROOT"]
ZARR_ROOT = CONFIGS[_env]["ZARR_ROOT"]
USER_DATA = CONFIGS[_env]["USER_DATA"]

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        # Allow shell scripts to query paths: python paths.py RAW_ROOT
        print(globals().get(sys.argv[1], ""))

