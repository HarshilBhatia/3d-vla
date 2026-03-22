# Environment Setup

## Prerequisites

- CUDA 12.8 (tested on NVIDIA A100). Adjust the torch install URL below if your CUDA version differs.
- [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html) (or conda/mamba)

## 1. Create the environment

```bash
micromamba create -n gr00t python=3.10 -c conda-forge -y
micromamba activate gr00t
```

## 2. Install PyTorch (CUDA 12.8)

```bash
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 torchcodec==0.4.0 \
    --index-url https://download.pytorch.org/whl/cu128
```

> For a different CUDA version replace `cu128` with e.g. `cu121` and adjust the index URL.

## 3. Install flash-attn

flash-attn must be installed **after** torch (it compiles against the installed torch CUDA headers).

```bash
pip install flash-attn==2.7.4.post1 --no-build-isolation
```

This takes several minutes to compile.

## 4. Install the project (editable)

From the repo root:

```bash
pip install -e ".[dev]"
```

This installs all remaining dependencies from `pyproject.toml` and registers the `gr00t` package in editable mode so code changes are reflected immediately.

## 5. Verify

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import flash_attn; print('flash_attn ok')"
python -c "import gr00t; print('gr00t ok')"
```

## Data paths (Delta + Anvil shared storage)

| Path | Contents |
|------|----------|
| `/work/hdd/bgkz/droid_raw_large_superset` | Raw DROID dataset (23.7k episodes) |
| `/work/nvme/bgkz/droid_multilab_raw` | Multilab subset raw episodes |
| `/work/nvme/bgkz/droid_multilab_depths` | Extracted depth frames + serial_map.json |
| `/work/nvme/bgkz/droid_multilab_cache_ext2` | Backbone feature shards (OXE_DROID_EXT2) |
| `/work/nvme/bgkz/droid_multilab_depth_cam2cam_ext2` | Depth shards — cam2cam extrinsics |
| `/work/nvme/bgkz/droid_multilab_depth_cam2cam_campos_ext2` | Depth shards — cam2cam + camera_positions_3d |
| `/work/nvme/bgkz/droid_annotations/cam2cam_extrinsics.json` | Stereo calibration (91k episodes) |
| `/work/hdd/bgkz/hbhatia1/` | Training checkpoints |

## SLURM account / partition (Delta)

```
--account=bgkz-delta-gpu   (GPU jobs)
--account=bgkz-delta-cpu   (CPU jobs)
--partition=gpuA100x4,gpuA100x8
```

Update these for other clusters (e.g. Anvil uses different account/partition names).

## Quick training launch

```bash
# Debug run (5 steps, 1 GPU)
sbatch scripts/slurm/debug_delta_m_train.slurm

# Full deltaM run (no camera positions)
sbatch scripts/slurm/train_multilab_delta_m.slurm

# Full deltaM run (with camera positions — requires campos depth shards)
sbatch scripts/slurm/train_multilab_delta_m_campos.slurm
```
