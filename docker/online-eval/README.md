# Online-eval container (RLBench / CoppeliaSim)

Docker image for running `online_evaluation_rlbench/evaluate_policy.py` headlessly.
Everything the simulator needs is baked into the image; the repo itself is
bind-mounted so code edits require no rebuild.

## Build

```bash
bash docker/online-eval/build.sh            # -> 3dfa-online-eval:latest
IMAGE=myrepo/3dfa-eval TAG=v1 bash docker/online-eval/build.sh
```

## Verify

```bash
bash docker/online-eval/run.sh --smoke-test
```

Checks CoppeliaSim, `from pyrep import PyRep`, `from rlbench.environment import
Environment`, torch/CUDA, and the version-sensitive deps (numpy 1.x, zarr 2.x,
transformers-with-torch-backend).

## Run an eval

```bash
DATA_DIRS=/grogu/user/harshilb:/grogu/datasets/hbhatia \
bash docker/online-eval/run.sh -- \
    xvfb-run -a --server-args="-screen 0 1280x1024x24 +extension GLX +render" \
    python online_evaluation_rlbench/evaluate_policy.py \
        dataset=OrbitalWrist bimanual=false task=open_drawer \
        checkpoint=/grogu/user/harshilb/train_logs/Orbital/my_run/last.pth \
        data_dir=/grogu/user/harshilb/orbital_rollouts \
        val_instructions=instructions/peract/instructions.json \
        cameras_file=instructions/orbital_cameras_grouped.json \
        task_group_mapping_file=instructions/task_group_mapping_subset.json \
        output_file=eval_logs/my_run/open_drawer.json \
        headless=true max_tries=1
```

`DATA_DIRS` is a colon-separated list of host paths bind-mounted read-only at the
same path inside the container, so checkpoint/dataset paths can be copied
verbatim from the SLURM scripts.

## What is in the image, and why

| Component | Where | Rationale |
|---|---|---|
| CoppeliaSim 4.1.0 Edu (Ubuntu 20.04) | `/opt/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04`, baked | 156 MB download, never changes |
| PyRep (repo fork) | installed into `/opt/venv`, baked | compiles a cffi extension against CoppeliaSim headers; doing it per-run is slow and would write `.so` files into the host checkout |
| RLBench (repo fork) | editable install off `/opt/src/RLBench`, baked | its `setup.py` has a hardcoded `packages` list that omits `rlbench.action_modes`, so a copying install is broken; editable off the *baked* copy avoids touching the mount |
| python 3.10 + deps | uv venv at `/opt/venv` | uv-managed CPython, deps pinned in `requirements.txt` |
| repo source | bind-mounted at `/workspace/3d_flowmatch_actor` | edit-and-rerun without a rebuild |

The container never writes into the mounted `PyRep/` or `RLBench/` directories —
they are shadowed by the installed copies. `online_evaluation_rlbench/get_stored_demos.py`
*appends* `<repo>/RLBench` to `sys.path`, so the installed copy still wins.

## Gotchas

- **`QT_QPA_PLATFORM` must stay unset.** Setting it to `offscreen` breaks
  CoppeliaSim's renderer; the entrypoint explicitly unsets it and relies on the
  bundled `xcb` plugin talking to Xvfb.
- **`xvfb-run` hangs if it runs as PID 1.** This looks exactly like a stuck eval
  (Xvfb up, no output, forever). The image therefore uses `tini` as its
  entrypoint and `run.sh` also passes `--init`. If you write your own
  `docker run`, keep `--init`.
- **`xvfb-run` must wrap the python process**, not the other way round; nesting
  it inside a script that later re-execs will lose `DISPLAY`.
- **`numpy` must stay <2 and `zarr` <3.** `datasets/utils.py` uses
  `zarr.storage.DirectoryStore` / `zarr.LRUStoreCache`, both removed in zarr 3,
  and the PyRep/RLBench-era code assumes numpy 1.x scalar behaviour.
- **`torch` must be >=2.5.** Older torch makes `transformers` silently disable
  its PyTorch backend, which breaks every `from_pretrained()` in
  `modeling/encoder/`. The smoke test asserts `transformers.is_torch_available()`.
- **`open3d` is imported before CoppeliaSim on purpose** (see the "DON'T DELETE
  THIS!" comments in `online_evaluation_rlbench/utils_with_*.py`) — it must load
  its GL symbols before CoppeliaSim's bundled libraries do.
- **Software rendering is enough for these tests** but slow; the mesa DRI
  drivers are installed so eval also works on a CPU-only host. With `--gpus all`
  CoppeliaSim still renders via its own GL path, not CUDA.
- **`openpi_client` is not installed** (not on PyPI). Only needed for
  `evaluate_policy_external.py`; see the note at the bottom of
  `requirements.txt`.
