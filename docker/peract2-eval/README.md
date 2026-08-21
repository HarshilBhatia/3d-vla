# PerAct2 (bimanual) eval image

`docker/online-eval/` builds the unimanual eval image. **Every bimanual eval
campaign in `docs/status/experiments.md` ran on the image built from *this*
directory**, not on that one. This is the recorded source for it.

## The chain

```
nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu20.04
  └─ docker/online-eval/Dockerfile   ->  3dfa-online-eval:latest
        (CoppeliaSim 4.1.0 Edu, uv CPython 3.10 venv at /opt/venv,
         the repo-root PyRep/ + RLBench/ unimanual PerAct forks)
       └─ docker/peract2-eval/Dockerfile  ->  3dfa-peract2:latest
             (uninstalls pyrep+rlbench, reinstalls the markusgrotz
              PerAct2 forks from /opt/src/{PyRep2,RLBench2})
```

The base image's forks are the **unimanual** PerAct lineage — no `dual_panda`
robot, no `rlbench.bimanual_tasks`, no `task_design_bimanual.ttt`. This layer
swaps them for the PerAct2 forks, which keep the same CoppeliaSim 4.1
requirement, so the base layer's simulator and venv are reused unchanged. The
`RUN` step ends in an import assertion block that fails the build if
`dual_panda`, the bimanual action modes, `task_design_bimanual.ttt`, or
`robot_ttms/dual_panda.ttm` are missing — a silent fallback to the unimanual
forks cannot get past it.

## Fork sources and exact commits

Both trees are **byte-identical to the public fork heads**; there are no local
modifications and therefore no patch files to apply.

| tree | source | commit | subject |
|---|---|---|---|
| `PyRep2` | https://github.com/markusgrotz/PyRep | `b8bd1d7a3182adcd570d001649c0849047ebf197` (`main`) | `move colorize function to robot_component class` |
| `RLBench2` | https://github.com/markusgrotz/RLBench | `8af748c51287989294e00c9c670e3330a0e35ed5` (`main`) | `add script to visualize trajectories` |

Verified 2026-08-21: the working trees used for the image build had no `.git`
metadata (they were exported copies), so provenance was established by hashing
each tree into the corresponding bare public clone and searching all reachable
commits for a matching tree object. Both matched exactly, and a `diff -r` against
a fresh checkout of each commit reported zero differences. Both commits are
reachable on the public `main` branches, so no fork tarball needs to be
preserved.

For contrast, the base image's unimanual forks (repo-root `PyRep/`, `RLBench/`,
both gitignored) are:

| tree | source | commit |
|---|---|---|
| `PyRep` | https://github.com/stepjam/PyRep | `8f420be8064b1970aae18a9cfbc978dfb15747ef` |
| `RLBench` | https://github.com/MohitShridhar/RLBench | `ad991951bc53e4f3b73b803a75cf4b7d55295cf7` |

## Rebuild

The `COPY` paths are relative to the build context, which must contain `PyRep/`
and `RLBench/` checked out at the commits above:

```bash
# 1. the base layer, from the repo root
bash docker/online-eval/build.sh                    # -> 3dfa-online-eval:latest

# 2. the PerAct2 layer, from a context holding the two forks
mkdir -p /tmp/peract2_build && cd /tmp/peract2_build
git clone https://github.com/markusgrotz/PyRep.git PyRep
git -C PyRep checkout b8bd1d7a3182adcd570d001649c0849047ebf197
git clone https://github.com/markusgrotz/RLBench.git RLBench
git -C RLBench checkout 8af748c51287989294e00c9c670e3330a0e35ed5
cp /path/to/3d_flowmatch_actor/docker/peract2-eval/Dockerfile .
docker build -t 3dfa-peract2:latest .
```

Run it with `docker/online-eval/run.sh`, overriding the image:

```bash
IMAGE=3dfa-peract2 bash docker/online-eval/run.sh --smoke-test
```

A rebuild is **not** bit-identical to the campaign image (apt and PyPI indexes
move). Prefer the ECR copy below when the goal is to reproduce a number rather
than to change the environment.

## The exact campaign image

```
241533154612.dkr.ecr.us-east-1.amazonaws.com/rfm-h-eval-job:hb-3dfa-peract2-20260811
```

| field | value |
|---|---|
| manifest digest | `sha256:fa7b2dd3d0e719ec067b5f08f89e748c95e04e12cee5170e17090f8550690b88` |
| local image ID | `sha256:e655420f5468a5159001b76059d84cf7d5a67d6088535252bfdadf756db69dec` |
| built | 2026-08-11T08:05:16Z |
| pushed | 2026-08-11T17:28:55Z |
| size | 20.3 GB local / 10.83 GB compressed in ECR |
| base image ID | `sha256:ee4062d0b436…` (`3dfa-online-eval:latest`) |

```bash
AWS_PROFILE=far-compute aws ecr get-login-password --region us-east-1 \
  | docker login --username AWS --password-stdin 241533154612.dkr.ecr.us-east-1.amazonaws.com
docker pull 241533154612.dkr.ecr.us-east-1.amazonaws.com/rfm-h-eval-job@sha256:fa7b2dd3d0e719ec067b5f08f89e748c95e04e12cee5170e17090f8550690b88
```

Pull **by digest**, not by tag — the tag is mutable. Some older notes give the
registry account as `913524929094`; the tag resolves under `241533154612`, which
is where `docker inspect` on the campaign image points and the only account the
campaign credentials can read.

## Gotchas

- Every gotcha in `docker/online-eval/README.md` still applies (unset
  `QT_QPA_PLATFORM`, `--init` for `xvfb-run`, `numpy<2`, `zarr<3`, `torch>=2.5`,
  `open3d` imported before CoppeliaSim). This layer changes only which
  PyRep/RLBench is installed.
- `pyrep` lands in `site-packages` (non-editable) but `rlbench` is an **editable**
  install off `/opt/src/RLBench2`. That is deliberate: the RLBench `setup.py`
  `packages` list omits subpackages such as `rlbench.action_modes`, so a copying
  install produces a broken tree. The assertions check
  `rlbench.__file__.startswith('/opt/src/RLBench2')` for exactly this reason.
- The layer also installs `natsort` and `pyquaternion`, which the PerAct2 forks
  need and the base image's requirements do not carry.
- `uv pip uninstall … || true` tolerates a base image where the packages are
  already absent; the import assertions are what actually gate correctness.
