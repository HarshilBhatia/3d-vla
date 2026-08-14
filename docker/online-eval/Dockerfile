# Online evaluation image for 3D Flow Matching Actor (RLBench / CoppeliaSim).
#
# Design decisions:
#   * Base is Ubuntu 20.04 + CUDA runtime. CoppeliaSim_Edu_V4_1_0 ships Qt5 / glibc
#     binaries built for focal; newer bases (22.04/24.04) fail to load its Qt plugins.
#   * CoppeliaSim 4.1 is baked in at /opt/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04 (~470 MB
#     unpacked). It never changes, so caching it in a layer is strictly better than
#     re-downloading 156 MB per container start.
#   * PyRep and RLBench are baked in as NON-EDITABLE installs from the repo's own
#     local forks (build context), not from upstream git. Rationale:
#       - PyRep compiles a cffi extension (pyrep/backend/_sim_cffi*.so) against the
#         CoppeliaSim headers and requires COPPELIASIM_ROOT at *setup* time. Doing
#         that at container start costs ~1 min per run and needs a toolchain present.
#       - An editable install into a bind-mounted repo would write build artifacts
#         (.so, build/, *.egg-info) into the host checkout, mutating the user's tree.
#       - Non-editable install puts pyrep/ and rlbench/ in site-packages, so the
#         mounted repo's PyRep/ and RLBench/ directories are inert. Note
#         online_evaluation_rlbench/get_stored_demos.py *appends* <repo>/RLBench to
#         sys.path, so the installed copy still wins.
#     Only the two forks are copied in; the rest of the repo (data, logs, ckpts) is
#     bind-mounted at runtime so code edits need no rebuild.
#   * The python env is uv-managed: uv-provided CPython 3.10 + a venv at /opt/venv.
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ---------------------------------------------------------------------------
# System packages: CoppeliaSim's Qt5/X11/GL runtime + xvfb + build toolchain
# (kept in the final image so PyRep's cffi extension can be rebuilt in-place
# if a user ever needs to, and because gcc is tiny next to torch/CUDA).
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        xz-utils \
        # PID-1 reaper: xvfb-run blocks forever if it runs as PID 1 (it waits on
        # signals that the kernel does not deliver to PID 1 by default).
        tini \
        # headless display
        xvfb \
        x11-utils \
        x11-xserver-utils \
        # GL / GLU
        libgl1-mesa-glx \
        libgl1-mesa-dri \
        libglu1-mesa \
        libegl1 \
        libgomp1 \
        # Qt5 xcb platform plugin dependencies (CoppeliaSim bundles Qt itself)
        libxcb1 \
        libxcb-icccm4 \
        libxcb-image0 \
        libxcb-keysyms1 \
        libxcb-randr0 \
        libxcb-render0 \
        libxcb-render-util0 \
        libxcb-shape0 \
        libxcb-shm0 \
        libxcb-sync1 \
        libxcb-util1 \
        libxcb-xfixes0 \
        libxcb-xinerama0 \
        libxcb-xkb1 \
        libxkbcommon0 \
        libxkbcommon-x11-0 \
        libx11-6 \
        libx11-xcb1 \
        libxext6 \
        libxrender1 \
        libxi6 \
        libsm6 \
        libice6 \
        libdbus-1-3 \
        libfontconfig1 \
        libfreetype6 \
        # ffmpeg libs CoppeliaSim links for video recording
        libavcodec58 \
        libavformat58 \
        libswscale5 \
        # misc runtime
        libusb-1.0-0 \
        libglib2.0-0 \
        libsndio7.0 \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# CoppeliaSim 4.1.0 (Edu, Ubuntu 20.04)
# coppeliarobotics.com 301-redirects to downloads.coppeliarobotics.com; we hit
# the downloads host directly and fall back to the legacy path.
# ---------------------------------------------------------------------------
ENV COPPELIASIM_ROOT=/opt/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04
ARG COPPELIASIM_TARBALL=CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
RUN set -eux; \
    cd /opt; \
    curl -fSL --retry 5 --retry-delay 5 -o cs.tar.xz \
        "https://downloads.coppeliarobotics.com/V4_1_0/${COPPELIASIM_TARBALL}" \
      || curl -fSL --retry 5 --retry-delay 5 -o cs.tar.xz \
        "https://www.coppeliarobotics.com/files/V4_1_0/${COPPELIASIM_TARBALL}"; \
    tar -xf cs.tar.xz; \
    rm cs.tar.xz; \
    test -f "${COPPELIASIM_ROOT}/libcoppeliaSim.so"; \
    test -x "${COPPELIASIM_ROOT}/coppeliaSim"; \
    # PyRep's setup.py normally appends this; write it up front so the license
    # gate is satisfied even if PyRep is reinstalled later.
    grep -q allowOldEduRelease "${COPPELIASIM_ROOT}/system/usrset.txt" \
      || printf '\nallowOldEduRelease=7501\n' >> "${COPPELIASIM_ROOT}/system/usrset.txt"

ENV LD_LIBRARY_PATH=${COPPELIASIM_ROOT}:${LD_LIBRARY_PATH} \
    QT_QPA_PLATFORM_PLUGIN_PATH=${COPPELIASIM_ROOT}

# ---------------------------------------------------------------------------
# uv + uv-managed CPython 3.10 + venv at /opt/venv
# ---------------------------------------------------------------------------
ENV UV_INSTALL_DIR=/usr/local/bin \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    UV_LINK_MODE=copy \
    VIRTUAL_ENV=/opt/venv
RUN curl -fsSL https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL=/usr/local/bin sh \
    && uv --version
RUN uv python install 3.10 \
    && uv venv --python 3.10 "${VIRTUAL_ENV}"
ENV PATH=${VIRTUAL_ENV}/bin:${PATH}

# ---------------------------------------------------------------------------
# Python deps. Torch first (own index), then everything else from PyPI.
#
# torch >= 2.5 is required: current transformers silently disables its PyTorch
# backend ("Disabling PyTorch because PyTorch >= 2.5 is required") on older
# torch, which would break every from_pretrained() call in modeling/encoder/.
# The cu124 wheels bundle their own CUDA/cuDNN libraries, so they run fine on
# the cu121 focal base image (only the host driver has to be new enough).
# ---------------------------------------------------------------------------
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu124
ARG TORCH_VERSION=2.6.0
ARG TORCHVISION_VERSION=0.21.0
RUN uv pip install --python "${VIRTUAL_ENV}/bin/python" \
        --index-url "${TORCH_INDEX}" \
        "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}"

COPY docker/online-eval/requirements.txt /tmp/requirements.txt
RUN uv pip install --python "${VIRTUAL_ENV}/bin/python" -r /tmp/requirements.txt

# ---------------------------------------------------------------------------
# PyRep + RLBench from the repo's local forks (see header for rationale).
# ---------------------------------------------------------------------------
COPY PyRep /opt/src/PyRep
COPY RLBench /opt/src/RLBench
RUN uv pip install --python "${VIRTUAL_ENV}/bin/python" --no-build-isolation \
        cffi setuptools wheel \
    && uv pip install --python "${VIRTUAL_ENV}/bin/python" --no-build-isolation /opt/src/PyRep \
    # RLBench's setup.py lists `packages` explicitly and omits
    # rlbench.action_modes, so a regular (copying) install produces a broken
    # tree. Install it editable against the *baked* /opt/src/RLBench copy: the
    # .pth entry exposes the whole source tree, subpackages and .ttm/.obj assets
    # alike, and nothing is written into the user's bind-mounted checkout.
    && uv pip install --python "${VIRTUAL_ENV}/bin/python" --no-deps -e /opt/src/RLBench \
    # Sanity: pyrep's cffi extension and RLBench's task assets must import from
    # the image, independent of any bind-mounted repo.
    && cd / && python -c "import pyrep, rlbench, os; \
import pyrep.backend._sim_cffi; \
from rlbench.action_modes.action_mode import ActionMode; \
print('pyrep', pyrep.__file__); print('rlbench', rlbench.__file__); \
assert 'site-packages' in pyrep.__file__; \
assert rlbench.__file__.startswith('/opt/src/RLBench'); \
assert os.path.isfile(os.path.join(os.path.dirname(rlbench.__file__), 'task_design.ttt'))"

# ---------------------------------------------------------------------------
# Runtime environment for the mounted repo.
# QT_QPA_PLATFORM is deliberately NOT set: CoppeliaSim must use the default xcb
# plugin against the Xvfb display. Setting it to "offscreen" breaks rendering.
# ---------------------------------------------------------------------------
ENV REPO_DIR=/workspace/3d_flowmatch_actor \
    PYTHONPATH=/workspace/3d_flowmatch_actor \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    XDG_RUNTIME_DIR=/tmp/runtime-root \
    HF_HOME=/workspace/.cache/huggingface \
    # Keeps tini reaping correctly (and quiet) even when the caller also passes
    # `docker run --init`, which makes docker-init PID 1 instead of tini.
    TINI_SUBREAPER=1
RUN mkdir -p /workspace/3d_flowmatch_actor /tmp/runtime-root && chmod 700 /tmp/runtime-root
VOLUME ["/workspace/3d_flowmatch_actor"]
WORKDIR /workspace/3d_flowmatch_actor

COPY docker/online-eval/entrypoint.sh /usr/local/bin/entrypoint.sh
COPY docker/online-eval/smoke_test.sh /usr/local/bin/3dfa-smoke-test
RUN chmod +x /usr/local/bin/entrypoint.sh /usr/local/bin/3dfa-smoke-test
# tini as PID 1 so xvfb-run (and CoppeliaSim's child processes) get proper
# signal handling and reaping; without it `docker run ... xvfb-run ...` hangs.
ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/entrypoint.sh"]
CMD ["bash"]
