# PerAct2 (bimanual) variant of the 3DFA online-eval image.
#
# The base image bakes the unimanual PerAct forks of PyRep/RLBench, which have
# no dual_panda robot and no rlbench.bimanual_tasks. This layer replaces both
# with the markusgrotz PerAct2 forks (same CoppeliaSim 4.1 requirement), so the
# orbital collector's PERACT2_PROFILE can run.
FROM 3dfa-online-eval:latest

COPY PyRep /opt/src/PyRep2
COPY RLBench /opt/src/RLBench2

RUN uv pip uninstall --python /opt/venv/bin/python pyrep rlbench || true \
    && uv pip install --python /opt/venv/bin/python --no-build-isolation /opt/src/PyRep2 \
    && uv pip install --python /opt/venv/bin/python --no-deps -e /opt/src/RLBench2 \
    && uv pip install --python /opt/venv/bin/python natsort pyquaternion \
    && cd / && python -c "import pyrep, rlbench, os; \
import pyrep.backend._sim_cffi; \
from pyrep.robots.arms.dual_panda import PandaLeft, PandaRight; \
from rlbench.action_modes.action_mode import BimanualMoveArmThenGripper; \
from rlbench.action_modes.arm_action_modes import BimanualEndEffectorPoseViaPlanning; \
from rlbench.action_modes.gripper_action_modes import BimanualDiscrete; \
import rlbench.bimanual_tasks.bimanual_push_box as _t; \
print('pyrep', pyrep.__file__); print('rlbench', rlbench.__file__); \
assert 'site-packages' in pyrep.__file__; \
assert rlbench.__file__.startswith('/opt/src/RLBench2'); \
assert os.path.isfile(os.path.join(os.path.dirname(rlbench.__file__), 'task_design_bimanual.ttt')); \
assert os.path.isfile(os.path.join(os.path.dirname(rlbench.__file__), 'robot_ttms', 'dual_panda.ttm'))"
