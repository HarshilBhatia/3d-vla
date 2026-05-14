"""
Shared constants for the orbital data collection pipeline.

Robot configurations are encoded in RobotProfile instances (PERACT_PROFILE,
PERACT2_PROFILE). Pass the appropriate profile to every pipeline function;
the default is always PERACT_PROFILE so existing callers are unaffected.
"""

from dataclasses import dataclass, field
from typing import Callable, List

import numpy as np

# ---------------------------------------------------------------------------
# Task lists
# ---------------------------------------------------------------------------

PERACT_TASKS = [
    "place_cups", "close_jar", "insert_onto_square_peg",
    "light_bulb_in", "meat_off_grill", "open_drawer",
    "place_shape_in_shape_sorter", "place_wine_at_rack_location",
    "push_buttons", "put_groceries_in_cupboard",
    "put_item_in_drawer", "put_money_in_safe", "reach_and_drag",
    "slide_block_to_color_target", "stack_blocks", "stack_cups",
    "sweep_to_dustpan_of_size", "turn_tap",
]

PERACT2_TASKS = [
    "bimanual_push_box", "bimanual_lift_ball", "bimanual_dual_push_buttons",
    "bimanual_pick_plate", "bimanual_put_item_in_drawer",
    "bimanual_put_bottle_in_fridge", "bimanual_handover_item",
    "bimanual_pick_laptop", "bimanual_straighten_rope",
    "bimanual_sweep_to_dustpan", "bimanual_lift_tray",
    "bimanual_handover_item_easy", "bimanual_take_tray_out_of_oven",
]

DEPTH_SCALE = 2 ** 24 - 1  # RGB-encoded depth scale (RLBench convention)

NCAM  = 3  # orbital_left, orbital_right, wrist  (PerAct default)
NHAND = 1  # single-arm Panda  (PerAct default)


def num2id(i: int) -> str:
    """Zero-pad a frame index to 4 digits."""
    return str(i).zfill(4)


# ---------------------------------------------------------------------------
# EEF / joint state extractors
# ---------------------------------------------------------------------------

def _eef_unimanual(obs) -> np.ndarray:
    return np.concatenate([obs.gripper_pose, [obs.gripper_open]]).astype(np.float32)


def _eef_bimanual(obs) -> np.ndarray:
    return np.concatenate([
        obs.left.gripper_pose,  [obs.left.gripper_open],
        obs.right.gripper_pose, [obs.right.gripper_open],
    ]).astype(np.float32)


def _joints_unimanual(obs) -> np.ndarray:
    return np.concatenate([obs.joint_positions, [obs.gripper_open]]).astype(np.float32)


def _joints_bimanual(obs) -> np.ndarray:
    return np.concatenate([
        obs.left.joint_positions,  [obs.left.gripper_open],
        obs.right.joint_positions, [obs.right.gripper_open],
    ]).astype(np.float32)


# ---------------------------------------------------------------------------
# Task loaders
# ---------------------------------------------------------------------------

def _task_loader_unimanual(task_name: str):
    from rlbench.backend.utils import task_file_to_task_class
    return task_file_to_task_class(task_name)


def _task_loader_bimanual(task_name: str):
    import importlib
    name = task_name.replace(".py", "")
    class_name = "".join(w[0].upper() + w[1:] for w in name.split("_"))
    mod = importlib.import_module("rlbench.bimanual_tasks.%s" % name)
    return getattr(mod, class_name)


# ---------------------------------------------------------------------------
# Action mode factories
# ---------------------------------------------------------------------------

def _action_mode_unimanual():
    from rlbench.action_modes.action_mode import MoveArmThenGripper
    from rlbench.action_modes.arm_action_modes import JointVelocity
    from rlbench.action_modes.gripper_action_modes import Discrete
    return MoveArmThenGripper(JointVelocity(), Discrete())


def _action_mode_bimanual():
    from rlbench.action_modes.action_mode import BimanualMoveArmThenGripper
    from rlbench.action_modes.arm_action_modes import BimanualEndEffectorPoseViaPlanning
    from rlbench.action_modes.gripper_action_modes import BimanualDiscrete
    return BimanualMoveArmThenGripper(
        BimanualEndEffectorPoseViaPlanning(), BimanualDiscrete()
    )


# ---------------------------------------------------------------------------
# RobotProfile
# ---------------------------------------------------------------------------

@dataclass
class RobotProfile:
    """All robot-specific constants and callables for the orbital pipeline.

    Pass an instance to every pipeline function.  Two ready-made profiles are
    provided: PERACT_PROFILE (single-arm panda) and PERACT2_PROFILE (dual_panda).
    """
    name: str                          # human-readable label
    robot_setup: str                   # RLBench Environment robot_setup arg
    nhand: int                         # 1 or 2
    wrist_cameras: List[str]           # e.g. ["wrist"] or ["wrist_left","wrist_right"]
    task_list: List[str]
    bimanual: bool                     # forwarded to keypoint_discovery()
    eef_fn: Callable = field(repr=False)      # obs -> (nhand*8,) float32
    joints_fn: Callable = field(repr=False)   # obs -> (nhand*8,) float32
    task_loader: Callable = field(repr=False) # task_name -> task_class
    make_action_mode: Callable = field(repr=False)  # () -> action_mode


PERACT_PROFILE = RobotProfile(
    name="panda",
    robot_setup="panda",
    nhand=1,
    wrist_cameras=["wrist"],
    task_list=PERACT_TASKS,
    bimanual=False,
    eef_fn=_eef_unimanual,
    joints_fn=_joints_unimanual,
    task_loader=_task_loader_unimanual,
    make_action_mode=_action_mode_unimanual,
)

PERACT2_PROFILE = RobotProfile(
    name="dual_panda",
    robot_setup="dual_panda",
    nhand=2,
    wrist_cameras=["wrist_left", "wrist_right"],
    task_list=PERACT2_TASKS,
    bimanual=True,
    eef_fn=_eef_bimanual,
    joints_fn=_joints_bimanual,
    task_loader=_task_loader_bimanual,
    make_action_mode=_action_mode_bimanual,
)
