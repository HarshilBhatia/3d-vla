"""
Shared constants for the orbital data collection pipeline.
"""

PERACT_TASKS = [
    "place_cups", "close_jar", "insert_onto_square_peg",
    "light_bulb_in", "meat_off_grill", "open_drawer",
    "place_shape_in_shape_sorter", "place_wine_at_rack_location",
    "push_buttons", "put_groceries_in_cupboard",
    "put_item_in_drawer", "put_money_in_safe", "reach_and_drag",
    "slide_block_to_color_target", "stack_blocks", "stack_cups",
    "sweep_to_dustpan_of_size", "turn_tap",
]

DEPTH_SCALE = 2 ** 24 - 1  # RGB-encoded depth scale (RLBench convention)

NCAM  = 3  # orbital_left, orbital_right, wrist
NHAND = 1  # single-arm Panda


def num2id(i: int) -> str:
    """Zero-pad a frame index to 4 digits."""
    return str(i).zfill(4)
