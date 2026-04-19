"""
Create task-to-camera-group mapping for orbital data generation.

18 PerAct tasks × 3 groups/task; cyclic assignment so each of 6 groups
appears in exactly 9 tasks.

  groups_for_task[i] = [G_{i%6+1}, G_{(i+1)%6+1}, G_{(i+2)%6+1}]

Saves task_group_mapping.json: {task_name: ["G1", "G2", ...], ...}
"""
import json
import os
import sys

PERACT_TASKS = [
    "place_cups", "close_jar", "insert_onto_square_peg",
    "light_bulb_in", "meat_off_grill", "open_drawer",
    "place_shape_in_shape_sorter", "place_wine_at_rack_location",
    "push_buttons", "put_groceries_in_cupboard",
    "put_item_in_drawer", "put_money_in_safe", "reach_and_drag",
    "slide_block_to_color_target", "stack_blocks", "stack_cups",
    "sweep_to_dustpan_of_size", "turn_tap",
]

N_GROUPS = 6


def build_mapping(tasks):
    mapping = {}
    for i, task in enumerate(tasks):
        groups = [
            "G{}".format((i + j) % N_GROUPS + 1)
            for j in range(3)
        ]
        mapping[task] = groups
    return mapping


def verify_mapping(mapping):
    from collections import Counter
    counts = Counter()
    for groups in mapping.values():
        for g in groups:
            counts[g] += 1
    print("[INFO] Group appearance counts:")
    for g in sorted(counts):
        print("  {}: {} tasks".format(g, counts[g]))
    assert all(v == 9 for v in counts.values()), \
        "Expected each group to appear in exactly 9 tasks!"
    print("[OK] Each group appears in exactly 9 tasks.")


def main():
    out_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "task_group_mapping.json",
    )

    mapping = build_mapping(PERACT_TASKS)
    verify_mapping(mapping)

    with open(out_path, "w") as f:
        json.dump(mapping, f, indent=2)
    print("[DONE] Saved mapping to {}".format(out_path))

    # Print summary
    print("\nTask → Groups:")
    for task, groups in mapping.items():
        print("  {:45s} {}".format(task, groups))


if __name__ == "__main__":
    main()
