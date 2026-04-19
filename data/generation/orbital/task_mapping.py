"""
Task-to-camera-group mapping for orbital data generation.

18 PerAct tasks × 3 groups/task; cyclic assignment so each of 6 groups
appears in exactly 9 tasks.

  groups_for_task[i] = [G_{i%6+1}, G_{(i+1)%6+1}, G_{(i+2)%6+1}]

Saves task_group_mapping.json at the repo root.
"""

import json
import os

from data.generation.orbital.constants import PERACT_TASKS

N_GROUPS = 6


def build_mapping(tasks=None):
    """Return {task_name: ["G1", "G2", "G3"]} for all tasks."""
    if tasks is None:
        tasks = PERACT_TASKS
    mapping = {}
    for i, task in enumerate(tasks):
        mapping[task] = [
            "G{}".format((i + j) % N_GROUPS + 1)
            for j in range(3)
        ]
    return mapping


def verify_mapping(mapping):
    from collections import Counter
    counts = Counter(g for groups in mapping.values() for g in groups)
    print("[INFO] Group appearance counts:")
    for g in sorted(counts):
        print("  {}: {} tasks".format(g, counts[g]))
    assert all(v == 9 for v in counts.values()), \
        "Expected each group to appear in exactly 9 tasks!"
    print("[OK] Each group appears in exactly 9 tasks.")


def main():
    out_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "task_group_mapping.json",
    )

    mapping = build_mapping()
    verify_mapping(mapping)

    with open(out_path, "w") as f:
        json.dump(mapping, f, indent=2)
    print("[DONE] Saved mapping to {}".format(out_path))

    print("\nTask → Groups:")
    for task, groups in mapping.items():
        print("  {:45s} {}".format(task, groups))


if __name__ == "__main__":
    main()
