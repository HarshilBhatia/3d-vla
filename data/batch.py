"""Public batch and action-shape utilities.

These helpers are intentionally independent of training orchestration.  They
are used by dataset loaders, training, and offline evaluation, so their public
home is the data domain rather than ``utils.trainers``.
"""

from __future__ import annotations

import torch


def base_collate_fn(batch):
    """Collate the RLBench trajectory dictionaries used by the loaders.

    String/list fields are flattened while tensor fields are concatenated.
    ``None`` tensor fields remain ``None``.  This preserves the historical
    collator's output and performance characteristics.
    """
    output = {}
    list_keys = ["task", "instr", "variation"]
    for key in list_keys:
        if key not in batch[0]:
            continue
        output[key] = []
        for item in batch:
            output[key].extend(item[key])

    output.update({
        key: (
            torch.cat([item[key] for item in batch])
            if batch[0][key] is not None else None
        )
        for key in batch[0]
        if key not in list_keys
    })
    return output


def actions_collate_fn(batch):
    """Collate action-only dictionaries."""
    return {"action": torch.cat([item["action"] for item in batch])}


def relative_to_absolute(action, proprio):
    """Convert relative xyz/Euler action deltas into absolute actions.

    ``action`` has shape ``(B, T, 8)`` and ``proprio`` has shape ``(B, 1, 7)``
    in the historical PerAct path.  The final gripper/action fields are kept
    unchanged; position and Euler orientation are accumulated from proprio.
    """
    pos = proprio[..., :3] + action[..., :3].cumsum(1)
    orn = proprio[..., 3:6] + action[..., 3:6].cumsum(1)
    orn = (orn + torch.pi) % (2 * torch.pi) - torch.pi
    return torch.cat([pos, orn, action[..., 6:]], -1)
