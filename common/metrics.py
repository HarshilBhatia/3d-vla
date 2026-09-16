"""Model-independent action metrics shared by training and evaluation."""

import torch


def compute_metrics(pred, gt):
    """Compute aggregate and per-sample trajectory metrics."""
    pos_l2 = ((pred[..., :3] - gt[..., :3]) ** 2).sum(-1).sqrt()
    quat_l1 = (pred[..., 3:-1] - gt[..., 3:-1]).abs().sum(-1)
    quat_l1_ = (pred[..., 3:-1] + gt[..., 3:-1]).abs().sum(-1)
    select_mask = (quat_l1 < quat_l1_).float()
    quat_l1 = select_mask * quat_l1 + (1 - select_mask) * quat_l1_
    openness = ((pred[..., -1:] >= 0.5) == (gt[..., -1:] >= 0.5)).bool()
    prefix = "traj_"
    aggregate = {
        prefix + "pos_l2": pos_l2.mean(),
        prefix + "pos_acc_001": (pos_l2 < 0.01).float().mean(),
        prefix + "rot_l1": quat_l1.mean(),
        prefix + "rot_acc_0025": (quat_l1 < 0.025).float().mean(),
        prefix + "gripper": openness.flatten().float().mean(),
    }
    per_sample = {
        prefix + "pos_l2": pos_l2.mean(-1),
        prefix + "pos_acc_001": (pos_l2 < 0.01).float().mean(-1),
        prefix + "rot_l1": quat_l1.mean(-1),
        prefix + "rot_acc_0025": (quat_l1 < 0.025).float().mean(-1),
        prefix + "gripper": openness.float().flatten(1).mean(-1),
    }
    return aggregate, per_sample
