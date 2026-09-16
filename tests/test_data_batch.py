"""Contract tests for data-domain batch/action helpers."""

import pytest

torch = pytest.importorskip("torch")

from data.batch import actions_collate_fn, base_collate_fn, relative_to_absolute


def test_base_collate_preserves_rlbench_list_tensor_and_none_fields():
    batch = [
        {"task": ["open"], "instr": ["open drawer"], "variation": [0],
         "action": torch.ones(1, 2), "pcd": None},
        {"task": ["close"], "instr": ["close drawer"], "variation": [1],
         "action": torch.zeros(1, 2), "pcd": None},
    ]
    out = base_collate_fn(batch)
    assert out["task"] == ["open", "close"]
    assert out["instr"] == ["open drawer", "close drawer"]
    assert out["variation"] == [0, 1]
    assert torch.equal(out["action"], torch.tensor([[1.0, 1.0], [0.0, 0.0]]))
    assert out["pcd"] is None


def test_actions_collate_concatenates_action_rows():
    out = actions_collate_fn([{"action": torch.ones(2, 3)}, {"action": torch.zeros(1, 3)}])
    assert out["action"].shape == (3, 3)
    assert torch.equal(out["action"][-1], torch.zeros(3))


def test_relative_to_absolute_accumulates_and_wraps_euler_angles():
    action = torch.tensor([[[1.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.2, 1.0],
                            [1.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.3, 1.0]]])
    proprio = torch.tensor([[[10.0, 20.0, 30.0, 2.5, 0.0, 0.0, 0.0]]])
    out = relative_to_absolute(action, proprio)
    assert torch.allclose(out[0, :, :3], torch.tensor([[11.0, 20.0, 30.0], [12.0, 20.0, 30.0]]))
    expected_orn = torch.tensor([[-0.7831853, 0.0, 0.0], [2.2168148, 0.0, 0.0]])
    assert torch.allclose(out[0, :, 3:6], expected_orn, atol=1e-6)
    assert torch.equal(out[0, :, 6:], action[0, :, 6:])
