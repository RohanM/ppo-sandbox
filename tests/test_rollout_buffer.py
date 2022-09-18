import pytest
import torch
from torch import tensor
import numpy as np
from ppo_from_scratch import RolloutBuffer

@pytest.fixture
def buf():
    buf = RolloutBuffer(8)
    for i in range(3):
        buf.add_obs(
            state=tensor([1, 2, 3, 4, 5, 6, 7, 8]),
            action=tensor([1]),
            action_logp=tensor(0.5),
            mask=np.array([[True, True, False][i]]),
            reward=np.array([1])
        )
    buf.add_obs(
        state=tensor([1, 2, 3, 4, 5, 6, 7, 8]),
        action=tensor([1]),
        action_logp=tensor(0.5),
        mask=np.array([True]),
        reward=np.array([1])
    )
    buf.add_obs(
        state=tensor([1, 2, 3, 4, 5, 6, 7, 8]),
        action=tensor([1]),
        action_logp=tensor(0.5),
        mask=np.array([True]),
        reward=np.array([1])
    )
    buf.prep_data(tensor([[[0]], [[1]], [[2]], [[3]], [[4]]]))
    return buf

def test_get_states(buf):
    assert torch.equal(
        buf.get_states(),
        tensor([
            [1, 2, 3, 4, 5, 6, 7, 8],
            [1, 2, 3, 4, 5, 6, 7, 8],
            [1, 2, 3, 4, 5, 6, 7, 8],
            [1, 2, 3, 4, 5, 6, 7, 8],
            [1, 2, 3, 4, 5, 6, 7, 8],
        ])
    )

def test_get_actions(buf):
    assert torch.equal(buf.get_actions(), tensor([1, 1, 1, 1, 1]))

def test_get_actions_logps(buf):
    assert torch.equal(buf.get_actions_logps(), tensor([0.5, 0.5, 0.5, 0.5, 0.5]))

def test_get_masks(buf):
    assert torch.equal(buf.get_masks(), tensor([True, True, False, True, True]))

def test_get_rewards(buf):
    assert torch.equal(buf.get_rewards(), tensor([1, 1, 1, 1, 1]).float())

def test_get_returns(buf):
    assert torch.isclose(
        buf.get_returns(),
        tensor([[3], [2], [1], [6], [5]]).float(),
        atol=0.2
    ).all()

def test_build_advantages(buf):
    assert torch.isclose(
        buf.get_advantages(),
        tensor([[3], [1], [-1], [3], [1]]).float(),
        atol=0.2
    ).all()

def test_dataset_len(buf):
    assert len(buf) == 5

def test_dataset_getitem(buf):
    data = buf[0]
    assert len(data) == 7
    assert torch.equal(data[0], tensor([1, 2, 3, 4, 5, 6, 7, 8]))
    assert data[1] == 1
    assert torch.equal(data[2], tensor(0.5000))
    assert data[3].all()
    assert data[4] == 1
    assert torch.isclose(data[5], tensor([2.9676]), atol=0.01)
    assert torch.isclose(data[6], tensor([2.9676]), atol=0.01)
