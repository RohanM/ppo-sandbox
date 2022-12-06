import pytest
import torch
from torch import tensor
import numpy as np
from ppo_from_scratch import RolloutBuffer, RolloutDataset

@pytest.fixture
def dataset():
    buf = RolloutBuffer()
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
    values = tensor([[[0]], [[1]], [[2]], [[3]], [[4]]])
    return RolloutDataset(buf, values, 8)

def test_states(dataset):
    assert torch.equal(
        dataset.states,
        tensor([
            [1, 2, 3, 4, 5, 6, 7, 8],
            [1, 2, 3, 4, 5, 6, 7, 8],
            [1, 2, 3, 4, 5, 6, 7, 8],
            [1, 2, 3, 4, 5, 6, 7, 8],
            [1, 2, 3, 4, 5, 6, 7, 8],
        ])
    )

def test_actions(dataset):
    assert torch.equal(dataset.actions, tensor([1, 1, 1, 1, 1]))

def test_actions_logps(dataset):
    assert torch.equal(dataset.actions_logps, tensor([0.5, 0.5, 0.5, 0.5, 0.5]))

def test_masks(dataset):
    assert torch.equal(dataset.masks, tensor([True, True, False, True, True]))

def test_rewards(dataset):
    assert torch.equal(dataset.rewards, tensor([1, 1, 1, 1, 1]).float())

def test_returns(dataset):
    assert torch.isclose(
        dataset.returns,
        tensor([[3], [2], [1], [6], [5]]).float(),
        atol=0.2
    ).all()

def test_build_advantages(dataset):
    assert torch.isclose(
        dataset.advantages,
        tensor([[3], [1], [-1], [3], [1]]).float(),
        atol=0.2
    ).all()

def test_dataset_len(dataset):
    assert len(dataset) == 5

def test_dataset_getitem(dataset):
    data = dataset[0]
    assert len(data) == 7
    assert torch.equal(data[0], tensor([1, 2, 3, 4, 5, 6, 7, 8]))
    assert data[1] == 1
    assert torch.equal(data[2], tensor(0.5000))
    assert data[3].all()
    assert data[4] == 1
    assert torch.isclose(data[5], tensor([2.9676]), atol=0.01)
    assert torch.isclose(data[6], tensor([2.9676]), atol=0.01)
