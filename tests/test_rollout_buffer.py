import pytest
import torch
from torch import tensor
from ppo_from_scratch import RolloutBuffer

@pytest.fixture
def buf():
    buf = RolloutBuffer()
    for i in range(3):
        buf.add_obs(
            state=tensor([1, 2, 3]),
            action=1,
            action_logp=tensor(0.5),
            mask=[True, True, False][i],
            reward=1
        )
    buf.add_obs(
        state=tensor([1, 2, 3]),
        action=1,
        action_logp=tensor(0.5),
        mask=True,
        reward=1
    )
    buf.add_obs(
        state=tensor([1, 2, 3]),
        action=1,
        action_logp=tensor(0.5),
        mask=True,
        reward=1
    )
    return buf

def test_get_states(buf):
    assert torch.equal(
        buf.get_states(),
        tensor([[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]])
    )

def test_get_actions(buf):
    assert torch.equal(buf.get_actions(), tensor([[1], [1], [1], [1], [1]]))

def test_get_actions_logps(buf):
    assert torch.equal(buf.get_actions_logps(), tensor([[0.5], [0.5], [0.5], [0.5], [0.5]]))

def test_get_masks(buf):
    assert buf.get_masks() == [True, True, False, True, True]

def test_get_rewards(buf):
    assert torch.equal(buf.get_rewards(), tensor([[1], [1], [1], [1], [1]]))

def test_get_returns(buf):
    returns = buf.get_returns()
    expected = tensor([[3], [2], [1], [2], [1]]).float()
    assert torch.equal(returns, expected)

def test_build_advantages(buf):
    advantages = buf.build_advantages(tensor([0, 1, 2, 3, 4]), gamma=0.9, lmbda=0.8)
    advantages = buf.get_advantages()
    expected = tensor([[2.68], [1.08], [-1.00], [2.03], [0.60]])
    assert torch.isclose(advantages, expected, atol=0.01).all()

def test_dataset_len(buf):
    buf.build_returns()
    buf.build_advantages(tensor([0, 1, 2, 3, 4]), gamma=0.9, lmbda=0.8)
    assert len(buf) == 5

def test_dataset_getitem(buf):
    buf.build_returns()
    buf.build_advantages(tensor([0, 1, 2, 3, 4]), gamma=0.9, lmbda=0.8)
    data = buf[0]
    assert len(data) == 7
    assert torch.equal(data[0], tensor([1, 2, 3]))
    assert data[1] == 1
    assert torch.equal(data[2], tensor(0.5000))
    assert data[3] == True
    assert data[4] == 1
    assert torch.equal(data[5], tensor([3.]))
    assert torch.equal(data[6], tensor([2.6776]))

# Check expected values for advantages against equations
