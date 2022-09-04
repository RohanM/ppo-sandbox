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

# Check expected values for advantages against equations
