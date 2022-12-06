import pytest
import torch
from torch import tensor
import numpy as np
from ppo_from_scratch import RolloutBuffer

@pytest.fixture
def buf():
    return RolloutBuffer()


def test_add_obs(buf):
    buf.add_obs(
        state=tensor([1, 2, 3, 4, 5, 6, 7, 8]),
        action=tensor([1]),
        action_logp=tensor(0.5),
        mask=np.array([True]),
        reward=np.array([1])
    )
    assert (buf.states[0] == tensor([1, 2, 3, 4, 5, 6, 7, 8])).all().item()
    assert (buf.actions[0] == tensor([1])).all().item()
    assert (buf.actions_logps[0] == tensor(0.5)).all().item()
    assert buf.masks[0] == np.array([True])
    assert buf.rewards[0] == np.array([1])

def test_reset(buf):
    buf.add_obs(
        state=tensor([1, 2, 3, 4, 5, 6, 7, 8]),
        action=tensor([1]),
        action_logp=tensor(0.5),
        mask=np.array([True]),
        reward=np.array([1])
    )
    buf.reset()
    assert buf.states == []
    assert buf.actions == []
    assert buf.actions_logps == []
    assert buf.masks == []
    assert buf.rewards == []
