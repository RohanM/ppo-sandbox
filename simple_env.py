import gym
import numpy as np
from numpy.typing import NDArray
from typing import Any, Optional

# A very simple environment that gives -1 reward for every step, but 20 reward
# for a sequence of 5 identical steps.
class SimpleEnvV0(gym.Env[NDArray[np.int32], int]):
    state: NDArray[np.int32]

    def __init__(self, render_mode: None = None) -> None:
        assert render_mode == None
        self.repeats_needed = 5
        self.reward_per_turn = -1
        self.reward_on_success = 20

        self.observation_space = gym.spaces.MultiDiscrete([2] * self.repeats_needed)
        self.action_space = gym.spaces.Discrete(2)

    def reset(self, seed: Optional[int] = None, options: Optional[dict[Any, Any]] = None) -> tuple[NDArray[np.int32], dict[Any, Any]]:
        super().reset()
        self.state = np.zeros(self.repeats_needed, dtype=np.int32)
        self.done = False
        return self.state, {}

    def step(self, action: int) -> tuple[NDArray[np.int32], float, bool, bool, dict[Any, Any]]:
        if action in [0, 1]:
            self.state = np.concatenate((self.state[1:], np.array([action+1])))
        else:
            raise ValueError("Invalid action")
        self.done = (self.state == 1).all()
        reward = self.reward_on_success if self.done else self.reward_per_turn
        return self.state, reward, self.done, False, {}


# A "copy me" env - +1 for successful copy, -1 for fail, -10 for if score
# drops below -50.
class SimpleEnvV1(gym.Env[int, int]):
    def __init__(self, render_mode: None = None, size: int = 5, fail_threshold: int = -50) -> None:
        assert render_mode == None
        self.size = size
        self.fail_threshold = fail_threshold
        self.observation_space = gym.spaces.Discrete(self.size)
        self.action_space = gym.spaces.Discrete(self.size)

    def reset(self, seed: Optional[int] = None, options: Optional[dict[Any, Any]] = None) -> tuple[int, dict[Any, Any]]:
        super().reset()
        self.state = np.random.randint(self.size)
        self.score = 0
        self.done = False
        return self.state, {}

    def step(self, action: int) -> tuple[int, float, bool, bool, dict[Any, Any]]:
        if action == self.state:
            reward = 1
        else:
            reward = -1
        self.score += reward
        self.done = self.score < self.fail_threshold
        info = { 'score': self.score }

        return self.state, reward, self.done, False, info


gym.envs.registration.register(id='SimpleEnv-v0', entry_point=SimpleEnvV0)
gym.envs.registration.register(id='SimpleEnv-v1', entry_point=SimpleEnvV1)
