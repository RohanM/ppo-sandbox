import gym
import numpy as np
from numpy.typing import NDArray
from typing import Any

# A very simple environment that gives -1 reward for every step, but 20 reward
# for a sequence of 5 identical steps.
class SimpleEnvV0(gym.Env[NDArray[np.int32], int]):
    state: NDArray[np.int32]

    def __init__(self, render_mode=None):
        assert render_mode == None
        self.repeats_needed = 5
        self.reward_per_turn = -1
        self.reward_on_success = 20

        self.observation_space = gym.spaces.MultiDiscrete([2] * self.repeats_needed)
        self.action_space = gym.spaces.Discrete(2)

    def reset(self, seed=None, return_info=None, options=None):
        super().reset()
        self.state = np.zeros(self.repeats_needed, dtype=np.int32)
        self.done = False
        return self.state

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
    def __init__(self, render_mode=None, size=5, fail_threshold=-50):
        assert render_mode == None
        self.size = size
        self.fail_threshold = fail_threshold
        self.observation_space = gym.spaces.Discrete(self.size)
        self.action_space = gym.spaces.Discrete(self.size)

    def reset(self, seed=None, return_info=None, options=None):
        super().reset()
        self.state = np.random.randint(self.size)
        self.score = 0
        self.done = False
        return self.state

    def step(self, action):
        if action == self.state:
            reward = 1
        else:
            reward = -1
        self.score += reward
        self.done = self.score < self.fail_threshold
        info = { 'score': self.score }

        return self.state, reward, self.done, info


gym.envs.registration.register(id='SimpleEnv-v0', entry_point=SimpleEnvV0)
gym.envs.registration.register(id='SimpleEnv-v1', entry_point=SimpleEnvV1)
