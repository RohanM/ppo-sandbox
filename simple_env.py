import gym
import numpy as np


# A very simple environment that gives -1 reward for every step, but 20 reward
# for a sequence of 5 identical steps.
class SimpleEnv(gym.Env):
    def __init__(self, render_mode=None):
        assert render_mode == None
        self.repeats_needed = 5
        self.reward_per_turn = -1
        self.reward_on_success = 20

        self.observation_space = gym.spaces.MultiDiscrete([2] * self.repeats_needed)
        self.action_space = gym.spaces.Discrete(2)

    def reset(self, seed=None, return_info=None, options=None):
        super().reset()
        self.state = np.zeros(self.repeats_needed)
        self.done = False
        return self.state

    def step(self, action):
        if action in [0, 1]:
            self.state = np.concatenate((self.state[1:], [action+1]))
        else:
            raise ValueError("Invalid action")
        self.done = (self.state == 1).all()
        reward = self.reward_on_success if self.done else self.reward_per_turn
        info = {}
        return self.state, reward, self.done, info


gym.envs.registration.register(id='SimpleEnv-v0', entry_point=SimpleEnv)
