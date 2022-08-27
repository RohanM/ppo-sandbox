import gym
import numpy as np
from torch import nn, tensor, Tensor

# Observations: Tensor[8]
# Actions: Four discrete actions
class ActorModel(nn.Module):
    def __init__(self, num_input=8, num_hidden=32, num_output=4):
        super().__init__()
        self.linear1 = nn.Linear(num_input, num_hidden)
        self.linear2 = nn.Linear(num_hidden, num_hidden)
        self.linear3 = nn.Linear(num_hidden, num_output)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x: Tensor):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        out = self.softmax(out)
        return out


class CriticModel(nn.Module):
    def __init__(self, num_input=8, num_hidden=32):
        super().__init__()
        num_output = 1

        self.linear1 = nn.Linear(num_input, num_hidden)
        self.linear2 = nn.Linear(num_hidden, num_hidden)
        self.linear3 = nn.Linear(num_hidden, num_output)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x: Tensor):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        out = self.tanh(out)
        return out


env = gym.make('LunarLander-v2', new_step_api=True, render_mode='human')
n_state = env.observation_space.shape[0]
n_actions = env.action_space.n

ppo_steps = 128
max_episodes = 50
actor = ActorModel(num_input=n_state, num_output=n_actions)
critic = CriticModel(num_input=n_state)


for episode in range(max_episodes):
    states = []
    actions = []
    values = []
    masks = []
    rewards = []
    actions_probs = []

    state = env.reset()

    for i in range(ppo_steps):
        state_input = tensor(state)
        action_dist = actor(state_input)
        q_value = critic(state_input)
        action = np.random.choice(n_actions, p=action_dist.detach().numpy())

        observation, reward, terminated, truncated, info = env.step(action)

        mask = not (terminated or truncated)

        states.append(state)
        actions.append(action)
        values.append(q_value.item())
        masks.append(mask)
        rewards.append(reward)
        actions_probs.append(action_dist.detach().numpy())

        state = observation

        if terminated or truncated:
            env.reset()
            break

    # Check if we've reached our target performance
    #if best_reward > 0.9:
    #    break
