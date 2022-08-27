import gym
import numpy as np
import torch
from torch import nn, tensor, Tensor, optim
from torch.nn import functional as F

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

def get_advantages(values, masks, rewards):
    lmbda = 0.95
    gamma = 0.99

    returns = []
    gae = 0
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * values[i + 1] * masks[i] - values[i]
        gae = delta + gamma * lmbda * masks[i] * gae
        returns.insert(0, gae + values[i])

    adv = np.array(returns) - values[:-1]
    return returns, (adv - np.mean(adv)) / (np.std(adv) + 1e-10)


def ppo_loss(newpolicy_probs, oldpolicy_probs, advantages, rewards, values):
    epsilon = 0.2
    critic_discount = 0.5
    entropy_beta = 0.001

    advantages = tensor(advantages).unsqueeze(dim=1)
    ratio = torch.exp(torch.log(newpolicy_probs + 1e-10) - torch.log(oldpolicy_probs + 1e-10)).detach()
    p1 = ratio * advantages
    p2 = torch.clip(ratio, min=1 - epsilon, max=1 + epsilon) * advantages
    actor_loss = -torch.mean(torch.minimum(p1, p2))
    critic_loss = torch.mean(torch.square(rewards - values[:-1]))
    total_loss = critic_discount * critic_loss + actor_loss - entropy_beta * torch.mean(
        -(newpolicy_probs * torch.log(newpolicy_probs + 1e-10))
    )
    return total_loss



env = gym.make('LunarLander-v2', new_step_api=True, render_mode='human')
n_state = env.observation_space.shape[0]
n_actions = env.action_space.n

ppo_steps = 128
max_episodes = 50
num_epochs = 8

actor = ActorModel(num_input=n_state, num_output=n_actions)
critic = CriticModel(num_input=n_state)

actor_opt = optim.Adam(actor.parameters(), lr=1e-4)
critic_opt = optim.Adam(critic.parameters(), lr=1e-4)

for episode in range(max_episodes):
    states = []
    actions = []
    values = []
    masks = []
    rewards = []
    actions_probs = tensor([])

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
        actions_probs = torch.cat((actions_probs, action_dist.detach().unsqueeze(dim=0)), dim=0)

        state = observation

        if terminated or truncated:
            env.reset()
            break


    q_value = critic(tensor(states[-1]))
    values.append(q_value.item())
    returns, advantages = get_advantages(values, masks, rewards)

    rewards = tensor(rewards)
    values = tensor(values)
    returns = tensor(returns).float()

    # Training loop
    actor.train()
    critic.train()
    states_input = tensor(states)
    for epoch in range(num_epochs):
        # One episode per batch - is this optimal?
        actor_loss = ppo_loss(actor(states_input), actions_probs, advantages, rewards, values)
        critic_loss = F.mse_loss(critic(states_input).squeeze(), returns)

        actor_loss.backward()
        critic_loss.backward()
        actor_opt.step()
        critic_opt.step()
        actor_opt.zero_grad()
        critic_opt.zero_grad()

        if epoch == 7:
            print(f'Episode: {episode}, actor loss: {actor_loss.item()}, critic loss: {critic_loss.item()}')
