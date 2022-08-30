import gym
import simple_env
import numpy as np
import torch
from torch import nn, tensor, Tensor, optim
from torch.nn import functional as F

ppo_steps = 4000
max_episodes = 100
num_epochs = 10

actor_lr = 3e-4
critic_lr = 1e-3

lmbda = 0.95
gamma = 0.99
epsilon = 0.2
entropy_coeff = 0


class ActorModel(nn.Sequential):
    def __init__(self, num_input=8, num_hidden=32, num_output=4):
        layers = [
            nn.Linear(num_input, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_output),
            nn.Softmax(dim=1),
        ]
        super().__init__(*layers)


class CriticModel(nn.Sequential):
    def __init__(self, num_input=8, num_hidden=32):
        num_output = 1
        layers = [
            nn.Linear(num_input, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_output),
            nn.Tanh(),
        ]
        super().__init__(*layers)


def get_advantages(values, masks, rewards):
    returns = []
    gae = 0
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * values[i + 1] * masks[i] - values[i]
        gae = delta + gamma * lmbda * masks[i] * gae
        returns.insert(0, gae + values[i])
    t_returns = torch.cat(returns).float().unsqueeze(dim=1)

    adv = t_returns - values[:-1]
    return t_returns, (adv - adv.mean()) / (adv.std() + 1e-10)


def actor_loss(newpolicy_probs, oldpolicy_probs, advantages):
    ratio = torch.exp(torch.log(newpolicy_probs + 1e-10) - torch.log(oldpolicy_probs + 1e-10))
    p1 = ratio * advantages
    p2 = torch.clip(ratio, min=1 - epsilon, max=1 + epsilon) * advantages
    entropy = entropy_coeff * (newpolicy_probs * torch.log(newpolicy_probs + 1e-10)).mean()
    actor_loss = -torch.min(p1, p2).mean() - entropy

    # approx_kl = (torch.log(oldpolicy_probs) - torch.log(newpolicy_probs)).mean().item()
    # clipped = ratio.gt(1+epsilon) | ratio.lt(1-epsilon)
    # clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
    # print(f'KL: {approx_kl:.4f}, clipfrac: {clipfrac:.4f}')

    return actor_loss

def critic_loss(values, rewards):
    return F.mse_loss(values[:-1], rewards)


def cat(a, b):
    return torch.cat((a, b.float().unsqueeze(dim=0)))


#env = gym.make('LunarLander-v2', new_step_api=True)
env = gym.make('SimpleEnv-v0', new_step_api=True)

if isinstance(env.observation_space, gym.spaces.MultiDiscrete):
    n_state = len(env.observation_space)
else:
    n_state = env.observation_space.shape[0]
n_actions = env.action_space.n

actor = ActorModel(num_input=n_state, num_output=n_actions)
critic = CriticModel(num_input=n_state)

actor_opt = optim.Adam(actor.parameters(), lr=actor_lr)
critic_opt = optim.Adam(critic.parameters(), lr=critic_lr)

for episode in range(max_episodes):
    states = tensor([])
    actions = []
    masks = []
    rewards = tensor([])
    actions_probs = tensor([])

    state = env.reset()

    for i in range(ppo_steps):
        state_input = tensor(state).float()
        action_dist = actor(state_input.unsqueeze(dim=0)).squeeze()
        action = np.random.choice(n_actions, p=action_dist.detach().numpy())

        observation, reward, terminated, truncated, info = env.step(action)

        mask = not (terminated or truncated)

        states = cat(states, state_input)
        actions.append(action)
        masks.append(mask)
        rewards = cat(rewards, tensor(reward).unsqueeze(dim=0))
        actions_probs = cat(actions_probs, action_dist.detach())

        state = observation

        if terminated or truncated:
            env.reset()

    print(f'{rewards.mean():.4f}, {rewards.max()}, {np.count_nonzero(actions)}')

    values = critic(torch.cat((states, states[-1].unsqueeze(dim=0)))).detach()
    returns, advantages = get_advantages(values, masks, rewards)

    # Training loop
    actor.train()
    critic.train()
    for epoch in range(num_epochs):
        new_actions_probs = actor(states)
        values = critic(torch.cat((states, states[-1].unsqueeze(dim=0))))
        actor_loss_v = actor_loss(new_actions_probs, actions_probs, advantages)
        critic_loss_v = critic_loss(values, rewards)

        actor_loss_v.backward(retain_graph=True)
        actor_opt.step()
        actor_opt.zero_grad()

        critic_loss_v.backward()
        critic_opt.step()
        critic_opt.zero_grad()
