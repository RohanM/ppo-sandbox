import gym
import numpy as np
import torch
from torch import nn, tensor, Tensor, optim
from torch.nn import functional as F

# Observations: Tensor[8]
# Actions: Four discrete actions
class ActorModel(nn.Sequential):
    def __init__(self, num_input=8, num_hidden=32, num_output=4):
        layers = [
            nn.Linear(num_input, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_output),
            nn.Softmax(dim=0),
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
    lmbda = 0.95
    gamma = 0.99

    returns = []
    gae = 0
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * values[i + 1] * masks[i] - values[i]
        gae = delta + gamma * lmbda * masks[i] * gae
        returns.insert(0, gae + values[i])
    t_returns = torch.cat(returns).float().unsqueeze(dim=1)

    adv = t_returns - values[:-1]
    return t_returns, (adv - adv.mean()) / (adv.std() + 1e-10)


def ppo_loss(newpolicy_probs, oldpolicy_probs, advantages, rewards, values):
    epsilon = 0.2
    critic_discount = 0.5
    entropy_beta = 0.001

    advantages = advantages.unsqueeze(dim=1)
    ratio = torch.exp(torch.log(newpolicy_probs + 1e-10) - torch.log(oldpolicy_probs + 1e-10)).detach()
    p1 = ratio * advantages
    p2 = torch.clip(ratio, min=1 - epsilon, max=1 + epsilon) * advantages
    actor_loss = -torch.mean(torch.minimum(p1, p2))
    critic_loss = torch.mean(torch.square(rewards - values[:-1]))
    total_loss = critic_discount * critic_loss + actor_loss - entropy_beta * torch.mean(
        -(newpolicy_probs * torch.log(newpolicy_probs + 1e-10))
    )
    return total_loss


def cat(a, b):
    return torch.cat((a, b.float().unsqueeze(dim=0)))


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
    states = tensor([])
    actions = []
    masks = []
    rewards = tensor([])
    actions_probs = tensor([])

    state = env.reset()

    for i in range(ppo_steps):
        state_input = tensor(state)
        action_dist = actor(state_input)
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
            break

    # Training loop
    actor.train()
    critic.train()
    for epoch in range(num_epochs):
        # One episode per batch - is this optimal?
        new_actions_probs = actor(states)
        values = critic(torch.cat((states, states[-1].unsqueeze(dim=0))))
        returns, advantages = get_advantages(values, masks, rewards)

        actor_loss, critic_loss = ppo_loss(new_actions_probs, actions_probs, advantages, rewards, values)

        actor_loss.backward(retain_graph=True)
        actor_opt.step()
        actor_opt.zero_grad()

        critic_loss.backward()
        critic_opt.step()
        critic_opt.zero_grad()

        if epoch == 7:
            print(f'Episode: {episode}, actor loss: {actor_loss.item()}, critic loss: {critic_loss.item()}')
