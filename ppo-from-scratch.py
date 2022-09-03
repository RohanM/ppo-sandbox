import gym
import simple_env
import numpy as np
import torch
from torch import nn, tensor, Tensor, optim
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

ppo_steps = 4000
max_episodes = 100
num_epochs = 10

actor_lr = 3e-4
critic_lr = 1e-3

lmbda = 0.95
gamma = 0.99
epsilon = 0.2

writer = SummaryWriter()

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


def actor_loss(newpolicy_logp, oldpolicy_logp, advantages):
    ratio = torch.exp(newpolicy_logp - oldpolicy_logp)
    p1 = ratio * advantages
    p2 = torch.clip(ratio, min=1 - epsilon, max=1 + epsilon) * advantages
    actor_loss = -torch.min(p1, p2).mean()

    approx_kl = (oldpolicy_logp - newpolicy_logp).mean().item()
    clipped = ratio.gt(1+epsilon) | ratio.lt(1-epsilon)
    clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()

    return actor_loss, { 'approx_kl': approx_kl, 'clipfrac': clipfrac }

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
    states = []
    actions = []
    masks = []
    rewards = []
    actions_logps = []

    state = env.reset()

    for i in range(ppo_steps):
        state_input = tensor(state).float()
        action_dist = actor(state_input.unsqueeze(dim=0)).squeeze()

        dist = torch.distributions.Categorical(probs=action_dist)
        action = dist.sample()
        action_logp = dist.log_prob(action) # The log prob of the action we took

        observation, reward, terminated, truncated, info = env.step(action.detach().data.numpy())

        mask = not (terminated or truncated)

        states.append(state_input)
        actions.append(action)
        actions_logps.append(action_logp)
        masks.append(mask)
        rewards.append(reward)

        state = observation

        if terminated or truncated:
            env.reset()

    states_t = torch.stack(states)
    states_extended_t = torch.stack(states + states[-1:])
    actions_logps_t = torch.stack(actions_logps).unsqueeze(1)
    values = critic(states_extended_t).detach()
    rewards_t = torch.tensor(rewards).unsqueeze(1)
    returns, advantages = get_advantages(values, masks, rewards)

    num_eps = np.count_nonzero(masks)
    avg_reward = rewards_t.sum().item() / num_eps
    print(f'{rewards_t.mean():.4f}, {rewards_t.max()}, {num_eps}, {avg_reward:.4f}')

    writer.add_scalar('avg reward', avg_reward, episode)
    writer.add_scalar('max reward', rewards_t.max().item(), episode)
    writer.add_scalar('avg episode length', ppo_steps / num_eps, episode)

    # Training loop
    actor.train()
    critic.train()
    for epoch in range(num_epochs):
        new_actions_dists = actor(states_t)
        dist = torch.distributions.Categorical(probs=new_actions_dists)
        new_actions_logps = dist.log_prob(tensor(actions)).unsqueeze(dim=1)

        values = critic(states_extended_t)
        actor_loss_v, actor_loss_info = actor_loss(
            new_actions_logps,
            actions_logps_t.detach(),
            advantages.detach()
        )
        critic_loss_v = critic_loss(values, rewards_t)

        actor_loss_v.backward(retain_graph=True)
        writer.add_histogram(
            "gradients/actor",
            torch.cat([p.grad.view(-1) for p in actor.parameters()]),
            episode
        )
        actor_opt.step()
        actor_opt.zero_grad()

        critic_loss_v.backward()
        writer.add_histogram(
            "gradients/critic",
            torch.cat([p.grad.view(-1) for p in critic.parameters()]),
            episode
        )
        critic_opt.step()
        critic_opt.zero_grad()

    writer.add_scalar('actor loss', actor_loss_v.item(), episode)
    writer.add_scalar('critic loss', critic_loss_v.item(), episode)
    writer.add_scalar('actor kl', actor_loss_info['approx_kl'], episode)
    writer.add_scalar('actor clipfrac', actor_loss_info['clipfrac'], episode)
