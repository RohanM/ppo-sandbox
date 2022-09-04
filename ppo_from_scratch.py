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


class RolloutBuffer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.states = []
        self.actions = []
        self.actions_logps = []
        self.masks = []
        self.rewards = []

    def add_obs(self, state, action, action_logp, mask, reward):
        self.states.append(state)
        self.actions.append(action)
        self.actions_logps.append(action_logp)
        self.masks.append(mask)
        self.rewards.append(reward)

    def get_states(self, extended: bool = False):
        states = self.states + self.states[-1:] if extended else self.states
        return torch.stack(states)

    def get_actions(self):
        return torch.tensor(self.actions).unsqueeze(1)

    def get_actions_logps(self):
        return torch.stack(self.actions_logps).unsqueeze(1)

    def get_masks(self):
        return self.masks

    def get_rewards(self):
        return tensor(self.rewards).unsqueeze(1)

    def get_returns(self, values):
        returns = []
        gae = 0
        for i in reversed(range(len(self.rewards))):
            delta = self.rewards[i] + gamma * values[i + 1] * self.masks[i] - values[i]
            gae = delta + gamma * lmbda * self.masks[i] * gae
            returns.insert(0, gae + values[i])
        returns_t = torch.cat(returns).float().unsqueeze(dim=1)
        return returns_t

    def get_advantages(self, returns, values):
        adv = returns - values[:-1]
        return (adv - adv.mean()) / (adv.std() + 1e-10)


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



env = gym.make('CartPole-v1', new_step_api=True)

if isinstance(env.observation_space, gym.spaces.MultiDiscrete):
    n_state = len(env.observation_space)
else:
    n_state = env.observation_space.shape[0]
n_actions = env.action_space.n

actor = ActorModel(num_input=n_state, num_output=n_actions)
critic = CriticModel(num_input=n_state)

actor_opt = optim.Adam(actor.parameters(), lr=actor_lr)
critic_opt = optim.Adam(critic.parameters(), lr=critic_lr)

if __name__ == '__main__':
    writer = SummaryWriter()
    writer.add_hparams({
        'rollout_steps': ppo_steps,
        'max_episodes': max_episodes,
        'num_epochs': num_epochs,
        'actor_lr': actor_lr,
        'critic_lr': critic_lr,
        'lmbda': lmbda,
        'gamma': gamma,
        'epsilon': epsilon,
    }, {})

    for episode in range(max_episodes):
        buf = RolloutBuffer()

        state = env.reset()

        for i in range(ppo_steps):
            state_input = tensor(state).float()
            action_dist = actor(state_input.unsqueeze(dim=0)).squeeze()

            dist = torch.distributions.Categorical(probs=action_dist)
            action = dist.sample()
            action_logp = dist.log_prob(action) # The log prob of the action we took

            observation, reward, terminated, truncated, info = env.step(action.detach().data.numpy())

            mask = not (terminated or truncated)

            buf.add_obs(state_input, action, action_logp, mask, reward)

            state = observation

            if terminated or truncated:
                env.reset()

        states = buf.get_states()
        states_extended = buf.get_states(extended=True)
        actions_logps = buf.get_actions_logps()
        masks = buf.get_masks()
        values = critic(states_extended)
        rewards = buf.get_rewards()
        returns = buf.get_returns(values)
        advantages = buf.get_advantages(returns, values)

        num_eps = ppo_steps - np.count_nonzero(masks)
        avg_reward = rewards.sum().item() / num_eps
        print(f'{rewards.mean():.4f}, {rewards.max()}, {num_eps}, {avg_reward:.4f}')

        writer.add_scalar('avg reward', avg_reward, episode)
        writer.add_scalar('max reward', rewards.max().item(), episode)
        writer.add_scalar('avg episode length', ppo_steps / num_eps, episode)

        # Training loop
        actor.train()
        critic.train()
        for epoch in range(num_epochs):
            new_actions_dists = actor(states)
            dist = torch.distributions.Categorical(probs=new_actions_dists)
            new_actions_logps = dist.log_prob(buf.get_actions())

            values = critic(states_extended)
            actor_loss_v, actor_loss_info = actor_loss(
                new_actions_logps,
                actions_logps.detach(),
                advantages.detach()
            )
            critic_loss_v = critic_loss(values, rewards)

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

            writer.add_histogram("loss/advantages", advantages, episode)
            writer.add_histogram("loss/values", values, episode)
            writer.add_histogram("loss/returns", returns, episode)

            writer.add_scalar('actor loss', actor_loss_v.item(), episode)
            writer.add_scalar('critic loss', critic_loss_v.item(), episode)
            writer.add_scalar('actor kl', actor_loss_info['approx_kl'], episode)
            writer.add_scalar('actor clipfrac', actor_loss_info['clipfrac'], episode)
