import gym
import simple_env
import numpy as np
import torch
from torch import nn, tensor, Tensor, optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

rollout_steps = 500
max_episodes = 1000
num_epochs = 2

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
        ]
        super().__init__(*layers)


class RolloutBuffer(Dataset):
    def __init__(self):
        self.reset()

    def reset(self):
        self.states = []
        self.actions = []
        self.actions_logps = []
        self.masks = []
        self.rewards = []
        self.returns = None
        self.advantages = None

    def add_obs(self, state: Tensor, action: int, action_logp: float, mask: bool, reward: float):
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

    def get_returns(self):
        if self.returns is None: self.build_returns()
        return self.returns

    def get_advantages(self):
        return self.advantages

    def build_returns(self):
        batch_size = len(self.rewards)
        returns = torch.zeros(batch_size)

        for t in reversed(range(batch_size)):
            next_return = returns[t+1] if t < batch_size-1 else 0
            returns[t] = self.rewards[t] + (next_return * self.masks[t])

        self.returns = returns.unsqueeze(dim=1)

    def build_advantages(self, values, gamma=0.99, lmbda=0.95):
        batch_size = len(self.rewards)
        advantages = torch.zeros(batch_size)

        for t in reversed(range(batch_size)):
            next_value = values[t + 1] if t < batch_size-1 else values[t]
            next_advantage = advantages[t + 1] if t < batch_size-1 else advantages[t]

            delta = self.rewards[t] + (gamma * next_value * self.masks[t]) - values[t]
            advantages[t] = delta + (gamma * lmbda * next_advantage * self.masks[t])

        self.advantages = advantages.unsqueeze(dim=1)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return (
            self.states[idx],
            self.actions[idx],
            self.actions_logps[idx],
            self.masks[idx],
            self.rewards[idx],
            self.returns[idx],
            self.advantages[idx],
        )


class Trainer:
    def __init__(self, actor, critic, writer: SummaryWriter):
        self.actor = actor
        self.critic = critic
        self.actor_opt = optim.Adam(actor.parameters(), lr=actor_lr)
        self.critic_opt = optim.Adam(critic.parameters(), lr=critic_lr)
        self.writer = writer


    def actor_loss(self, newpolicy_logp, oldpolicy_logp, advantages):
        ratio = torch.exp(newpolicy_logp - oldpolicy_logp)
        p1 = ratio * advantages
        p2 = torch.clip(ratio, min=1 - epsilon, max=1 + epsilon) * advantages
        actor_loss = -torch.min(p1, p2).mean()

        approx_kl = (oldpolicy_logp - newpolicy_logp).mean().item()
        clipped = ratio.gt(1+epsilon) | ratio.lt(1-epsilon)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()

        return actor_loss, { 'approx_kl': approx_kl, 'clipfrac': clipfrac }


    def train(self, num_epochs: int, buf: RolloutBuffer):
        data_loader = DataLoader(buf, batch_size=64, shuffle=True)
        self.actor.train()
        self.critic.train()
        for epoch in range(num_epochs):
            for states, actions, actions_logps, _, _, returns, advantages in data_loader:
                new_actions_dists = self.actor(states)
                dist = torch.distributions.Categorical(probs=new_actions_dists)
                new_actions_logps = dist.log_prob(actions)
                values = self.critic(states)

                actor_loss_v, actor_loss_info = self.actor_loss(
                    new_actions_logps,
                    actions_logps.detach(),
                    advantages.detach()
                )
                critic_loss_v = F.mse_loss(values, returns)

                actor_loss_v.backward(retain_graph=True)
                self.writer.add_histogram(
                    "gradients/actor",
                    torch.cat([p.grad.view(-1) for p in actor.parameters()]),
                    episode
                )
                self.actor_opt.step()
                self.actor_opt.zero_grad()

                critic_loss_v.backward()
                self.writer.add_histogram(
                    "gradients/critic",
                    torch.cat([p.grad.view(-1) for p in critic.parameters()]),
                    episode
                )
                self.critic_opt.step()
                self.critic_opt.zero_grad()

                self.writer.add_scalar('actor loss', actor_loss_v.item(), episode)
                self.writer.add_scalar('critic loss', critic_loss_v.item(), episode)
                self.writer.add_scalar('actor kl', actor_loss_info['approx_kl'], episode)
                self.writer.add_scalar('actor clipfrac', actor_loss_info['clipfrac'], episode)


def cat(a, b):
    return torch.cat((a, b.float().unsqueeze(dim=0)))

def normalise(t: Tensor) -> Tensor:
    return (t - t.mean()) / (t.std() + 1e-10)

env = gym.make('CartPole-v1', new_step_api=True)

if isinstance(env.observation_space, gym.spaces.MultiDiscrete):
    n_state = len(env.observation_space)
else:
    n_state = env.observation_space.shape[0]
n_actions = env.action_space.n

actor = ActorModel(num_input=n_state, num_output=n_actions)
critic = CriticModel(num_input=n_state)


if __name__ == '__main__':
    writer = SummaryWriter()
    writer.add_hparams({
        'rollout_steps': rollout_steps,
        'max_episodes': max_episodes,
        'num_epochs': num_epochs,
        'actor_lr': actor_lr,
        'critic_lr': critic_lr,
        'lmbda': lmbda,
        'gamma': gamma,
        'epsilon': epsilon,
    }, {})

    trainer = Trainer(actor, critic, writer)

    for episode in range(max_episodes):
        buf = RolloutBuffer()

        state = env.reset()

        for i in range(rollout_steps):
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
        actions_logps = buf.get_actions_logps()
        masks = buf.get_masks()
        values = critic(states)
        rewards = buf.get_rewards()
        returns = buf.get_returns()
        buf.build_advantages(values)
        advantages = normalise(buf.get_advantages())

        num_eps = rollout_steps - np.count_nonzero(masks)
        avg_reward = rewards.sum().item() / num_eps
        print(f'{rewards.mean():.4f}, {rewards.max()}, {num_eps}, {avg_reward:.4f}')

        writer.add_histogram("loss/advantages", advantages, episode)
        writer.add_histogram("loss/values", values, episode)
        writer.add_histogram("loss/returns", returns, episode)

        writer.add_scalar('avg reward', avg_reward, episode)
        writer.add_scalar('max reward', rewards.max().item(), episode)
        writer.add_scalar('avg episode length', rollout_steps / num_eps, episode)

        trainer.train(num_epochs, buf)

    writer.close()
    torch.save(actor.state_dict(), 'actor.pth')
    torch.save(critic.state_dict(), 'critic.pth')
