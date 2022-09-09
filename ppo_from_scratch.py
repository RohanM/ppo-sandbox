import gym
import simple_env
import numpy as np
import torch
from torch import nn, tensor, Tensor, optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import wandb
import argparse


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class ActorModel(nn.Sequential):
    def __init__(self, num_input=8, num_hidden=32, num_output=4):
        layers = [
            layer_init(nn.Linear(num_input, num_hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(num_hidden, num_hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(num_hidden, num_output), std=0.01),
            nn.Softmax(dim=1),
        ]
        super().__init__(*layers)


class CriticModel(nn.Sequential):
    def __init__(self, num_input=8, num_hidden=32):
        num_output = 1
        layers = [
            layer_init(nn.Linear(num_input, num_hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(num_hidden, num_hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(num_hidden, num_output), std=1.0),
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
        return self.returns

    def get_advantages(self):
        return self.advantages

    def build_returns_advantages(self, values, gamma=0.99, lmbda=0.95):
        batch_size = len(self.rewards)
        advantages = torch.zeros(batch_size)

        for t in reversed(range(batch_size)):
            next_value = values[t + 1] if t < batch_size-1 else values[t]
            next_advantage = advantages[t + 1] if t < batch_size-1 else advantages[t]

            delta = self.rewards[t] + (gamma * next_value * self.masks[t]) - values[t]
            advantages[t] = delta + (gamma * lmbda * next_advantage * self.masks[t])

        self.advantages = advantages.unsqueeze(dim=1)
        self.returns = self.advantages + values

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
    def __init__(self, actor, critic, actor_lr, critic_lr, batch_size, epsilon, wandb):
        self.actor = actor
        self.critic = critic
        self.actor_opt = optim.Adam(actor.parameters(), lr=actor_lr)
        self.critic_opt = optim.Adam(critic.parameters(), lr=critic_lr)
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.wandb = wandb


    def actor_loss(self, newpolicy_logp, oldpolicy_logp, advantages):
        ratio = torch.exp(newpolicy_logp - oldpolicy_logp)
        p1 = ratio * advantages
        p2 = torch.clip(ratio, min=1 - self.epsilon, max=1 + self.epsilon) * advantages
        actor_loss = -torch.min(p1, p2).mean()

        approx_kl = (oldpolicy_logp - newpolicy_logp).mean().item()
        clipped = ratio.gt(1+self.epsilon) | ratio.lt(1-self.epsilon)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()

        return actor_loss, { 'approx_kl': approx_kl, 'clipfrac': clipfrac }


    def train(self, num_epochs: int, buf: RolloutBuffer):
        data_loader = DataLoader(buf, batch_size=self.batch_size, shuffle=True)
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
                critic_loss_v = F.mse_loss(values, returns.detach())

                actor_loss_v.backward(retain_graph=True)
                # self.wandb.log({
                #     'gradients/actor': torch.cat([p.grad.view(-1) for p in actor.parameters()])
                # })
                self.actor_opt.step()
                self.actor_opt.zero_grad()

                critic_loss_v.backward()
                # self.wandb.log({
                #     'gradients/critic': torch.cat([p.grad.view(-1) for p in critic.parameters()])
                # })
                self.critic_opt.step()
                self.critic_opt.zero_grad()

                self.wandb.log({
                    'loss/actor': actor_loss_v.item(),
                    'loss/critic': critic_loss_v.item(),
                    'actor kl': actor_loss_info['approx_kl'],
                    'actor clipfrac': actor_loss_info['clipfrac'],
                })


def cat(a, b):
    return torch.cat((a, b.float().unsqueeze(dim=0)))

def normalise(t: Tensor) -> Tensor:
    return (t - t.mean()) / (t.std() + 1e-10)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gym', type=str, default='LunarLander-v2')
    parser.add_argument('--exp-name', type=str, default=None)
    parser.add_argument('--rollout-steps', type=int, default=4000)
    parser.add_argument('--max-episodes', type=int, default=1000)
    parser.add_argument('--num-epochs', type=int, default=4)
    parser.add_argument('--actor-lr', type=float, default=0.0003)
    parser.add_argument('--critic-lr', type=float, default=0.001)
    parser.add_argument('--lmbda', type=float, default=0.95)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epsilon', type=float, default=0.2)
    return parser.parse_args()



if __name__ == '__main__':
    args = parse_args()

    env = gym.make('LunarLander-v2', new_step_api=True)

    if isinstance(env.observation_space, gym.spaces.MultiDiscrete):
        n_state = len(env.observation_space)
    else:
        n_state = env.observation_space.shape[0]
        n_actions = env.action_space.n

    actor = ActorModel(num_input=n_state, num_output=n_actions)
    critic = CriticModel(num_input=n_state)

    wandb.init(
        project='ppo-sandbox-lunar-lander',
        name=args.exp_name,
        config={
            'rollout_steps': args.rollout_steps,
            'max_episodes': args.max_episodes,
            'num_epochs': args.num_epochs,
            'actor_lr': args.actor_lr,
            'critic_lr': args.critic_lr,
            'lmbda': args.lmbda,
            'gamma': args.gamma,
            'epsilon': args.epsilon,
        }
    )

    wandb.watch(actor)
    wandb.watch(critic)

    trainer = Trainer(actor, critic, args.actor_lr, args.critic_lr, args.batch_size, args.epsilon, wandb)
    avg_rewards = []

    for episode in range(args.max_episodes):
        buf = RolloutBuffer()

        state = env.reset()

        for i in range(args.rollout_steps):
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
        buf.build_returns_advantages(values)
        returns = buf.get_returns()
        advantages = normalise(buf.get_advantages())

        num_eps = args.rollout_steps - np.count_nonzero(masks)
        if masks[-1]: num_eps += 1

        avg_reward = rewards.sum().item() / num_eps
        avg_rewards.append(avg_reward)
        print(f'{episode+1}/{args.max_episodes}, {rewards.mean():.4f}, {rewards.max():.4f}, {num_eps}, {avg_reward:.4f}')

        wandb.log({
            'episode/advantages': advantages,
            'episode/values': values,
            'episode/returns': returns,
            'avg reward': avg_reward,
            'max reward': rewards.max().item(),
            'avg episode length': args.rollout_steps / num_eps,
        })

        trainer.train(args.num_epochs, buf)

    torch.save(actor.state_dict(), 'actor.pth')
    torch.save(critic.state_dict(), 'critic.pth')
