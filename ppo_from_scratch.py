import gym
import random
import time
import simple_env
import numpy as np
import torch
from torch import nn, tensor, Tensor, optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import wandb
import argparse
from typing import cast, Callable


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class ActorModel(nn.Sequential):
    def __init__(self, num_input: int = 8, num_hidden: int = 32, num_output: int = 4):
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
    def __init__(self, num_input: int = 8, num_hidden: int = 32):
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
    def __init__(self, n_state: int, device: torch.device = torch.device('cpu')):
        self.n_state = n_state
        self.device = device
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

    def prep_data(self, values: Tensor):
        self.__build_returns_advantages(values)

        self.states = torch.stack(self.states).reshape(
            (-1, self.n_state)
        ).to(self.device)
        self.actions = torch.stack(self.actions).reshape(-1).to(self.device)
        self.actions_logps = torch.stack(self.actions_logps).reshape(-1).to(self.device)
        self.masks = tensor(np.array(self.masks)).reshape(-1).to(self.device)
        self.rewards = tensor(np.array(self.rewards)).float().reshape(-1).to(self.device)
        self.returns = self.returns.reshape(-1, 1)
        self.advantages = self.advantages.reshape(-1, 1)

    def get_states(self) -> list[Tensor]:
        return self.states

    def get_actions(self) -> Tensor:
        return self.actions

    def get_actions_logps(self) -> Tensor:
        return self.actions_logps

    def get_masks(self) -> list[bool]:
        return self.masks

    def get_rewards(self) -> Tensor:
        return self.rewards

    def get_returns(self) -> Tensor:
        return self.returns

    def get_advantages(self) -> Tensor:
        return self.advantages

    def __build_returns_advantages(self, values: Tensor, gamma: float = 0.99, lmbda: float = 0.95):
        batch_size = len(self.rewards)
        advantages = torch.zeros(batch_size).to(self.device)

        rewards = tensor(np.array(self.rewards)).float().unsqueeze(2).to(self.device)
        masks = tensor(np.array(self.masks)).unsqueeze(2).to(self.device)
        batch_size = rewards.shape[0]
        advantages = torch.zeros_like(rewards).to(self.device)

        for t in reversed(range(batch_size)):
            next_value = values[t + 1] if t < batch_size-1 else values[t]
            next_advantage = advantages[t + 1] if t < batch_size-1 else advantages[t]

            delta = rewards[t] + (gamma * next_value * masks[t]) - values[t]
            advantages[t] = delta + (gamma * lmbda * next_advantage * masks[t])

        self.advantages = advantages
        self.returns = advantages + values

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
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
    def __init__(self, actor: ActorModel, critic: CriticModel, actor_lr: float, critic_lr: float, batch_size: int, epsilon: float, wandb):
        self.actor = actor
        self.critic = critic
        self.actor_opt = optim.Adam(actor.parameters(), lr=actor_lr)
        self.critic_opt = optim.Adam(critic.parameters(), lr=critic_lr)
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.wandb = wandb


    def actor_loss(self, newpolicy_logp: Tensor, oldpolicy_logp: Tensor, advantages: Tensor) -> tuple[Tensor, dict]:
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

def get_device(args) -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available() and args.mps:
        return torch.device('mps')
    else:
        return torch.device('cpu')

def cat(a: Tensor, b: Tensor) -> Tensor:
    return torch.cat((a, b.float().unsqueeze(dim=0)))

def normalise(t: Tensor) -> Tensor:
    return (t - t.mean()) / (t.std() + 1e-10)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gym', type=str, default='LunarLander-v2')
    parser.add_argument('--exp-name', type=str, default=None)
    parser.add_argument('--tags', type=str, default=None)
    parser.add_argument('--num-envs', type=int, default=32)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--track', action='store_true')
    parser.add_argument('--record-video-every', type=int, default=None,
                        help='Record video every n steps (eg. 10000)')
    parser.add_argument('--rollout-steps', type=int, default=125)
    parser.add_argument('--max-episodes', type=int, default=500)
    parser.add_argument('--num-epochs', type=int, default=4)
    parser.add_argument('--actor-lr', type=float, default=0.0003)
    parser.add_argument('--critic-lr', type=float, default=0.001)
    parser.add_argument('--lmbda', type=float, default=0.95)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epsilon', type=float, default=0.2)
    parser.add_argument('--mps', action='store_true')
    return parser.parse_args()

def make_env(gym_id: str, seed: int, idx: int, exp_name: str, record_video_steps: bool) -> Callable:
    def thunk():
        env = gym.make(gym_id, render_mode='rgb_array')
        if record_video_steps is not None and idx == 0:
            env = gym.wrappers.RecordVideo(
                env,
                f'videos/{exp_name}',
                step_trigger=lambda t: t % record_video_steps == 0,
            )
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk


if __name__ == '__main__':
    args = parse_args()

    envs = gym.vector.AsyncVectorEnv(
        [make_env(args.gym, args.seed, i, args.exp_name, args.record_video_every) for i in range(args.num_envs)]
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    n_state = np.array(envs.single_observation_space.shape).prod()
    n_actions = cast(gym.spaces.Discrete, envs.single_action_space).n

    device = get_device(args)

    actor = ActorModel(num_input=n_state, num_output=n_actions)
    critic = CriticModel(num_input=n_state)

    actor.to(device)
    critic.to(device)

    wandb.init(
        mode='online' if args.track else 'disabled',
        project='ppo-sandbox-lunar-lander',
        name=args.exp_name,
        tags=args.tags,
        config={
            'seed': args.seed,
            'num_envs': args.num_envs,
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
        start_episode_time = time.time()
        buf = RolloutBuffer(n_state, device)

        state, info = envs.reset(seed=args.seed)

        for i in range(args.rollout_steps):
            state_input = tensor(state).to(device).float()
            action_dist = actor(state_input)

            dist = torch.distributions.Categorical(probs=action_dist)
            action = dist.sample()
            action_logp = dist.log_prob(action) # The log prob of the action we took

            observation, reward, terminated, truncated, info = envs.step(action.detach().cpu().numpy())

            mask = ~(terminated | truncated)

            buf.add_obs(state_input, action, action_logp, mask, reward)

            state = observation


        vector_states = torch.stack(buf.get_states()).to(device)
        values = critic(vector_states)
        buf.prep_data(values)

        states = buf.get_states()
        actions_logps = buf.get_actions_logps()
        masks = buf.get_masks()
        rewards = buf.get_rewards()
        returns = buf.get_returns()
        advantages = buf.get_advantages()

        num_eps = (args.rollout_steps * args.num_envs) - np.count_nonzero(masks)
        if masks[-1]: num_eps += 1

        avg_reward = rewards.sum().item() / num_eps
        avg_rewards.append(avg_reward)
        print(f'{episode+1}/{args.max_episodes}, {rewards.mean():.4f}, {rewards.max():.4f}, {num_eps}, {avg_reward:.4f}')

        start_training_time = time.time()

        trainer.train(args.num_epochs, buf)

        end_training_time = time.time()

        rollout_time = start_training_time - start_episode_time
        training_time = end_training_time - start_training_time
        total_time = end_training_time - start_episode_time
        step_rate = args.rollout_steps * args.num_envs / total_time
        frac_training = training_time / total_time

        wandb.log({
            'episode/advantages': advantages,
            'episode/values': values,
            'episode/returns': returns,
            'avg reward': avg_reward,
            'max reward': rewards.max().item(),
            'avg episode length': args.rollout_steps / num_eps,
            'timing/step rate': step_rate,
            'timing/training fraction': frac_training,
        })

    torch.save(actor.state_dict(), 'actor.pth')
    torch.save(critic.state_dict(), 'critic.pth')
