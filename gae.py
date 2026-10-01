# a2c is just extension of reinforce TD target instead of reward

import torch
from tqdm import tqdm
from torch import nn as nn
from torch.optim import AdamW
import gymnasium as gym
from utils import plot_stats

env = gym.make("CartPole-v1", render_mode="rgb_array")
obs, info = env.reset(seed=0)

dims = env.observation_space.shape[0]
actions = env.action_space.n

num_envs = 8

class PreprocessEnv: 
    def __init__(self, env):
        self.env = env

    def reset(self, **kwargs):
        state, info = self.env.reset(**kwargs)
        return torch.from_numpy(state).float()

    def step(self, actions):
        actions = actions.squeeze(-1).numpy()
        next_state, reward, terminated, truncated, info = self.env.step(actions)
        done = terminated | truncated 
        next_state = torch.from_numpy(next_state).float()
        reward = torch.from_numpy(reward).float().unsqueeze(1)
        terminated = torch.from_numpy(terminated).unsqueeze(1)
        done = torch.from_numpy(done).unsqueeze(1)
        return next_state, reward, terminated, done, info

policy = nn.Sequential(
    nn.Linear(dims, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, actions),
    nn.Softmax(dim=-1))

value_net = nn.Sequential(
    nn.Linear(dims, 128),
    nn.ReLU(),
    nn.Linear(128,64),
    nn.ReLU(),
    nn.Linear(64, 1)
)

def a2c(policy, value_net, episodes, alpha=1e-3, value_lr=1e-3, gamma=0.99, lam=0.95):
    optim = AdamW(policy.parameters(), lr=alpha)
    value_optim = AdamW(value_net.parameters(), lr=value_lr)
    vec_env = gym.vector.SyncVectorEnv(
        [lambda: gym.make("CartPole-v1") for _ in range(num_envs)]
    )
    parallel_env = PreprocessEnv(vec_env)
    stats = {'PG Loss': [], 'Value Loss': [], 'Returns': []}

    for episode in tqdm(range(1, episodes + 1)):
        state = parallel_env.reset()
        done_b = torch.zeros((num_envs, 1), dtype=torch.bool)
        ep_return = torch.zeros((num_envs, 1))
        transitions = []
        while not done_b.all():
            action = policy(state).multinomial(1).detach()
            next_state, reward, terminated, done, _ = parallel_env.step(action)
            transitions.append([state, action, ~done_b * reward, ~done_b, next_state, terminated, done])

            ep_return += ~done_b *reward
            done_b |= done
            state = next_state

        states      = torch.stack([tr[0] for tr in transitions])   # (T, N, 4)
        acts        = torch.stack([tr[1] for tr in transitions])   # (T, N, 1)
        rewards     = torch.stack([tr[2] for tr in transitions])   # (T, N, 1)
        valid       = torch.stack([tr[3] for tr in transitions])   # (T, N, 1)
        next_states = torch.stack([tr[4] for tr in transitions])   # (T, N, 4)
        terminated  = torch.stack([tr[5] for tr in transitions])   # (T, N, 1)

        dones = torch.stack([tr[6] for tr in transitions])

        # computed backward with a recursion just like your return loop:
        # Âₜ = δₜ + γλ · Âₜ₊₁
        # lambda = 0, exactly the A2C
        # lambda 1, the sum telescopes to G_t - V(s_t), exactly REINFORCE with baseline
        # λ ≈ 0.95 (the usual default, also in rsl_rl): mostly real rewards over the next ~20 steps,
        values = value_net(states)
        with torch.no_grad():
            next_values = value_net(next_states)
        
        deltas = rewards + gamma * next_values * (~terminated) - values.detach()

        # GAE: backward recursion, cut at episode ends
        advantages = torch.zeros_like(deltas)
        gae = torch.zeros((num_envs, 1))
        for t in reversed(range(len(transitions))):
            gae = deltas[t] + gamma * lam * (~dones[t]) * gae
            advantages[t] = gae

        # Critic: the lambda-return
        value_targets = advantages + values.detach()
        value_loss = torch.sum(((values - value_targets) ** 2)[valid]) / valid.sum()
        value_optim.zero_grad()
        value_loss.backward()
        value_optim.step()

        # Actor: advantage = GAE advantage 
        advantages = (advantages - advantages[valid].mean()) / (advantages[valid].std() + 1e-8)

        probs = policy(states)
        log_probs = torch.log(probs + 1e-6)
        action_log_probs = log_probs.gather(2, acts)
        entropy = -(probs * log_probs).sum(-1, keepdim=True)
        pg_loss = -(action_log_probs * advantages + 0.01 * entropy)[valid].mean()

        optim.zero_grad()
        pg_loss.backward()
        optim.step()
        
        stats['PG Loss'].append(pg_loss.item())
        stats['Value Loss'].append(value_loss.item())
        stats['Returns'].append(ep_return.mean().item())

    return stats

stats = a2c(policy, value_net, 1000)

plot_stats(stats)

eval_env = gym.make("CartPole-v1", render_mode="human")
obs, info = eval_env.reset()
done = False
while not done:
    with torch.no_grad():
        probs = policy(torch.from_numpy(obs).float())
    action = probs.argmax().item()          # greedy action for evaluation
    obs, reward, terminated, truncated, info = eval_env.step(action)
    done = terminated or truncated

eval_env.close()