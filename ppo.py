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

def ppo(policy, value_net, iterations=150, rollout_len=128, epochs=4, num_minibatches=4,
        lr=3e-4, gamma=0.99, lam=0.95, clip_eps=0.2, ent_coef=0.01):
    optim = AdamW(policy.parameters(), lr=lr)
    value_optim = AdamW(value_net.parameters(), lr=lr)
    vec_env = gym.vector.SyncVectorEnv(
        [lambda: gym.make("CartPole-v1") for _ in range(num_envs)]
    )
    parallel_env = PreprocessEnv(vec_env)

    stats = {'PG Loss': [], 'Value Loss': [], 'Approx KL': [], 'Returns': []}

    # the environments are reset ONCE; Episodes run on across rollouts
    state = parallel_env.reset(seed=0)
    prev_done = torch.zeros((num_envs, 1), dtype=torch.bool)
    running_return = torch.zeros((num_envs, 1))
    last_mean_return = 0.0

    for iteration in tqdm(range(iterations)):
        # 1. Collect a fied length rollout with the current policy 
        buf = {k: [] for k in ['states', 'actions', 'old_log_probs', 'rewards',
                               'next_states', 'terminated', 'dones', 'valid']}
        finished_returns = []

        for _ in range(rollout_len):
            with torch.no_grad():
                probs = policy(state)
                action = probs.multinomial(1)
                old_log_prob = torch.log(probs.gather(-1, action) + 1e-6)
 
            next_state, reward, terminated, done, _ = parallel_env.step(action)
 
            # After an env finishes, its next step() is an autoreset step:
            # the action is ignored and the transition is not real -> mask it out.
            valid = ~prev_done
 
            buf['states'].append(state)
            buf['actions'].append(action)
            buf['old_log_probs'].append(old_log_prob)
            buf['rewards'].append(reward)
            buf['next_states'].append(next_state)
            buf['terminated'].append(terminated)
            buf['dones'].append(done)
            buf['valid'].append(valid)
 
            # Logging: episode returns of environments that just finished
            running_return += valid * reward
            if done.any():
                finished_returns.extend(running_return[done].tolist())
                running_return[done] = 0.0
 
            prev_done = done
            state = next_state
 
        states        = torch.stack(buf['states'])         # (T, N, 4)
        acts          = torch.stack(buf['actions'])        # (T, N, 1)
        old_log_probs = torch.stack(buf['old_log_probs'])  # (T, N, 1)
        rewards       = torch.stack(buf['rewards'])        # (T, N, 1)
        next_states   = torch.stack(buf['next_states'])    # (T, N, 4)
        terminated    = torch.stack(buf['terminated'])     # (T, N, 1)
        dones         = torch.stack(buf['dones'])          # (T, N, 1)
        valid         = torch.stack(buf['valid'])          # (T, N, 1)

        # ---------- 2. GAE, exactly as in your a2c_gae, computed once per rollout ----------
        with torch.no_grad():
            values = value_net(states)
            next_values = value_net(next_states)
            deltas = rewards + gamma * next_values * (~terminated) - values
 
            advantages = torch.zeros_like(deltas)
            gae = torch.zeros((num_envs, 1))
            for t in reversed(range(rollout_len)):
                gae = deltas[t] + gamma * lam * (~dones[t]) * gae
                advantages[t] = gae
            value_targets = advantages + values           # lambda-returns

        # ---------- 3. Flatten the valid transitions into one batch ----------
        mask = valid.squeeze(-1)                          # (T, N)
        b_states        = states[mask]                    # (B, 4)
        b_actions       = acts[mask]                      # (B, 1)
        b_old_log_probs = old_log_probs[mask]             # (B, 1)
        b_advantages    = advantages[mask]                # (B, 1)
        b_targets       = value_targets[mask]             # (B, 1)
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
 
        # ---------- 4. Several epochs of minibatch updates on the SAME batch ----------
        # We use minibatch to update to get more useful gradient steps out of the same data
        # Collecting experience is the expensive part of RL, so PPO wants to learn as much as possible from each rollout
        B = b_states.shape[0]
        mb_size = max(1, B // num_minibatches)

        for epoch in range(epochs):
            perm = torch.randperm(B)
            for start in range(0, B, mb_size):
                idx = perm[start:start + mb_size]
 
                # Actor: clipped surrogate objective
                probs = policy(b_states[idx])
                log_probs = torch.log(probs + 1e-6)
                new_log_probs = log_probs.gather(-1, b_actions[idx])
                entropy = -torch.sum(probs * log_probs, dim=-1, keepdim=True)

                # rₜ(θ) = π_θ(aₜ|sₜ) / π_old(aₜ|sₜ)
                log_ratio = new_log_probs - b_old_log_probs[idx]
                ratio = torch.exp(log_ratio)              # pi_new(a|s) / pi_old(a|s)
                surr1 = ratio * b_advantages[idx]
                surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * b_advantages[idx]
                # L = −E[ min( rₜ · Âₜ, clip(rₜ, 1−ε, 1+ε) · Âₜ )]
                pg_loss = -(torch.min(surr1, surr2) + ent_coef * entropy).mean()
 
                optim.zero_grad()
                pg_loss.backward()
                optim.step()
 
                # Critic: regress V(s) toward the lambda-returns
                value_loss = ((value_net(b_states[idx]) - b_targets[idx]) ** 2).mean()
                value_optim.zero_grad()
                value_loss.backward()
                value_optim.step()
 
                # How far the policy has moved from the old one (diagnostic only)
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean()
 
        if finished_returns:
            last_mean_return = sum(finished_returns) / len(finished_returns)
 
        stats['PG Loss'].append(pg_loss.item())
        stats['Value Loss'].append(value_loss.item())
        stats['Approx KL'].append(approx_kl.item())
        stats['Returns'].append(last_mean_return)
 
    return stats

stats = ppo(policy, value_net)
plot_stats(stats)
 
eval_env = gym.make("CartPole-v1", render_mode="human")
obs, info = eval_env.reset()
done = False
while not done:
    with torch.no_grad():
        probs = policy(torch.from_numpy(obs).float())
    action = probs.argmax().item()
    obs, reward, terminated, truncated, info = eval_env.step(action)
    done = terminated or truncated
eval_env.close()