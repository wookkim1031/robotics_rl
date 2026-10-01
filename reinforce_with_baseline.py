import os
import torch    
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch import nn as nn
from torch.optim import AdamW
from utils import plot_stats

import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="rgb_array")
obs, info = env.reset(seed=0)

#plt.imshow(env.render())
#plt.show()

dims = env.observation_space.shape[0]
actions = env.action_space.n

# State dimensions: [cart position, cart velocity, pole angle, pole angular velocity]
# Actions (0,1): 0: pushes the cart to the left, 1: pushes the cart to the right   
print(f"State dimensions: {dims}. Actions: {actions}")
#  [ 0.03147498 0.01224748 0.00424683 -0.0458429 ]
# - The cart is slightly to the right of the center (0.03147498)
# - The cart is moving slightly to the right (0.01224748)
# - The pole is slightly tilted to the right (0.00424683)
# - The pole is rotating slightly to the left (-0.0458429)
print(f"Sample state: {env.reset()}")

num_envs = 8

class PreprocessEnv:
    """Converts a vectorized Gymnasium env's outputs to PyTorch tensors."""
    def __init__(self, env):
        self.env = env

    def reset(self, **kwargs):
        state, info = self.env.reset(**kwargs)
        return torch.from_numpy(state).float()                    # (num_envs, 4)

    def step(self, actions):
        actions = actions.squeeze(-1).numpy()                     # (num_envs,)
        next_state, reward, terminated, truncated, info = self.env.step(actions)
        done = terminated | truncated
        next_state = torch.from_numpy(next_state).float()         # (num_envs, 4)
        reward = torch.from_numpy(reward).float().unsqueeze(1)    # (num_envs, 1)
        done = torch.from_numpy(done).unsqueeze(1)                # (num_envs, 1)
        return next_state, reward, done, info
    
policy = nn.Sequential(
    nn.Linear(dims, 128),  # Input layer (state dims to 128 units)
    nn.ReLU(),            # Activation function
    nn.Linear(128, 64),   # Hidden layer (128 to 64 units)
    nn.ReLU(),            # Activation function
    nn.Linear(64, actions), # Output layer (64 units to action space)
    nn.Softmax(dim=-1))   # Softmax for action probabilities

value_net = nn.Sequential(
    nn.Linear(dims, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 1))

def reinforce_baseline(policy, value_net, episodes, alpha=1e-3, value_lr=1e-3, gamma=0.99):
    # Optimizer for policy
    # optim stores lr = 0.001, weight_decay: 0.01, betas: (0.9, 0.999) 
    optim = AdamW(policy.parameters(), lr=alpha)
    value_optim = AdamW(value_net.parameters(), lr=value_lr)
    stats = {'PG Loss': [], 'Value Loss': [], 'Returns': []}
    vec_env = gym.vector.SyncVectorEnv(
        [lambda: gym.make("CartPole-v1") for _ in range(num_envs)]
    )
    parallel_env = PreprocessEnv(vec_env)

    for episode in tqdm(range(1, episodes + 1)): # Loop over episodes 
        state = parallel_env.reset()
        # Done flag
        done_b = torch.zeros((num_envs, 1), dtype=torch.bool)
        transitions = []
        # Episode return
        ep_return = torch.zeros((num_envs, 1))

        while not done_b.all():
            # Sample action
            action = policy(state).multinomial(1).detach()
            # Step env
            next_state, reward, done, _ = parallel_env.step(action)
            transitions.append([state, action, ~done_b * reward, ~done_b])
            ep_return += ~done_b * reward
            done_b |= done
            state = next_state

        states  = torch.stack([s for s, _, _, _ in transitions])
        acts    = torch.stack([a for _, a, _, _ in transitions])
        rewards = torch.stack([r for _, _, r, _ in transitions])
        valid   = torch.stack([m for _, _, _, m in transitions])

        returns = torch.zeros_like(rewards)
        G = torch.zeros((num_envs, 1)) # Initialize return
        for t in reversed(range(len(transitions))):
            G = rewards[t] + gamma * G # Calculate discounted return
            returns[t] = G

        # Critic: regress V(s) toward the actual returns
        values = value_net(states) # (T, N, 1)
        value_loss = ((values - returns) ** 2)[valid].mean()
        value_optim.zero_grad()
        value_loss.backward()
        value_optim.step()

        # Actor: advantage = actual return - predicted return 
        advantages = returns - values.detach()
        advantages = (advantages - advantages[valid].mean()) / (advantages[valid].std() + 1e-8)

        probs = policy(states)
        log_probs = torch.log(probs + 1e-6)
        # acts has the shpae of (T, N, 1) T: timestep, N: environment
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

stats = reinforce_baseline(policy,value_net, 500)

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