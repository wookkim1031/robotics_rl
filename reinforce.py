import os
import torch    
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

def reinforce(policy, episodes, alpha=1e-4, gamma=0.99):
    # Optimizer for policy
    # optim stores lr = 0.0001, weight_decay: 0.01, betas: (0.9, 0.999) 
    optim = AdamW(policy.parameters(), lr=alpha)
    stats = {'PG Loss': [], 'Returns': []} # Store training stats
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
            transitions.append([state, action, ~done_b * reward])
            ep_return += ~done_b * reward
            done_b |= done
            state = next_state

        G = torch.zeros((num_envs, 1)) # Initialize return
        for t, (state_t, action_t, reward_t) in reversed(list(enumerate(transitions))): # Calculate returns
            G = reward_t + gamma * G # Calculate discounted return
            probs_t = policy(state_t) # Get action probabilities
            log_probs_t = torch.log(probs_t + 1e-6) # Log probabilities
            action_log_prob_t = log_probs_t.gather(1, action_t) # Log prob of action

            entropy_t = - torch.sum(probs_t * log_probs_t, dim=-1, keepdim=True) # Calculate entropy
            gamma_t = gamma ** t # Discount factor
            pg_loss_t = - gamma_t * action_log_prob_t * G # Policy gradient loss
            total_loss_t = (pg_loss_t - 0.01 * entropy_t).mean() # Total loss

            policy.zero_grad() # Reset gradients

            # Compute ∂L/∂θ for every weights θ in the network using backpropagation
            # the results are stored in each parameter's .grad field
            total_loss_t.backward() # Calculate gradients

            # step uses those gradients to change the weights
            # plain gradient descents would be θ ← θ − α · ∂L/∂θ
            optim.step() # Update policy parameters

        stats['PG Loss'].append(total_loss_t.item()) # Store loss
        stats['Returns'].append(ep_return.mean().item()) # Store return

    return stats # Return training stats

stats = reinforce(policy, 500, alpha=1e-3)

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