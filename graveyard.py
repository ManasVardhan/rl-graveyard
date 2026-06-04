"""
RL Graveyard -- Train one agent and classify its death.
Minimal proof-of-concept.
"""

import gymnasium as gym
import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
# import wandb  # optional, install if needed
from dataclasses import dataclass
from typing import Literal

FailureMode = Literal["alive", "reward_hacking", "exploration_collapse", "death_spiral", "nan_explosion", "stalling", "overshooting"]

@dataclass
class Autopsy:
    algorithm: str
    environment: str
    steps: int
    failure_mode: FailureMode
    episode_returns: list[float]
    entropy_log: list[float]
    grad_norm_log: list[float]
    cause: str

class Agent(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.policy = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, act_dim)
        )
        self.value = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        logits = self.policy(x)
        value = self.value(x)
        return logits, value

def diagnose_death(episode_returns, entropy_log, grad_norm_log, max_steps=100000) -> tuple[FailureMode, str]:
    """Classify how an agent died."""
    
    # NaN explosion
    if any(np.isnan(r) for r in episode_returns[-10:]):
        return "nan_explosion", "Loss became NaN in final 10 episodes"
    
    # Exploration collapse: entropy flatlined
    if len(entropy_log) > 100 and np.mean(entropy_log[-50:]) < 0.1:
        return "exploration_collapse", f"Entropy collapsed to {np.mean(entropy_log[-50:]):.3f}"
    
    # Death spiral: returns crashing
    if len(episode_returns) > 50:
        recent = np.mean(episode_returns[-10:])
        past = np.mean(episode_returns[-50:-10])
        if recent < past * 0.5:
            return "death_spiral", f"Returns dropped {((past-recent)/past)*100:.0f}%"
    
    # Stalling: no improvement for 500 episodes
    if len(episode_returns) > 500:
        window = episode_returns[-500:]
        if np.std(window) / (abs(np.mean(window)) + 1e-6) < 0.05:
            return "stalling", "Flatlined for 500 episodes"
    
    # Overshooting: high variance oscillation
    if len(episode_returns) > 50:
        window = episode_returns[-50:]
        cv = np.std(window) / (abs(np.mean(window)) + 1e-6)
        if cv > 2.0:
            return "overshooting", f"Return CV = {cv:.1f}, oscillating wildly"
    
    # Reward hacking: sudden massive spike then crash
    if len(episode_returns) > 20:
        max_recent = max(episode_returns[-20:])
        baseline = np.mean(episode_returns[:-20]) if len(episode_returns) > 20 else 0
        if max_recent > baseline * 5 + 100:
            return "reward_hacking", f"Sudden {max_recent:.0f}x spike above baseline"
    
    return "alive", "Agent survived"

def train_ppo(env_name="CartPole-v1", max_steps=100000, seed=42):
    """Train a PPO agent and autopsy the result."""
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    
    agent = Agent(obs_dim, act_dim)
    optimizer = torch.optim.Adam(agent.parameters(), lr=3e-4)
    
    episode_returns = []
    entropy_log = []
    grad_norm_log = []
    
    obs, _ = env.reset(seed=seed)
    obs = torch.FloatTensor(obs)
    
    episode_return = 0
    step = 0
    
    while step < max_steps:
        logits, value = agent(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        next_obs, reward, terminated, truncated, _ = env.step(action.item())
        next_obs = torch.FloatTensor(next_obs)
        
        episode_return += reward
        step += 1
        
        # Simple one-step update (not full PPO, just enough to train)
        _, next_value = agent(next_obs)
        td_target = reward + 0.99 * next_value.detach()
        advantage = td_target - value
        
        policy_loss = -log_prob * advantage.detach()
        value_loss = advantage.pow(2)
        entropy = dist.entropy()
        
        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
        
        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
        grad_norm_log.append(grad_norm.item())
        optimizer.step()
        
        entropy_log.append(entropy.item())
        
        obs = next_obs
        
        if terminated or truncated:
            episode_returns.append(episode_return)
            episode_return = 0
            obs, _ = env.reset()
            obs = torch.FloatTensor(obs)
    
    env.close()
    
    failure_mode, cause = diagnose_death(episode_returns, entropy_log, grad_norm_log, max_steps)
    
    autopsy = Autopsy(
        algorithm="PPO",
        environment=env_name,
        steps=step,
        failure_mode=failure_mode,
        episode_returns=episode_returns,
        entropy_log=entropy_log,
        grad_norm_log=grad_norm_log,
        cause=cause
    )
    
    return autopsy

def print_death_certificate(autopsy: Autopsy):
    """Generate a shareable death certificate."""
    
    status = "DECEASED" if autopsy.failure_mode != "alive" else "SURVIVED"
    emoji = "☠️" if status == "DECEASED" else "✅"
    
    print(f"""
{'='*60}
           DEATH CERTIFICATE {emoji}
{'='*60}

Agent:     {autopsy.algorithm}
Environment: {autopsy.environment}
Status:    {status}
Steps:     {autopsy.steps:,}

CAUSE OF DEATH: {autopsy.failure_mode.upper()}
AUTOPSY: {autopsy.cause}

Final 10 episode returns: {[f"{r:.1f}" for r in autopsy.episode_returns[-10:]]}
Final entropy: {np.mean(autopsy.entropy_log[-10:]):.3f}
{'='*60}
    """)
    
    return status == "DECEASED"

if __name__ == "__main__":
    # Run one agent
    print("Training PPO on CartPole-v1...")
    autopsy = train_ppo("CartPole-v1", max_steps=50000, seed=42)
    died = print_death_certificate(autopsy)
    
    if died:
        print(f"\nAgent died from: {autopsy.failure_mode}")
    else:
        print("\nAgent survived!")
    
    # Quick stat
    print(f"\nEpisodes completed: {len(autopsy.episode_returns)}")
    print(f"Mean final return: {np.mean(autopsy.episode_returns[-20:]):.1f}")
