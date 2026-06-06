"""Synchronous Advantage Actor-Critic with n-step returns.

Update flow:
  - Collect n=5 steps of experience
  - Compute n-step returns bootstrapped with V(s_{t+n})
  - Advantages = returns - V(s_t)
  - Joint loss: policy gradient + value MSE - entropy bonus
"""
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from agents.base import Agent


class ActorCriticNet(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden, act_dim)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.shared(x)
        return self.policy_head(h), self.value_head(h).squeeze(-1)


class A2CAgent(Agent):
    """A2C with shared body. On-policy."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        lr: float = 7e-4,
        gamma: float = 0.99,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        grad_clip: float = 0.5,
        n_steps: int = 5,
    ):
        super().__init__(obs_dim, act_dim)
        self.net = ActorCriticNet(obs_dim, act_dim)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.lr = lr
        self.gamma = gamma
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.grad_clip = grad_clip
        self.n_steps = n_steps

    def act(self, obs: np.ndarray) -> tuple[int, dict[str, float]]:
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        logits, value = self.net(obs_t)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return int(action.item()), {
            "log_prob": float(dist.log_prob(action).item()),
            "value": float(value.item()),
            "entropy": float(dist.entropy().item()),
        }

    def update(self, batch: dict[str, Any]) -> dict[str, float]:
        obs_list: list[np.ndarray] = batch["obs"]
        actions: list[int] = batch["actions"]
        rewards: list[float] = batch["rewards"]
        dones: list[bool] = batch["dones"]
        bootstrap_value: float = batch["bootstrap_value"]

        obs_t = torch.as_tensor(np.array(obs_list), dtype=torch.float32)
        actions_t = torch.as_tensor(actions, dtype=torch.int64)

        # n-step returns
        returns = []
        R = bootstrap_value
        for r, done in zip(reversed(rewards), reversed(dones)):
            R = r + self.gamma * R * (0.0 if done else 1.0)
            returns.insert(0, R)
        returns_t = torch.tensor(returns, dtype=torch.float32)

        logits, values = self.net(obs_t)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions_t)
        entropy = dist.entropy().mean()

        advantages = returns_t - values.detach()
        policy_loss = -(log_probs * advantages).mean()
        value_loss = F.mse_loss(values, returns_t)
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)
        self.optimizer.step()

        return {
            "loss": float(loss.item()),
            "grad_norm": float(grad_norm.item()),
            "entropy": float(entropy.item()),
        }
