"""Monte-Carlo policy gradient (REINFORCE) with entropy bonus.

The simplest possible policy gradient: collect a full episode, compute
discounted returns, take a single gradient step. No value baseline in v1
(deliberately — REINFORCE without baseline is the canonical
high-variance failure case we want in the graveyard).
"""
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from agents.base import Agent


class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class REINFORCEAgent(Agent):
    """REINFORCE. On-policy. Updates once per episode."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        entropy_coef: float = 0.01,
        grad_clip: float = 0.5,
    ):
        super().__init__(obs_dim, act_dim)
        self.policy = PolicyNet(obs_dim, act_dim)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.grad_clip = grad_clip

    def act(self, obs: np.ndarray) -> tuple[int, dict[str, float]]:
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        logits = self.policy(obs_t)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return int(action.item()), {
            "log_prob": float(log_prob.item()),
            "value": 0.0,
            "entropy": float(entropy.item()),
            # Hidden tensor handles for the runner to pass into update()
            "_log_prob_tensor": log_prob,
            "_entropy_tensor": entropy,
        }

    def update(self, batch: dict[str, Any]) -> dict[str, float]:
        rewards: list[float] = batch["rewards"]
        log_probs: list[torch.Tensor] = batch["log_probs"]
        entropies: list[torch.Tensor] = batch["entropies"]

        # Discounted returns G_t = sum_{k>=0} gamma^k r_{t+k}
        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        # No normalization — bare REINFORCE without baseline is intentionally high-variance.
        # The unstable signal is the point: it's a canonical failure mode in the graveyard.
        returns_t = torch.tensor(returns, dtype=torch.float32)

        log_probs_t = torch.stack(log_probs)
        entropies_t = torch.stack(entropies)

        policy_loss = -(log_probs_t * returns_t).mean()
        entropy_bonus = entropies_t.mean()
        loss = policy_loss - self.entropy_coef * entropy_bonus

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
        self.optimizer.step()

        return {
            "loss": float(loss.item()),
            "grad_norm": float(grad_norm.item()),
            "entropy": float(entropy_bonus.item()),
        }
