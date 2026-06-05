"""Proximal Policy Optimization with GAE-λ and clipped surrogate.

Update flow:
  - Collect T steps of experience (T = rollout_length)
  - Compute GAE advantages
  - Run K epochs of minibatch SGD over the rollout
  - Each minibatch optimizes clip(r(θ), 1-ε, 1+ε) * A^GAE
"""
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from agents.a2c import ActorCriticNet  # Same architecture
from agents.base import Agent


def compute_gae(
    rewards: list[float],
    values: list[float],
    dones: list[bool],
    bootstrap_value: float,
    gamma: float,
    lam: float,
) -> tuple[list[float], list[float]]:
    """Generalized Advantage Estimation.

    Returns (advantages, returns) where returns = advantages + values.
    """
    advantages: list[float] = [0.0] * len(rewards)
    last_gae = 0.0
    for t in reversed(range(len(rewards))):
        nonterminal = 0.0 if dones[t] else 1.0
        next_value = bootstrap_value if t == len(rewards) - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        last_gae = delta + gamma * lam * nonterminal * last_gae
        advantages[t] = last_gae
    returns = [a + v for a, v in zip(advantages, values)]
    return advantages, returns


class PPOAgent(Agent):
    """PPO with clipped surrogate objective."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_eps: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        grad_clip: float = 0.5,
        k_epochs: int = 4,
        minibatch_size: int = 32,
    ):
        super().__init__(obs_dim, act_dim)
        self.net = ActorCriticNet(obs_dim, act_dim)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.grad_clip = grad_clip
        self.k_epochs = k_epochs
        self.minibatch_size = minibatch_size

    def act(self, obs: np.ndarray) -> tuple[int, dict[str, float]]:
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            logits, value = self.net(obs_t)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return int(action.item()), {
            "log_prob": float(dist.log_prob(action).item()),
            "value": float(value.item()),
            "entropy": float(dist.entropy().item()),
        }

    def update(self, batch: dict[str, Any]) -> dict[str, float]:
        obs = torch.as_tensor(np.array(batch["obs"]), dtype=torch.float32)
        actions = torch.as_tensor(batch["actions"], dtype=torch.int64)
        log_probs_old = torch.as_tensor(batch["log_probs_old"], dtype=torch.float32)

        advantages, returns = compute_gae(
            batch["rewards"],
            batch["values"],
            batch["dones"],
            batch["bootstrap_value"],
            self.gamma,
            self.lam,
        )
        advantages_t = torch.as_tensor(advantages, dtype=torch.float32)
        returns_t = torch.as_tensor(returns, dtype=torch.float32)
        # Normalize advantages
        if advantages_t.numel() > 1:
            advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        n = obs.shape[0]
        last_metrics = {}
        for _ in range(self.k_epochs):
            idx = torch.randperm(n)
            for start in range(0, n, self.minibatch_size):
                mb = idx[start : start + self.minibatch_size]
                mb_obs = obs[mb]
                mb_actions = actions[mb]
                mb_log_probs_old = log_probs_old[mb]
                mb_advantages = advantages_t[mb]
                mb_returns = returns_t[mb]

                logits, values = self.net(mb_obs)
                dist = Categorical(logits=logits)
                log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(log_probs - mb_log_probs_old)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values, mb_returns)
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.net.parameters(), self.grad_clip
                )
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = (mb_log_probs_old - log_probs).mean()

                last_metrics = {
                    "loss": float(loss.item()),
                    "grad_norm": float(grad_norm.item()),
                    "entropy": float(entropy.item()),
                    "kl_divergence": float(approx_kl.item()),
                }

        return last_metrics
