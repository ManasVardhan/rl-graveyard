"""DQN family: vanilla DQN, Double DQN, Dueling DQN.

All three share replay buffer + target network + ε-greedy. They differ
in:
  - DQN: target = r + γ * max_a' Q_target(s', a')
  - Double DQN: target = r + γ * Q_target(s', argmax_a' Q_online(s', a'))
  - Dueling DQN: Q = V(s) + (A(s,a) - mean_a A(s,a)); other math identical
"""
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.base import Agent


class QNet(nn.Module):
    """Standard Q-network."""

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


class DuelingQNet(nn.Module):
    """Dueling architecture: shared trunk, separate V and A heads."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.value_head = nn.Linear(hidden, 1)
        self.advantage_head = nn.Linear(hidden, act_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.trunk(x)
        v = self.value_head(h)
        a = self.advantage_head(h)
        return v + a - a.mean(dim=-1, keepdim=True)


class DQNAgent(Agent):
    """Vanilla DQN with replay + target net + ε-greedy."""

    NET_CLASS = QNet

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        lr: float = 5e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 50_000,
        target_sync_every: int = 1000,
        grad_clip: float = 10.0,
    ):
        super().__init__(obs_dim, act_dim)
        self.q_net = self.NET_CLASS(obs_dim, act_dim)
        self.target_net = self.NET_CLASS(obs_dim, act_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())
        for p in self.target_net.parameters():
            p.requires_grad = False
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.lr = lr
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.target_sync_every = target_sync_every
        self.grad_clip = grad_clip
        self.epsilon = epsilon_start
        self._step = 0

    def step_epsilon(self, step: int) -> None:
        """Linearly anneal ε from start to end over epsilon_decay_steps."""
        frac = min(step / max(self.epsilon_decay_steps, 1), 1.0)
        self.epsilon = self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def sync_target(self) -> None:
        self.target_net.load_state_dict(self.q_net.state_dict())

    def act(self, obs: np.ndarray) -> tuple[int, dict[str, float]]:
        if np.random.random() < self.epsilon:
            action = int(np.random.randint(0, self.act_dim))
            return action, {"log_prob": 0.0, "value": 0.0, "entropy": float(np.log(self.act_dim))}
        with torch.no_grad():
            q = self.q_net(torch.as_tensor(obs, dtype=torch.float32))
        action = int(q.argmax().item())
        return action, {"log_prob": 0.0, "value": float(q.max().item()), "entropy": 0.0}

    def _td_target(self, next_obs: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        """Standard DQN target. Subclasses override for variants."""
        with torch.no_grad():
            next_q = self.target_net(next_obs).max(dim=1).values
            return rewards + self.gamma * next_q * (1.0 - dones)

    def update(self, batch: dict[str, Any]) -> dict[str, float]:
        obs = torch.as_tensor(batch["obs"], dtype=torch.float32)
        actions = torch.as_tensor(batch["actions"], dtype=torch.int64)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32)
        next_obs = torch.as_tensor(batch["next_obs"], dtype=torch.float32)
        dones = torch.as_tensor(batch["dones"], dtype=torch.float32)

        q_values = self.q_net(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        td_target = self._td_target(next_obs, rewards, dones)
        loss = F.smooth_l1_loss(q_values, td_target)

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.grad_clip)
        self.optimizer.step()

        self._step += 1
        if self._step % self.target_sync_every == 0:
            self.sync_target()

        return {
            "loss": float(loss.item()),
            "grad_norm": float(grad_norm.item()),
            "entropy": 0.0,
        }


class DoubleDQNAgent(DQNAgent):
    """Double DQN: decouples action selection (online net) from evaluation (target net).

    Reduces overestimation bias compared to vanilla DQN.
    """

    def _td_target(self, next_obs: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            online_argmax = self.q_net(next_obs).argmax(dim=1)
            next_q = self.target_net(next_obs).gather(1, online_argmax.unsqueeze(1)).squeeze(1)
            return rewards + self.gamma * next_q * (1.0 - dones)


class DuelingDQNAgent(DQNAgent):
    """Dueling DQN: same training math as DQN, different network architecture."""

    NET_CLASS = DuelingQNet
