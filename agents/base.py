"""Agent ABC. All algorithms implement this interface."""
from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Agent(ABC):
    """Base class for all RL agents.

    Each algorithm has its own training loop (on-policy vs off-policy
    collection patterns differ too much for a shared trainer), but they
    all expose the same act/update interface so the runner can drive them
    uniformly.
    """

    def __init__(self, obs_dim: int, act_dim: int):
        self.obs_dim = obs_dim
        self.act_dim = act_dim

    @abstractmethod
    def act(self, obs: np.ndarray) -> tuple[int, dict[str, float]]:
        """Pick an action.

        Returns:
            (action, info) where info contains at minimum:
              - log_prob: float — log π(a|s), or 0.0 for off-policy ε-greedy
              - value: float — V(s), or 0.0 if no value head
              - entropy: float — H(π(·|s))
        """

    @abstractmethod
    def update(self, batch: Any) -> dict[str, float]:
        """Update the agent. Batch shape is algorithm-specific.

        Returns metrics dict with at minimum: loss, grad_norm, entropy.
        """
