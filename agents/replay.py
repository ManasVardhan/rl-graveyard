"""Replay buffer for off-policy algorithms (DQN family)."""
import numpy as np


class ReplayBuffer:
    """Fixed-size circular buffer of transitions."""

    def __init__(self, capacity: int, obs_dim: int):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self._idx = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        self.obs[self._idx] = obs
        self.actions[self._idx] = action
        self.rewards[self._idx] = reward
        self.next_obs[self._idx] = next_obs
        self.dones[self._idx] = 1.0 if done else 0.0
        self._idx = (self._idx + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict:
        if self._size == 0:
            raise ValueError("Cannot sample from empty buffer")
        idx = np.random.randint(0, self._size, size=batch_size)
        return {
            "obs": self.obs[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "next_obs": self.next_obs[idx],
            "dones": self.dones[idx],
        }
