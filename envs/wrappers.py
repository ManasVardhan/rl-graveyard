"""Logging wrappers for environments."""
import gymnasium as gym


class LoggingWrapper(gym.Wrapper):
    """Record episode-level statistics: returns and lengths.

    Per-step metrics that depend on the agent (entropy, grad_norm) are
    recorded by the runner, not here — the env doesn't know about them.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.episode_returns: list[float] = []
        self.episode_lengths: list[int] = []
        self._current_return = 0.0
        self._current_length = 0

    def reset(self, *, seed=None, options=None):
        self._current_return = 0.0
        self._current_length = 0
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._current_return += float(reward)
        self._current_length += 1
        if terminated or truncated:
            self.episode_returns.append(self._current_return)
            self.episode_lengths.append(self._current_length)
        return obs, reward, terminated, truncated, info
