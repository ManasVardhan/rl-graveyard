"""Tests for envs.wrappers.LoggingWrapper."""
import numpy as np
from envs.registry import make_env
from envs.wrappers import LoggingWrapper


def test_records_episode_returns():
    env = LoggingWrapper(make_env("CartPole-v1", seed=0))
    env.reset(seed=0)
    for _ in range(50):
        obs, reward, term, trunc, info = env.step(env.action_space.sample())
        if term or trunc:
            env.reset()
    assert len(env.episode_returns) >= 1
    assert all(isinstance(r, float) for r in env.episode_returns)


def test_records_episode_lengths_match_returns():
    env = LoggingWrapper(make_env("CartPole-v1", seed=0))
    env.reset(seed=0)
    for _ in range(50):
        obs, reward, term, trunc, info = env.step(env.action_space.sample())
        if term or trunc:
            env.reset()
    assert len(env.episode_returns) == len(env.episode_lengths)
