"""Tests for the RewardTrap gridworld."""
import gymnasium as gym
import numpy as np
import pytest
from envs.reward_trap import RewardTrapEnv
from envs.registry import REGISTRY


def test_reward_trap_in_registry():
    assert "RewardTrap-v0" in REGISTRY


def test_action_space_is_discrete_4():
    env = RewardTrapEnv()
    assert isinstance(env.action_space, gym.spaces.Discrete)
    assert env.action_space.n == 4  # up, down, left, right


def test_reach_goal_terminates_with_reward_10():
    env = RewardTrapEnv()
    env.reset(seed=0)
    # Move right 4 times then down 4 times — should reach (4,4)
    for _ in range(4):
        env.step(3)  # right
    for _ in range(3):
        obs, r, term, trunc, info = env.step(1)  # down
        assert not term
    obs, r, term, trunc, info = env.step(1)  # final down
    assert term
    assert r > 9.0  # +10 minus tiny step penalty


def test_trap_tile_pays_per_step_and_doesnt_terminate():
    env = RewardTrapEnv()
    env.reset(seed=0)
    # Move right 2, down 2 to reach trap at (2,2)
    for _ in range(2):
        env.step(3)
    for _ in range(2):
        obs, r, term, trunc, info = env.step(1)
    # Now standing on trap
    obs, r, term, trunc, info = env.step(0)  # try to move up (will leave trap)
    # Move back onto trap
    obs, r, term, trunc, info = env.step(1)  # down — back on trap
    assert r > 0.4  # trap pays +0.5 minus step penalty
    assert not term


def test_truncates_at_max_steps():
    env = RewardTrapEnv()
    env.reset(seed=0)
    for i in range(500):
        obs, r, term, trunc, info = env.step(0)  # stay near origin by oscillating
        if term or trunc:
            break
    assert trunc or term  # must end one way or another


def test_obs_is_2d_position():
    env = RewardTrapEnv()
    obs, _ = env.reset(seed=0)
    assert obs.shape == (2,)
