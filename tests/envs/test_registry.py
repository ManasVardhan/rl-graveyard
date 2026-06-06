"""Tests for envs.registry."""
import pytest
from envs.registry import REGISTRY, make_env


def test_classic_envs_present():
    for name in ["CartPole-v1", "Acrobot-v1", "MountainCar-v0", "LunarLander-v3"]:
        assert name in REGISTRY


def test_make_env_returns_seeded_env():
    env = make_env("CartPole-v1", seed=42)
    obs1, _ = env.reset()
    env2 = make_env("CartPole-v1", seed=42)
    obs2, _ = env2.reset()
    assert (obs1 == obs2).all()
    env.close()
    env2.close()


def test_unknown_env_raises():
    with pytest.raises(KeyError):
        make_env("Nonexistent-v0", seed=0)


def test_metadata_has_act_type():
    assert REGISTRY["CartPole-v1"].act_type == "discrete"
