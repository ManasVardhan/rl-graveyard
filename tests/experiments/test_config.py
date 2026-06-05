"""Tests for ExperimentConfig."""
import pytest
from experiments.config import ExperimentConfig, build_agent, SUPPORTED_ALGOS


def test_supported_algos_match_spec():
    assert SUPPORTED_ALGOS == {"PPO", "A2C", "REINFORCE", "DQN", "DoubleDQN", "DuelingDQN"}


def test_config_defaults():
    cfg = ExperimentConfig(algo="PPO", env="CartPole-v1", seed=0)
    assert cfg.total_steps == 200_000
    assert cfg.hparams == {}


def test_build_agent_dispatches_correctly():
    cfg = ExperimentConfig(algo="PPO", env="CartPole-v1", seed=0)
    agent = build_agent(cfg, obs_dim=4, act_dim=2)
    from agents.ppo import PPOAgent
    assert isinstance(agent, PPOAgent)


def test_build_agent_double_dqn():
    cfg = ExperimentConfig(algo="DoubleDQN", env="CartPole-v1", seed=0)
    agent = build_agent(cfg, obs_dim=4, act_dim=2)
    from agents.dqn import DoubleDQNAgent
    assert isinstance(agent, DoubleDQNAgent)


def test_unknown_algo_raises():
    cfg = ExperimentConfig(algo="Nonsense", env="CartPole-v1", seed=0)
    with pytest.raises(ValueError):
        build_agent(cfg, obs_dim=4, act_dim=2)
