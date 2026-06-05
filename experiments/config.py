"""Experiment configuration and agent factory."""
from dataclasses import dataclass, field

from agents.a2c import A2CAgent
from agents.base import Agent
from agents.dqn import DoubleDQNAgent, DQNAgent, DuelingDQNAgent
from agents.ppo import PPOAgent
from agents.reinforce import REINFORCEAgent


SUPPORTED_ALGOS = {"PPO", "A2C", "REINFORCE", "DQN", "DoubleDQN", "DuelingDQN"}


@dataclass
class ExperimentConfig:
    algo: str
    env: str
    seed: int
    total_steps: int = 200_000
    hparams: dict = field(default_factory=dict)


_REGISTRY = {
    "PPO": PPOAgent,
    "A2C": A2CAgent,
    "REINFORCE": REINFORCEAgent,
    "DQN": DQNAgent,
    "DoubleDQN": DoubleDQNAgent,
    "DuelingDQN": DuelingDQNAgent,
}


def build_agent(cfg: ExperimentConfig, obs_dim: int, act_dim: int) -> Agent:
    """Construct the agent for a config."""
    if cfg.algo not in _REGISTRY:
        raise ValueError(f"Unknown algo: {cfg.algo}. Supported: {SUPPORTED_ALGOS}")
    cls = _REGISTRY[cfg.algo]
    return cls(obs_dim=obs_dim, act_dim=act_dim, **cfg.hparams)
