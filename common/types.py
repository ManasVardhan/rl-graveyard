"""Shared dataclasses used across packages."""
from dataclasses import dataclass, field
from typing import Literal


FailureMode = Literal[
    "alive",
    "nan_explosion",
    "exploration_collapse",
    "death_spiral",
    "stalling",
    "overshooting",
    "reward_hacking",
]


ActType = Literal["discrete", "continuous"]


@dataclass
class Trajectory:
    """All metrics recorded during a training run."""
    algo: str
    env: str
    seed: int
    total_steps: int
    episode_returns: list[float]
    episode_lengths: list[int]
    entropy_log: list[float]
    grad_norm_log: list[float]
    value_log: list[float]
    hparams: dict


@dataclass
class Verdict:
    """Output of the failure classifier."""
    failure_mode: FailureMode
    cause: str


@dataclass(frozen=True)
class EnvMeta:
    """Static metadata about an environment."""
    name: str
    max_steps: int
    success_threshold: float
    act_type: ActType
