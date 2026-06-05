"""Registry of supported environments."""
from __future__ import annotations

import gymnasium as gym

from common.types import EnvMeta


REGISTRY: dict[str, EnvMeta] = {
    "CartPole-v1": EnvMeta(
        name="CartPole-v1",
        max_steps=500,
        success_threshold=475.0,
        act_type="discrete",
    ),
    "Acrobot-v1": EnvMeta(
        name="Acrobot-v1",
        max_steps=500,
        success_threshold=-100.0,
        act_type="discrete",
    ),
    "MountainCar-v0": EnvMeta(
        name="MountainCar-v0",
        max_steps=200,
        success_threshold=-110.0,
        act_type="discrete",
    ),
    "LunarLander-v3": EnvMeta(
        name="LunarLander-v3",
        max_steps=1000,
        success_threshold=200.0,
        act_type="discrete",
    ),
}


def make_env(name: str, seed: int) -> gym.Env:
    """Construct a Gymnasium env, seeded deterministically."""
    if name not in REGISTRY:
        raise KeyError(f"Unknown env: {name}. Available: {list(REGISTRY)}")
    env = gym.make(name)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env
