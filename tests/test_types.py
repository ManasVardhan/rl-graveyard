"""Tests for common.types."""
from common.types import Trajectory, Verdict, EnvMeta


def test_trajectory_construction():
    traj = Trajectory(
        algo="PPO",
        env="CartPole-v1",
        seed=0,
        total_steps=1000,
        episode_returns=[10.0, 20.0],
        episode_lengths=[100, 200],
        entropy_log=[0.5, 0.4],
        grad_norm_log=[1.0, 0.9],
        value_log=[5.0, 6.0],
        hparams={"lr": 3e-4},
    )
    assert traj.algo == "PPO"
    assert len(traj.episode_returns) == 2


def test_verdict_construction():
    v = Verdict(failure_mode="exploration_collapse", cause="entropy < 0.1")
    assert v.failure_mode == "exploration_collapse"


def test_env_meta_freezes():
    import dataclasses
    meta = EnvMeta(
        name="CartPole-v1",
        max_steps=500,
        success_threshold=475.0,
        act_type="discrete",
    )
    assert dataclasses.is_dataclass(meta)
