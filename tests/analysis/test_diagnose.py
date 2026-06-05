"""Tests for the per-failure-mode detectors and diagnose()."""
import numpy as np
from analysis.diagnose import detect_nan_explosion
from common.types import Trajectory


def _make_traj(**overrides) -> Trajectory:
    defaults = dict(
        algo="PPO",
        env="CartPole-v1",
        seed=0,
        total_steps=1000,
        episode_returns=[10.0] * 100,
        episode_lengths=[100] * 100,
        entropy_log=[0.5] * 1000,
        grad_norm_log=[1.0] * 1000,
        value_log=[5.0] * 1000,
        hparams={},
    )
    defaults.update(overrides)
    return Trajectory(**defaults)


def test_nan_in_returns_detected():
    traj = _make_traj(episode_returns=[10.0] * 95 + [float("nan")] * 5)
    v = detect_nan_explosion(traj)
    assert v is not None
    assert v.failure_mode == "nan_explosion"


def test_nan_in_grad_norm_detected():
    traj = _make_traj(grad_norm_log=[1.0] * 990 + [float("nan")] * 10)
    v = detect_nan_explosion(traj)
    assert v is not None


def test_clean_trajectory_not_flagged():
    traj = _make_traj()
    assert detect_nan_explosion(traj) is None
