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


from analysis.diagnose import detect_exploration_collapse


def test_exploration_collapse_low_entropy():
    """Entropy < 0.1 in final 50 steps → collapse."""
    traj = _make_traj(entropy_log=[0.5] * 500 + [0.05] * 500)
    v = detect_exploration_collapse(traj)
    assert v is not None
    assert v.failure_mode == "exploration_collapse"


def test_exploration_collapse_not_triggered_by_high_entropy():
    traj = _make_traj(entropy_log=[0.5] * 1000)
    assert detect_exploration_collapse(traj) is None


def test_exploration_collapse_requires_enough_data():
    """With < 100 entropy samples, can't conclude collapse."""
    traj = _make_traj(entropy_log=[0.0] * 50)
    assert detect_exploration_collapse(traj) is None


from analysis.diagnose import detect_death_spiral


def test_death_spiral_returns_crash():
    """Recent 10 episodes < 50% of preceding 40 episodes → death spiral."""
    early = [100.0] * 50
    late = [10.0] * 10
    traj = _make_traj(episode_returns=early + late, episode_lengths=[100] * 60)
    v = detect_death_spiral(traj)
    assert v is not None
    assert v.failure_mode == "death_spiral"


def test_death_spiral_not_triggered_by_stable_returns():
    traj = _make_traj(episode_returns=[100.0] * 60, episode_lengths=[100] * 60)
    assert detect_death_spiral(traj) is None


def test_death_spiral_requires_50_episodes():
    traj = _make_traj(episode_returns=[100.0] * 30 + [10.0] * 5, episode_lengths=[100] * 35)
    assert detect_death_spiral(traj) is None
