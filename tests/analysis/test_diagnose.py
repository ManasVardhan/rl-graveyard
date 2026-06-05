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


from analysis.diagnose import detect_stalling


def test_stalling_low_variance():
    """Coefficient of variation < 5% over last 500 episodes → stalling."""
    traj = _make_traj(
        episode_returns=[10.0 + np.random.uniform(-0.1, 0.1) for _ in range(600)],
        episode_lengths=[100] * 600,
    )
    v = detect_stalling(traj)
    assert v is not None
    assert v.failure_mode == "stalling"


def test_stalling_not_triggered_by_learning():
    traj = _make_traj(
        episode_returns=list(np.linspace(10, 100, 600)),
        episode_lengths=[100] * 600,
    )
    assert detect_stalling(traj) is None


def test_stalling_requires_500_episodes():
    traj = _make_traj(episode_returns=[10.0] * 400, episode_lengths=[100] * 400)
    assert detect_stalling(traj) is None


from analysis.diagnose import detect_overshooting


def test_overshooting_high_variance():
    """CV > 2 over last 50 episodes → wild oscillation."""
    returns = [100.0 if i % 2 == 0 else 5.0 for i in range(60)]
    traj = _make_traj(episode_returns=returns, episode_lengths=[100] * 60)
    v = detect_overshooting(traj)
    assert v is not None
    assert v.failure_mode == "overshooting"


def test_overshooting_not_triggered_by_moderate_variance():
    traj = _make_traj(
        episode_returns=[100.0 + np.random.uniform(-5, 5) for _ in range(60)],
        episode_lengths=[100] * 60,
    )
    assert detect_overshooting(traj) is None


from analysis.diagnose import detect_reward_hacking


def test_reward_hacking_in_reward_trap():
    """RewardTrap-v0 with returns in trap-farming range + max-length episodes."""
    traj = _make_traj(
        env="RewardTrap-v0",
        episode_returns=[200.0] * 50 + [240.0] * 10,
        episode_lengths=[500] * 60,
    )
    v = detect_reward_hacking(traj)
    assert v is not None
    assert v.failure_mode == "reward_hacking"


def test_reward_hacking_not_flagged_when_reaching_goal():
    """High reward + short episodes = reached goal legitimately."""
    traj = _make_traj(
        env="RewardTrap-v0",
        episode_returns=[9.5] * 60,
        episode_lengths=[20] * 60,
    )
    assert detect_reward_hacking(traj) is None


def test_reward_hacking_not_flagged_outside_reward_trap():
    """Detector is scoped to RewardTrap-v0 in v1."""
    traj = _make_traj(
        env="CartPole-v1",
        episode_returns=[200.0] * 60,
        episode_lengths=[500] * 60,
    )
    assert detect_reward_hacking(traj) is None
