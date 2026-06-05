"""Rule-based failure mode classifier.

Each detector inspects a Trajectory and returns a Verdict if the signature
of its failure mode is present, else None. The orchestrator diagnose()
runs them in priority order (NaN first, then unrecoverable patterns,
then recoverable patterns) and returns the first match.
"""
from typing import Optional

import numpy as np

from common.types import Trajectory, Verdict


def detect_nan_explosion(traj: Trajectory) -> Optional[Verdict]:
    """Any NaN in episode returns or grad norms in the final 10% of training."""
    # Why this threshold: NaNs poison gradients permanently; even one in the
    # tail means the model has diverged irreversibly.
    tail_returns = traj.episode_returns[-max(10, len(traj.episode_returns) // 10):] or []
    if any(not np.isfinite(r) for r in tail_returns):
        return Verdict("nan_explosion", "NaN in episode returns")
    tail_grad = traj.grad_norm_log[-max(50, len(traj.grad_norm_log) // 10):] or []
    if any(not np.isfinite(g) for g in tail_grad):
        return Verdict("nan_explosion", "NaN in gradient norms")
    return None


def detect_exploration_collapse(traj: Trajectory) -> Optional[Verdict]:
    """Policy entropy collapsed to near-deterministic in late training.

    Why this threshold: 0.1 nats ≈ 90/10 split for binary action; effectively
    deterministic. Requires ≥100 entropy samples to ensure we're past
    early-training high-entropy phase.
    """
    if len(traj.entropy_log) < 100:
        return None
    tail = np.array(traj.entropy_log[-50:])
    if tail.mean() < 0.1:
        return Verdict(
            "exploration_collapse",
            f"Entropy collapsed to {tail.mean():.3f} in final 50 steps",
        )
    return None


def detect_death_spiral(traj: Trajectory) -> Optional[Verdict]:
    """Returns crashed: recent mean < 50% of preceding mean.

    Why this threshold: small drops are noise; 50% is large enough that
    even noisy learning curves don't trip it accidentally.
    """
    if len(traj.episode_returns) < 50:
        return None
    recent = float(np.mean(traj.episode_returns[-10:]))
    past = float(np.mean(traj.episode_returns[-50:-10]))
    if past > 0 and recent < past * 0.5:
        drop_pct = (past - recent) / past * 100
        return Verdict("death_spiral", f"Returns dropped {drop_pct:.0f}% in final 10 episodes")
    return None


def detect_stalling(traj: Trajectory) -> Optional[Verdict]:
    """Flat returns for hundreds of episodes.

    Why this threshold: CV < 5% over 500 episodes is well below random noise
    for a learning agent; effectively no signal of improvement.
    """
    if len(traj.episode_returns) < 500:
        return None
    window = np.array(traj.episode_returns[-500:])
    cv = window.std() / (abs(window.mean()) + 1e-6)
    if cv < 0.05:
        return Verdict("stalling", f"Returns flat (CV={cv:.3f}) for 500 episodes")
    return None


def detect_overshooting(traj: Trajectory) -> Optional[Verdict]:
    """Wild oscillation: CV > 0.8 over last 50 episodes.

    Why this threshold: a learning agent stabilizes; CV of 0.8 means stddev
    is 80% of the mean — alternating between high and low rewards means
    the agent's policy is thrashing.
    """
    if len(traj.episode_returns) < 50:
        return None
    window = np.array(traj.episode_returns[-50:])
    cv = window.std() / (abs(window.mean()) + 1e-6)
    if cv > 0.8:
        return Verdict("overshooting", f"Returns oscillating (CV={cv:.1f})")
    return None


def detect_reward_hacking(traj: Trajectory) -> Optional[Verdict]:
    """Detect trap-farming in the RewardTrap-v0 env.

    Why this is env-scoped: classic Gym envs are too well-designed to exhibit
    reward hacking. The trap detection looks for the specific signature of
    long episodes with returns in the trap-farming band (positive but well
    below goal reward).

    Signal: env is RewardTrap-v0 AND mean(last 50 return) > 1.0 AND
    mean(last 50 length) > 400 (near 500-step cap). Trap pays 0.5/step, so
    parking on it for 400+ steps yields > 200 return.
    """
    if traj.env != "RewardTrap-v0":
        return None
    if len(traj.episode_returns) < 50:
        return None
    recent_returns = np.array(traj.episode_returns[-50:])
    recent_lengths = np.array(traj.episode_lengths[-50:])
    if recent_returns.mean() > 1.0 and recent_lengths.mean() > 400:
        return Verdict(
            "reward_hacking",
            f"Trap-farming: mean return {recent_returns.mean():.1f} with "
            f"mean length {recent_lengths.mean():.0f} (cap=500)",
        )
    return None
