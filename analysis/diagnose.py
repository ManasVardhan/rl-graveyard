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
