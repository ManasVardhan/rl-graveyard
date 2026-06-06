"""Tests for the replay buffer."""
import numpy as np
from agents.replay import ReplayBuffer


def test_add_and_sample():
    buf = ReplayBuffer(capacity=100, obs_dim=4)
    for i in range(50):
        buf.add(
            obs=np.full(4, i, dtype=np.float32),
            action=i % 2,
            reward=float(i),
            next_obs=np.full(4, i + 1, dtype=np.float32),
            done=False,
        )
    batch = buf.sample(batch_size=16)
    assert batch["obs"].shape == (16, 4)
    assert batch["actions"].shape == (16,)
    assert batch["rewards"].shape == (16,)
    assert batch["next_obs"].shape == (16, 4)
    assert batch["dones"].shape == (16,)


def test_circular_overwrite():
    buf = ReplayBuffer(capacity=5, obs_dim=2)
    for i in range(7):
        buf.add(
            obs=np.array([i, i], dtype=np.float32),
            action=0,
            reward=float(i),
            next_obs=np.array([i + 1, i + 1], dtype=np.float32),
            done=False,
        )
    # After 7 inserts into cap-5 buffer, items 0,1 should be overwritten by 5,6
    assert len(buf) == 5
    # Sample the entire buffer; rewards must come from {2,3,4,5,6}
    batch = buf.sample(batch_size=5)
    for r in batch["rewards"]:
        assert r in {2.0, 3.0, 4.0, 5.0, 6.0}


def test_sample_raises_if_empty():
    buf = ReplayBuffer(capacity=10, obs_dim=2)
    import pytest
    with pytest.raises(ValueError):
        buf.sample(batch_size=4)
