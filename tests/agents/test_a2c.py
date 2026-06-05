"""Tests for A2C agent."""
import numpy as np
import torch
from agents.a2c import A2CAgent


def test_act_returns_value_estimate():
    agent = A2CAgent(obs_dim=4, act_dim=2)
    obs = np.zeros(4, dtype=np.float32)
    action, info = agent.act(obs)
    assert action in (0, 1)
    assert "value" in info
    assert isinstance(info["value"], float)


def test_update_with_n_step_rollout():
    agent = A2CAgent(obs_dim=4, act_dim=2)
    obs = np.zeros(4, dtype=np.float32)
    rollout = {
        "obs": [obs] * 5,
        "actions": [0, 1, 0, 1, 0],
        "rewards": [1.0, 1.0, 1.0, 1.0, 1.0],
        "dones": [False, False, False, False, True],
        "bootstrap_value": 0.0,
    }
    metrics = agent.update(rollout)
    assert "loss" in metrics
    assert "grad_norm" in metrics
    assert "entropy" in metrics


def test_parameters_change_after_update():
    torch.manual_seed(0)
    agent = A2CAgent(obs_dim=4, act_dim=2, lr=1e-2)
    obs = np.zeros(4, dtype=np.float32)
    initial = {k: v.clone() for k, v in agent.net.state_dict().items()}

    rollout = {
        "obs": [obs] * 5,
        "actions": [0, 1, 0, 1, 0],
        "rewards": [1.0] * 5,
        "dones": [False] * 4 + [True],
        "bootstrap_value": 0.0,
    }
    agent.update(rollout)
    after = agent.net.state_dict()
    assert any(not torch.equal(initial[k], after[k]) for k in initial)
