"""Tests for PPO agent."""
import numpy as np
import torch
from agents.ppo import PPOAgent, compute_gae


def test_compute_gae_terminal():
    rewards = [1.0, 1.0, 1.0]
    values = [0.5, 0.6, 0.7]
    dones = [False, False, True]
    advantages, returns = compute_gae(
        rewards, values, dones, bootstrap_value=0.0, gamma=0.99, lam=0.95
    )
    assert len(advantages) == 3
    assert len(returns) == 3
    # On terminal step, advantage = r - V(s_t)
    assert abs(advantages[-1] - (1.0 - 0.7)) < 1e-6


def test_compute_gae_nonterminal_bootstrap():
    rewards = [0.0, 0.0, 0.0]
    values = [0.0, 0.0, 0.0]
    dones = [False, False, False]
    advantages, _ = compute_gae(
        rewards, values, dones, bootstrap_value=10.0, gamma=1.0, lam=1.0
    )
    # With γ=1, λ=1: advantage_0 = sum of future rewards + bootstrap = 10
    assert abs(advantages[0] - 10.0) < 1e-6


def test_act_and_update():
    agent = PPOAgent(obs_dim=4, act_dim=2)
    obs = np.zeros(4, dtype=np.float32)
    action, info = agent.act(obs)
    assert action in (0, 1)
    rollout = {
        "obs": [obs] * 32,
        "actions": [0, 1] * 16,
        "log_probs_old": [info["log_prob"]] * 32,
        "rewards": [1.0] * 32,
        "dones": [False] * 31 + [True],
        "values": [info["value"]] * 32,
        "bootstrap_value": 0.0,
    }
    metrics = agent.update(rollout)
    assert "loss" in metrics
    assert "grad_norm" in metrics
    assert "entropy" in metrics
    assert "kl_divergence" in metrics


def test_parameters_change_after_update():
    torch.manual_seed(0)
    agent = PPOAgent(obs_dim=4, act_dim=2, lr=1e-2, k_epochs=4)
    obs = np.zeros(4, dtype=np.float32)
    initial = {k: v.clone() for k, v in agent.net.state_dict().items()}
    rollout = {
        "obs": [obs] * 32,
        "actions": [0, 1] * 16,
        "log_probs_old": [0.0] * 32,
        "rewards": [1.0] * 32,
        "dones": [False] * 31 + [True],
        "values": [0.0] * 32,
        "bootstrap_value": 0.0,
    }
    agent.update(rollout)
    after = agent.net.state_dict()
    assert any(not torch.equal(initial[k], after[k]) for k in initial)
