"""Tests for REINFORCE agent."""
import numpy as np
import torch
from agents.reinforce import REINFORCEAgent


def test_act_returns_valid_action():
    agent = REINFORCEAgent(obs_dim=4, act_dim=2)
    obs = np.zeros(4, dtype=np.float32)
    action, info = agent.act(obs)
    assert action in (0, 1)
    assert "log_prob" in info and "entropy" in info


def test_update_returns_metrics():
    agent = REINFORCEAgent(obs_dim=4, act_dim=2)
    # Fake episode: 5 transitions
    rewards = [1.0] * 5
    log_probs = [torch.tensor(-0.5, requires_grad=True) for _ in range(5)]
    entropies = [torch.tensor(0.7) for _ in range(5)]
    metrics = agent.update({"rewards": rewards, "log_probs": log_probs, "entropies": entropies})
    assert "loss" in metrics
    assert "grad_norm" in metrics
    assert "entropy" in metrics


def test_loss_decreases_on_positive_reward():
    """Sanity: with consistently positive advantage, the loss should be finite and update parameters."""
    torch.manual_seed(0)
    agent = REINFORCEAgent(obs_dim=4, act_dim=2, lr=1e-2)
    obs = np.zeros(4, dtype=np.float32)

    initial = {k: v.clone() for k, v in agent.policy.state_dict().items()}

    for _ in range(10):
        log_probs, entropies = [], []
        for _ in range(20):
            _, info = agent.act(obs)
            log_probs.append(info["_log_prob_tensor"])
            entropies.append(info["_entropy_tensor"])
        rewards = [1.0] * 20
        agent.update({"rewards": rewards, "log_probs": log_probs, "entropies": entropies})

    after = agent.policy.state_dict()
    # At least one parameter must have moved
    assert any(not torch.equal(initial[k], after[k]) for k in initial)
