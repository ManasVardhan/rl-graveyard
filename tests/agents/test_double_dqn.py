"""Tests for Double DQN."""
import numpy as np
import torch
from agents.dqn import DoubleDQNAgent


def test_double_dqn_td_target_uses_online_argmax():
    """Double DQN target = r + γ * Q_target(s', argmax_a Q_online(s', a))."""
    torch.manual_seed(0)
    agent = DoubleDQNAgent(obs_dim=4, act_dim=2)
    next_obs = torch.randn(8, 4)
    rewards = torch.ones(8)
    dones = torch.zeros(8)

    with torch.no_grad():
        online_argmax = agent.q_net(next_obs).argmax(dim=1)
        target_q = agent.target_net(next_obs).gather(1, online_argmax.unsqueeze(1)).squeeze(1)
        expected = rewards + agent.gamma * target_q

    actual = agent._td_target(next_obs, rewards, dones)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_double_dqn_differs_from_dqn_target():
    """Double and vanilla DQN should yield different targets when online != target."""
    from agents.dqn import DQNAgent
    torch.manual_seed(0)
    dqn = DQNAgent(obs_dim=4, act_dim=2)
    ddqn = DoubleDQNAgent(obs_dim=4, act_dim=2)
    # Copy q_net params so online weights match; then perturb online of both
    ddqn.q_net.load_state_dict(dqn.q_net.state_dict())
    ddqn.target_net.load_state_dict(dqn.target_net.state_dict())
    with torch.no_grad():
        for p in ddqn.q_net.parameters():
            p.add_(0.5)
        for p in dqn.q_net.parameters():
            p.add_(0.5)

    next_obs = torch.randn(8, 4)
    rewards = torch.ones(8)
    dones = torch.zeros(8)
    a = dqn._td_target(next_obs, rewards, dones)
    b = ddqn._td_target(next_obs, rewards, dones)
    assert not torch.allclose(a, b, atol=1e-6)
