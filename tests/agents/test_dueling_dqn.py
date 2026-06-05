"""Tests for Dueling DQN."""
import torch
from agents.dqn import DuelingDQNAgent, DuelingQNet


def test_dueling_qnet_output_shape():
    net = DuelingQNet(obs_dim=4, act_dim=3)
    obs = torch.zeros(8, 4)
    q = net(obs)
    assert q.shape == (8, 3)


def test_dueling_advantage_zero_mean_property():
    """V+A-mean(A) means advantages sum to zero per state."""
    net = DuelingQNet(obs_dim=4, act_dim=3)
    obs = torch.randn(8, 4)
    h = net.trunk(obs)
    a = net.advantage_head(h)
    centered = a - a.mean(dim=-1, keepdim=True)
    assert torch.allclose(centered.sum(dim=-1), torch.zeros(8), atol=1e-5)


def test_dueling_agent_uses_dueling_net():
    agent = DuelingDQNAgent(obs_dim=4, act_dim=2)
    assert isinstance(agent.q_net, DuelingQNet)
    assert isinstance(agent.target_net, DuelingQNet)
