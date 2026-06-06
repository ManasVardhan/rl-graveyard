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
    """When online net argmax disagrees with target net argmax,
    DDQN's _td_target uses the online's pick (evaluated by target),
    while vanilla DQN uses target's own max — yielding different values.
    """
    from agents.dqn import DQNAgent

    class FixedNet:
        """Stand-in for q_net/target_net that returns a constant Q tensor."""
        def __init__(self, q):
            self.q = q

        def __call__(self, x):
            return self.q

        def eval(self):
            return self

        def parameters(self):
            return iter([])

    # Construct Q-values where target picks action 1 but online picks action 0
    target_q = torch.tensor([[1.0, 5.0], [2.0, 4.0]])
    online_q = torch.tensor([[9.0, 0.0], [8.0, 1.0]])

    next_obs = torch.zeros(2, 4)
    rewards = torch.tensor([0.0, 0.0])
    dones = torch.tensor([0.0, 0.0])

    ddqn = DoubleDQNAgent(obs_dim=4, act_dim=2)
    ddqn.q_net = FixedNet(online_q)
    ddqn.target_net = FixedNet(target_q)

    dqn = DQNAgent(obs_dim=4, act_dim=2)
    dqn.q_net = FixedNet(online_q)
    dqn.target_net = FixedNet(target_q)

    # DDQN: online picks action 0 → target_q[:, 0] = [1.0, 2.0] → r + γ * [1,2]
    ddqn_target = ddqn._td_target(next_obs, rewards, dones)
    expected_ddqn = ddqn.gamma * torch.tensor([1.0, 2.0])
    assert torch.allclose(ddqn_target, expected_ddqn, atol=1e-6)

    # Vanilla DQN: target.max along action axis → [5.0, 4.0]
    dqn_target = dqn._td_target(next_obs, rewards, dones)
    expected_dqn = dqn.gamma * torch.tensor([5.0, 4.0])
    assert torch.allclose(dqn_target, expected_dqn, atol=1e-6)

    # Sanity: they really differ
    assert not torch.allclose(ddqn_target, dqn_target, atol=1e-6)
