"""Tests for DQN, Double DQN, Dueling DQN."""
import numpy as np
import torch
from agents.dqn import DQNAgent, QNet


def test_qnet_output_shape():
    net = QNet(obs_dim=4, act_dim=2)
    obs = torch.zeros(8, 4)
    q = net(obs)
    assert q.shape == (8, 2)


def test_act_epsilon_zero_is_greedy():
    torch.manual_seed(0)
    agent = DQNAgent(obs_dim=4, act_dim=2)
    agent.epsilon = 0.0  # force greedy
    obs = np.zeros(4, dtype=np.float32)
    action, info = agent.act(obs)
    with torch.no_grad():
        q = agent.q_net(torch.as_tensor(obs, dtype=torch.float32))
    assert action == int(q.argmax().item())


def test_act_epsilon_one_is_random():
    agent = DQNAgent(obs_dim=4, act_dim=2)
    agent.epsilon = 1.0
    obs = np.zeros(4, dtype=np.float32)
    actions = [agent.act(obs)[0] for _ in range(50)]
    # Should include both actions with overwhelming probability
    assert set(actions) == {0, 1}


def test_update_modifies_parameters():
    from agents.replay import ReplayBuffer
    torch.manual_seed(0)
    agent = DQNAgent(obs_dim=4, act_dim=2, lr=1e-2)
    buf = ReplayBuffer(capacity=100, obs_dim=4)
    for i in range(64):
        buf.add(
            obs=np.full(4, i / 64, dtype=np.float32),
            action=i % 2,
            reward=float(i % 3),
            next_obs=np.full(4, (i + 1) / 64, dtype=np.float32),
            done=(i % 10 == 0),
        )

    initial = {k: v.clone() for k, v in agent.q_net.state_dict().items()}
    batch = buf.sample(batch_size=32)
    agent.update(batch)
    after = agent.q_net.state_dict()
    assert any(not torch.equal(initial[k], after[k]) for k in initial)


def test_target_net_syncs():
    agent = DQNAgent(obs_dim=4, act_dim=2)
    # Perturb online net
    with torch.no_grad():
        for p in agent.q_net.parameters():
            p.add_(1.0)
    agent.sync_target()
    for online_p, target_p in zip(agent.q_net.parameters(), agent.target_net.parameters()):
        assert torch.equal(online_p, target_p)
