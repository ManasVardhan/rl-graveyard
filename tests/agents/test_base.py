"""Tests for agents.base.Agent ABC."""
import numpy as np
import pytest
import torch
from agents.base import Agent


def test_agent_is_abstract():
    with pytest.raises(TypeError):
        Agent(obs_dim=4, act_dim=2)


def test_concrete_subclass_must_implement_act_and_update():
    class Incomplete(Agent):
        pass

    with pytest.raises(TypeError):
        Incomplete(obs_dim=4, act_dim=2)


def test_concrete_subclass_works():
    class Stub(Agent):
        def act(self, obs):
            return 0, {"log_prob": 0.0, "value": 0.0, "entropy": 0.0}

        def update(self, batch):
            return {"loss": 0.0, "grad_norm": 0.0, "entropy": 0.0}

    agent = Stub(obs_dim=4, act_dim=2)
    action, info = agent.act(np.zeros(4))
    assert action == 0
    assert "entropy" in info
