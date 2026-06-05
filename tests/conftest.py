"""Shared pytest fixtures."""
import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def deterministic_seed():
    """Make every test deterministic."""
    np.random.seed(0)
    torch.manual_seed(0)
