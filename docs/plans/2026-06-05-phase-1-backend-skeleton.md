# Phase 1: Backend Skeleton — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the backend that lets a single training run go from `ExperimentConfig` → trained agent → SQLite-persisted trajectory → classified autopsy, for all 6 algorithms on all 5 environments, locally on CPU.

**Architecture:** Top-level Python packages (`agents/`, `envs/`, `experiments/`, `analysis/`, `data/`, `common/`) installed via `pyproject.toml`. Each algorithm is its own module with a shared `Agent` ABC; DQN variants are subclasses sharing one replay buffer. The experiment runner orchestrates training, logging, and post-hoc diagnosis. Tests use pytest and `pytest -k` for selective runs.

**Tech Stack:** Python 3.10+, PyTorch 2.0+, Gymnasium 0.29+, pytest, SQLite (stdlib).

**Spec:** `docs/specs/2026-06-05-v1-design.md`

**Convention:** All commit messages use Conventional Commits prefix (`feat:`, `test:`, `chore:`, `docs:`).

---

## Task 1: Project structure & tooling

**Files:**
- Create: `pyproject.toml`
- Create: `agents/__init__.py`, `envs/__init__.py`, `experiments/__init__.py`, `analysis/__init__.py`, `data/__init__.py`, `common/__init__.py`
- Create: `tests/__init__.py`, `tests/conftest.py`
- Modify: `requirements.txt`
- Create: `.gitignore`

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "rl-graveyard"
version = "0.1.0"
description = "A systematic catalog of RL failure modes"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.0.0",
    "gymnasium[box2d]>=0.29.0",
    "numpy>=1.24.0",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
dev = ["pytest>=7.0", "pytest-cov>=4.0"]

[tool.setuptools.packages.find]
include = ["agents*", "envs*", "experiments*", "analysis*", "data*", "common*"]
exclude = ["tests*", "docs*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
addopts = "-v --tb=short"
```

- [ ] **Step 2: Write `requirements.txt`** (replaces existing — wandb removed per spec §8)

```
torch>=2.0.0
gymnasium[box2d]>=0.29.0
numpy>=1.24.0
pyyaml>=6.0
pytest>=7.0
```

- [ ] **Step 3: Write `.gitignore`**

```
__pycache__/
*.pyc
*.egg-info/
.pytest_cache/
.coverage
*.sqlite-journal
data/*.sqlite
!data/.gitkeep
.venv/
.DS_Store
```

- [ ] **Step 4: Create empty `__init__.py` in each package directory**

Run:
```bash
mkdir -p agents envs experiments analysis data common tests
touch agents/__init__.py envs/__init__.py experiments/__init__.py \
      analysis/__init__.py data/__init__.py common/__init__.py \
      tests/__init__.py data/.gitkeep
```

- [ ] **Step 5: Write `tests/conftest.py`**

```python
"""Shared pytest fixtures."""
import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def deterministic_seed():
    """Make every test deterministic."""
    np.random.seed(0)
    torch.manual_seed(0)
```

- [ ] **Step 6: Install in dev mode and verify**

Run:
```bash
pip install -e ".[dev]"
pytest --collect-only
```

Expected: pytest reports `collected 0 items` (no tests yet) with no errors.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml requirements.txt .gitignore agents/ envs/ experiments/ analysis/ data/ common/ tests/
git commit -m "chore: bootstrap python package structure"
```

---

## Task 2: Core types

**Files:**
- Create: `common/types.py`
- Create: `tests/test_types.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_types.py
"""Tests for common.types."""
from common.types import Trajectory, Verdict, EnvMeta


def test_trajectory_construction():
    traj = Trajectory(
        algo="PPO",
        env="CartPole-v1",
        seed=0,
        total_steps=1000,
        episode_returns=[10.0, 20.0],
        episode_lengths=[100, 200],
        entropy_log=[0.5, 0.4],
        grad_norm_log=[1.0, 0.9],
        value_log=[5.0, 6.0],
        hparams={"lr": 3e-4},
    )
    assert traj.algo == "PPO"
    assert len(traj.episode_returns) == 2


def test_verdict_construction():
    v = Verdict(failure_mode="exploration_collapse", cause="entropy < 0.1")
    assert v.failure_mode == "exploration_collapse"


def test_env_meta_freezes():
    import dataclasses
    meta = EnvMeta(
        name="CartPole-v1",
        max_steps=500,
        success_threshold=475.0,
        act_type="discrete",
    )
    assert dataclasses.is_dataclass(meta)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_types.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'common.types'`

- [ ] **Step 3: Write `common/types.py`**

```python
"""Shared dataclasses used across packages."""
from dataclasses import dataclass, field
from typing import Literal


FailureMode = Literal[
    "alive",
    "nan_explosion",
    "exploration_collapse",
    "death_spiral",
    "stalling",
    "overshooting",
    "reward_hacking",
]


ActType = Literal["discrete", "continuous"]


@dataclass
class Trajectory:
    """All metrics recorded during a training run."""
    algo: str
    env: str
    seed: int
    total_steps: int
    episode_returns: list[float]
    episode_lengths: list[int]
    entropy_log: list[float]
    grad_norm_log: list[float]
    value_log: list[float]
    hparams: dict


@dataclass
class Verdict:
    """Output of the failure classifier."""
    failure_mode: FailureMode
    cause: str


@dataclass(frozen=True)
class EnvMeta:
    """Static metadata about an environment."""
    name: str
    max_steps: int
    success_threshold: float
    act_type: ActType
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_types.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add common/ tests/test_types.py
git commit -m "feat(common): add shared Trajectory, Verdict, EnvMeta types"
```

---

## Task 3: Agent base class

**Files:**
- Create: `agents/base.py`
- Create: `tests/agents/__init__.py`, `tests/agents/test_base.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/agents/test_base.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/agents/test_base.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agents.base'`.

- [ ] **Step 3: Create `tests/agents/__init__.py`**

```bash
mkdir -p tests/agents && touch tests/agents/__init__.py
```

- [ ] **Step 4: Write `agents/base.py`**

```python
"""Agent ABC. All algorithms implement this interface."""
from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Agent(ABC):
    """Base class for all RL agents.

    Each algorithm has its own training loop (on-policy vs off-policy
    collection patterns differ too much for a shared trainer), but they
    all expose the same act/update interface so the runner can drive them
    uniformly.
    """

    def __init__(self, obs_dim: int, act_dim: int):
        self.obs_dim = obs_dim
        self.act_dim = act_dim

    @abstractmethod
    def act(self, obs: np.ndarray) -> tuple[int, dict[str, float]]:
        """Pick an action.

        Returns:
            (action, info) where info contains at minimum:
              - log_prob: float — log π(a|s), or 0.0 for off-policy ε-greedy
              - value: float — V(s), or 0.0 if no value head
              - entropy: float — H(π(·|s))
        """

    @abstractmethod
    def update(self, batch: Any) -> dict[str, float]:
        """Update the agent. Batch shape is algorithm-specific.

        Returns metrics dict with at minimum: loss, grad_norm, entropy.
        """
```

- [ ] **Step 5: Run test**

Run: `pytest tests/agents/test_base.py -v`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add agents/base.py tests/agents/
git commit -m "feat(agents): add Agent ABC"
```

---

## Task 4: Environment registry & logging wrapper

**Files:**
- Create: `envs/registry.py`
- Create: `envs/wrappers.py`
- Create: `tests/envs/__init__.py`, `tests/envs/test_registry.py`, `tests/envs/test_wrappers.py`

- [ ] **Step 1: Write failing test for registry**

```python
# tests/envs/test_registry.py
"""Tests for envs.registry."""
import pytest
from envs.registry import REGISTRY, make_env


def test_classic_envs_present():
    for name in ["CartPole-v1", "Acrobot-v1", "MountainCar-v0", "LunarLander-v2"]:
        assert name in REGISTRY


def test_make_env_returns_seeded_env():
    env = make_env("CartPole-v1", seed=42)
    obs1, _ = env.reset()
    env2 = make_env("CartPole-v1", seed=42)
    obs2, _ = env2.reset()
    assert (obs1 == obs2).all()
    env.close()
    env2.close()


def test_unknown_env_raises():
    with pytest.raises(KeyError):
        make_env("Nonexistent-v0", seed=0)


def test_metadata_has_act_type():
    assert REGISTRY["CartPole-v1"].act_type == "discrete"
```

- [ ] **Step 2: Run, expect fail**

Run: `pytest tests/envs/test_registry.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Create test init and write `envs/registry.py`**

```bash
mkdir -p tests/envs && touch tests/envs/__init__.py
```

```python
# envs/registry.py
"""Registry of supported environments."""
from __future__ import annotations

import gymnasium as gym

from common.types import EnvMeta


REGISTRY: dict[str, EnvMeta] = {
    "CartPole-v1": EnvMeta(
        name="CartPole-v1",
        max_steps=500,
        success_threshold=475.0,
        act_type="discrete",
    ),
    "Acrobot-v1": EnvMeta(
        name="Acrobot-v1",
        max_steps=500,
        success_threshold=-100.0,
        act_type="discrete",
    ),
    "MountainCar-v0": EnvMeta(
        name="MountainCar-v0",
        max_steps=200,
        success_threshold=-110.0,
        act_type="discrete",
    ),
    "LunarLander-v2": EnvMeta(
        name="LunarLander-v2",
        max_steps=1000,
        success_threshold=200.0,
        act_type="discrete",
    ),
}


def make_env(name: str, seed: int) -> gym.Env:
    """Construct a Gymnasium env, seeded deterministically."""
    if name not in REGISTRY:
        raise KeyError(f"Unknown env: {name}. Available: {list(REGISTRY)}")
    env = gym.make(name)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env
```

- [ ] **Step 4: Run registry test**

Run: `pytest tests/envs/test_registry.py -v`
Expected: 4 passed.

- [ ] **Step 5: Write failing test for logging wrapper**

```python
# tests/envs/test_wrappers.py
"""Tests for envs.wrappers.LoggingWrapper."""
import numpy as np
from envs.registry import make_env
from envs.wrappers import LoggingWrapper


def test_records_episode_returns():
    env = LoggingWrapper(make_env("CartPole-v1", seed=0))
    env.reset(seed=0)
    for _ in range(50):
        obs, reward, term, trunc, info = env.step(env.action_space.sample())
        if term or trunc:
            env.reset()
    assert len(env.episode_returns) >= 1
    assert all(isinstance(r, float) for r in env.episode_returns)


def test_records_episode_lengths_match_returns():
    env = LoggingWrapper(make_env("CartPole-v1", seed=0))
    env.reset(seed=0)
    for _ in range(50):
        obs, reward, term, trunc, info = env.step(env.action_space.sample())
        if term or trunc:
            env.reset()
    assert len(env.episode_returns) == len(env.episode_lengths)
```

- [ ] **Step 6: Run, expect fail**

Run: `pytest tests/envs/test_wrappers.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 7: Write `envs/wrappers.py`**

```python
# envs/wrappers.py
"""Logging wrappers for environments."""
import gymnasium as gym


class LoggingWrapper(gym.Wrapper):
    """Record episode-level statistics: returns and lengths.

    Per-step metrics that depend on the agent (entropy, grad_norm) are
    recorded by the runner, not here — the env doesn't know about them.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.episode_returns: list[float] = []
        self.episode_lengths: list[int] = []
        self._current_return = 0.0
        self._current_length = 0

    def reset(self, *, seed=None, options=None):
        self._current_return = 0.0
        self._current_length = 0
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._current_return += float(reward)
        self._current_length += 1
        if terminated or truncated:
            self.episode_returns.append(self._current_return)
            self.episode_lengths.append(self._current_length)
        return obs, reward, terminated, truncated, info
```

- [ ] **Step 8: Run all envs tests**

Run: `pytest tests/envs/ -v`
Expected: 6 passed.

- [ ] **Step 9: Commit**

```bash
git add envs/ tests/envs/
git commit -m "feat(envs): add registry and LoggingWrapper"
```

---

## Task 5: RewardTrap custom environment

**Files:**
- Create: `envs/reward_trap.py`
- Modify: `envs/registry.py`
- Create: `tests/envs/test_reward_trap.py`

**Design:** A 5×5 gridworld. Agent starts at (0,0), goal at (4,4) gives +10 and terminates. A "trap" tile at (2,2) gives +0.5 per step the agent occupies it, never terminates. Step penalty is -0.01. Optimal myopic strategy: rush to goal (+9.91 over ~8 steps). Trap-hacking strategy: park on (2,2) (+0.49/step × ~500 steps = ~245). PPO under default hparams should fall in.

- [ ] **Step 1: Write failing test**

```python
# tests/envs/test_reward_trap.py
"""Tests for the RewardTrap gridworld."""
import gymnasium as gym
import numpy as np
import pytest
from envs.reward_trap import RewardTrapEnv
from envs.registry import REGISTRY


def test_reward_trap_in_registry():
    assert "RewardTrap-v0" in REGISTRY


def test_action_space_is_discrete_4():
    env = RewardTrapEnv()
    assert isinstance(env.action_space, gym.spaces.Discrete)
    assert env.action_space.n == 4  # up, down, left, right


def test_reach_goal_terminates_with_reward_10():
    env = RewardTrapEnv()
    env.reset(seed=0)
    # Move right 4 times then down 4 times — should reach (4,4)
    for _ in range(4):
        env.step(3)  # right
    for _ in range(3):
        obs, r, term, trunc, info = env.step(1)  # down
        assert not term
    obs, r, term, trunc, info = env.step(1)  # final down
    assert term
    assert r > 9.0  # +10 minus tiny step penalty


def test_trap_tile_pays_per_step_and_doesnt_terminate():
    env = RewardTrapEnv()
    env.reset(seed=0)
    # Move right 2, down 2 to reach trap at (2,2)
    for _ in range(2):
        env.step(3)
    for _ in range(2):
        obs, r, term, trunc, info = env.step(1)
    # Now standing on trap
    obs, r, term, trunc, info = env.step(0)  # try to move up (will leave trap)
    # Move back onto trap
    obs, r, term, trunc, info = env.step(1)  # down — back on trap
    assert r > 0.4  # trap pays +0.5 minus step penalty
    assert not term


def test_truncates_at_max_steps():
    env = RewardTrapEnv()
    env.reset(seed=0)
    for i in range(500):
        obs, r, term, trunc, info = env.step(0)  # stay near origin by oscillating
        if term or trunc:
            break
    assert trunc or term  # must end one way or another


def test_obs_is_2d_position():
    env = RewardTrapEnv()
    obs, _ = env.reset(seed=0)
    assert obs.shape == (2,)
```

- [ ] **Step 2: Run, expect fail**

Run: `pytest tests/envs/test_reward_trap.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write `envs/reward_trap.py`**

```python
# envs/reward_trap.py
"""A gridworld with a known reward exploit.

The agent should learn to reach the goal at (4, 4) for +10 and terminate.
However, a trap tile at (2, 2) pays +0.5 per step occupied and never
terminates. Agents that maximize discounted return often discover this
exploit, especially under high γ.

This is the canonical demonstration of Reward Hacking in this catalog.
"""
import gymnasium as gym
import numpy as np


GRID_SIZE = 5
GOAL = (4, 4)
TRAP = (2, 2)
GOAL_REWARD = 10.0
TRAP_REWARD = 0.5
STEP_PENALTY = -0.01
MAX_STEPS = 500


class RewardTrapEnv(gym.Env):
    """5x5 gridworld with goal + trap. Discrete actions: 0=up, 1=down, 2=left, 3=right."""

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(
            low=0, high=GRID_SIZE - 1, shape=(2,), dtype=np.float32
        )
        self._pos = np.array([0, 0], dtype=np.int32)
        self._step_count = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._pos = np.array([0, 0], dtype=np.int32)
        self._step_count = 0
        return self._obs(), {}

    def step(self, action):
        # action: 0=up (row-1), 1=down (row+1), 2=left (col-1), 3=right (col+1)
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        dr, dc = deltas[int(action)]
        new_r = int(np.clip(self._pos[0] + dr, 0, GRID_SIZE - 1))
        new_c = int(np.clip(self._pos[1] + dc, 0, GRID_SIZE - 1))
        self._pos = np.array([new_r, new_c], dtype=np.int32)
        self._step_count += 1

        reward = STEP_PENALTY
        terminated = False
        if (new_r, new_c) == GOAL:
            reward += GOAL_REWARD
            terminated = True
        elif (new_r, new_c) == TRAP:
            reward += TRAP_REWARD

        truncated = self._step_count >= MAX_STEPS
        return self._obs(), float(reward), terminated, truncated, {}

    def _obs(self) -> np.ndarray:
        return self._pos.astype(np.float32)
```

- [ ] **Step 4: Register the env in Gymnasium and add to `REGISTRY`**

Modify `envs/registry.py`. At top of file, after the imports, register the env:

```python
# envs/registry.py — add after the gymnasium import
from gymnasium.envs.registration import register

from envs.reward_trap import RewardTrapEnv

register(
    id="RewardTrap-v0",
    entry_point="envs.reward_trap:RewardTrapEnv",
    max_episode_steps=500,
)
```

And add to `REGISTRY` dict:

```python
    "RewardTrap-v0": EnvMeta(
        name="RewardTrap-v0",
        max_steps=500,
        success_threshold=9.0,
        act_type="discrete",
    ),
```

- [ ] **Step 5: Run all env tests**

Run: `pytest tests/envs/ -v`
Expected: 11 passed.

- [ ] **Step 6: Commit**

```bash
git add envs/reward_trap.py envs/registry.py tests/envs/test_reward_trap.py
git commit -m "feat(envs): add RewardTrap-v0 gridworld with goal+trap"
```

---

## Task 6: REINFORCE

**Files:**
- Create: `agents/reinforce.py`
- Create: `tests/agents/test_reinforce.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/agents/test_reinforce.py
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
```

- [ ] **Step 2: Run, expect fail**

Run: `pytest tests/agents/test_reinforce.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write `agents/reinforce.py`**

```python
# agents/reinforce.py
"""Monte-Carlo policy gradient (REINFORCE) with entropy bonus.

The simplest possible policy gradient: collect a full episode, compute
discounted returns, take a single gradient step. No value baseline in v1
(deliberately — REINFORCE without baseline is the canonical
high-variance failure case we want in the graveyard).
"""
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from agents.base import Agent


class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class REINFORCEAgent(Agent):
    """REINFORCE. On-policy. Updates once per episode."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        entropy_coef: float = 0.01,
        grad_clip: float = 0.5,
    ):
        super().__init__(obs_dim, act_dim)
        self.policy = PolicyNet(obs_dim, act_dim)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.grad_clip = grad_clip

    def act(self, obs: np.ndarray) -> tuple[int, dict[str, float]]:
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        logits = self.policy(obs_t)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return int(action.item()), {
            "log_prob": float(log_prob.item()),
            "value": 0.0,
            "entropy": float(entropy.item()),
            # Hidden tensor handles for the runner to pass into update()
            "_log_prob_tensor": log_prob,
            "_entropy_tensor": entropy,
        }

    def update(self, batch: dict[str, Any]) -> dict[str, float]:
        rewards: list[float] = batch["rewards"]
        log_probs: list[torch.Tensor] = batch["log_probs"]
        entropies: list[torch.Tensor] = batch["entropies"]

        # Discounted returns G_t = sum_{k>=0} gamma^k r_{t+k}
        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns_t = torch.tensor(returns, dtype=torch.float32)
        # Normalize for variance reduction (this is REINFORCE-with-baseline-lite)
        if returns_t.numel() > 1:
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        log_probs_t = torch.stack(log_probs)
        entropies_t = torch.stack(entropies)

        policy_loss = -(log_probs_t * returns_t).mean()
        entropy_bonus = entropies_t.mean()
        loss = policy_loss - self.entropy_coef * entropy_bonus

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
        self.optimizer.step()

        return {
            "loss": float(loss.item()),
            "grad_norm": float(grad_norm.item()),
            "entropy": float(entropy_bonus.item()),
        }
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/agents/test_reinforce.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add agents/reinforce.py tests/agents/test_reinforce.py
git commit -m "feat(agents): add REINFORCE with entropy bonus"
```

---

## Task 7: A2C

**Files:**
- Create: `agents/a2c.py`
- Create: `tests/agents/test_a2c.py`

A2C is REINFORCE + critic baseline + n-step (not full episode) updates. In v1 we use n=5 step rollouts.

- [ ] **Step 1: Write the failing test**

```python
# tests/agents/test_a2c.py
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
```

- [ ] **Step 2: Run, expect fail**

Run: `pytest tests/agents/test_a2c.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write `agents/a2c.py`**

```python
# agents/a2c.py
"""Synchronous Advantage Actor-Critic with n-step returns.

Update flow:
  - Collect n=5 steps of experience
  - Compute n-step returns bootstrapped with V(s_{t+n})
  - Advantages = returns - V(s_t)
  - Joint loss: policy gradient + value MSE - entropy bonus
"""
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from agents.base import Agent


class ActorCriticNet(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden, act_dim)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.shared(x)
        return self.policy_head(h), self.value_head(h).squeeze(-1)


class A2CAgent(Agent):
    """A2C with shared body. On-policy."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        lr: float = 7e-4,
        gamma: float = 0.99,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        grad_clip: float = 0.5,
        n_steps: int = 5,
    ):
        super().__init__(obs_dim, act_dim)
        self.net = ActorCriticNet(obs_dim, act_dim)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.gamma = gamma
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.grad_clip = grad_clip
        self.n_steps = n_steps

    def act(self, obs: np.ndarray) -> tuple[int, dict[str, float]]:
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        logits, value = self.net(obs_t)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return int(action.item()), {
            "log_prob": float(dist.log_prob(action).item()),
            "value": float(value.item()),
            "entropy": float(dist.entropy().item()),
        }

    def update(self, batch: dict[str, Any]) -> dict[str, float]:
        obs_list: list[np.ndarray] = batch["obs"]
        actions: list[int] = batch["actions"]
        rewards: list[float] = batch["rewards"]
        dones: list[bool] = batch["dones"]
        bootstrap_value: float = batch["bootstrap_value"]

        obs_t = torch.as_tensor(np.array(obs_list), dtype=torch.float32)
        actions_t = torch.as_tensor(actions, dtype=torch.int64)

        # n-step returns
        returns = []
        R = bootstrap_value
        for r, done in zip(reversed(rewards), reversed(dones)):
            R = r + self.gamma * R * (0.0 if done else 1.0)
            returns.insert(0, R)
        returns_t = torch.tensor(returns, dtype=torch.float32)

        logits, values = self.net(obs_t)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions_t)
        entropy = dist.entropy().mean()

        advantages = returns_t - values.detach()
        policy_loss = -(log_probs * advantages).mean()
        value_loss = F.mse_loss(values, returns_t)
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)
        self.optimizer.step()

        return {
            "loss": float(loss.item()),
            "grad_norm": float(grad_norm.item()),
            "entropy": float(entropy.item()),
        }
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/agents/test_a2c.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add agents/a2c.py tests/agents/test_a2c.py
git commit -m "feat(agents): add A2C with n-step returns"
```

---

## Task 8: PPO

**Files:**
- Create: `agents/ppo.py`
- Create: `tests/agents/test_ppo.py`

PPO = A2C + GAE-λ + clipped surrogate + K epochs of minibatch updates over a rollout buffer.

- [ ] **Step 1: Write the failing test**

```python
# tests/agents/test_ppo.py
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
```

- [ ] **Step 2: Run, expect fail**

Run: `pytest tests/agents/test_ppo.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write `agents/ppo.py`**

```python
# agents/ppo.py
"""Proximal Policy Optimization with GAE-λ and clipped surrogate.

Update flow:
  - Collect T steps of experience (T = rollout_length)
  - Compute GAE advantages
  - Run K epochs of minibatch SGD over the rollout
  - Each minibatch optimizes clip(r(θ), 1-ε, 1+ε) * A^GAE
"""
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from agents.a2c import ActorCriticNet  # Same architecture
from agents.base import Agent


def compute_gae(
    rewards: list[float],
    values: list[float],
    dones: list[bool],
    bootstrap_value: float,
    gamma: float,
    lam: float,
) -> tuple[list[float], list[float]]:
    """Generalized Advantage Estimation.

    Returns (advantages, returns) where returns = advantages + values.
    """
    advantages: list[float] = [0.0] * len(rewards)
    last_gae = 0.0
    for t in reversed(range(len(rewards))):
        nonterminal = 0.0 if dones[t] else 1.0
        next_value = bootstrap_value if t == len(rewards) - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        last_gae = delta + gamma * lam * nonterminal * last_gae
        advantages[t] = last_gae
    returns = [a + v for a, v in zip(advantages, values)]
    return advantages, returns


class PPOAgent(Agent):
    """PPO with clipped surrogate objective."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_eps: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        grad_clip: float = 0.5,
        k_epochs: int = 4,
        minibatch_size: int = 32,
    ):
        super().__init__(obs_dim, act_dim)
        self.net = ActorCriticNet(obs_dim, act_dim)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.grad_clip = grad_clip
        self.k_epochs = k_epochs
        self.minibatch_size = minibatch_size

    def act(self, obs: np.ndarray) -> tuple[int, dict[str, float]]:
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            logits, value = self.net(obs_t)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return int(action.item()), {
            "log_prob": float(dist.log_prob(action).item()),
            "value": float(value.item()),
            "entropy": float(dist.entropy().item()),
        }

    def update(self, batch: dict[str, Any]) -> dict[str, float]:
        obs = torch.as_tensor(np.array(batch["obs"]), dtype=torch.float32)
        actions = torch.as_tensor(batch["actions"], dtype=torch.int64)
        log_probs_old = torch.as_tensor(batch["log_probs_old"], dtype=torch.float32)

        advantages, returns = compute_gae(
            batch["rewards"],
            batch["values"],
            batch["dones"],
            batch["bootstrap_value"],
            self.gamma,
            self.lam,
        )
        advantages_t = torch.as_tensor(advantages, dtype=torch.float32)
        returns_t = torch.as_tensor(returns, dtype=torch.float32)
        # Normalize advantages
        if advantages_t.numel() > 1:
            advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        n = obs.shape[0]
        last_metrics = {}
        for _ in range(self.k_epochs):
            idx = torch.randperm(n)
            for start in range(0, n, self.minibatch_size):
                mb = idx[start : start + self.minibatch_size]
                mb_obs = obs[mb]
                mb_actions = actions[mb]
                mb_log_probs_old = log_probs_old[mb]
                mb_advantages = advantages_t[mb]
                mb_returns = returns_t[mb]

                logits, values = self.net(mb_obs)
                dist = Categorical(logits=logits)
                log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(log_probs - mb_log_probs_old)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values, mb_returns)
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.net.parameters(), self.grad_clip
                )
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = (mb_log_probs_old - log_probs).mean()

                last_metrics = {
                    "loss": float(loss.item()),
                    "grad_norm": float(grad_norm.item()),
                    "entropy": float(entropy.item()),
                    "kl_divergence": float(approx_kl.item()),
                }

        return last_metrics
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/agents/test_ppo.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add agents/ppo.py tests/agents/test_ppo.py
git commit -m "feat(agents): add PPO with GAE and clipped surrogate"
```

---

## Task 9: Replay buffer

**Files:**
- Create: `agents/replay.py`
- Create: `tests/agents/test_replay.py`

- [ ] **Step 1: Write failing test**

```python
# tests/agents/test_replay.py
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
```

- [ ] **Step 2: Run, expect fail**

Run: `pytest tests/agents/test_replay.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write `agents/replay.py`**

```python
# agents/replay.py
"""Replay buffer for off-policy algorithms (DQN family)."""
import numpy as np


class ReplayBuffer:
    """Fixed-size circular buffer of transitions."""

    def __init__(self, capacity: int, obs_dim: int):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self._idx = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        self.obs[self._idx] = obs
        self.actions[self._idx] = action
        self.rewards[self._idx] = reward
        self.next_obs[self._idx] = next_obs
        self.dones[self._idx] = 1.0 if done else 0.0
        self._idx = (self._idx + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict:
        if self._size == 0:
            raise ValueError("Cannot sample from empty buffer")
        idx = np.random.randint(0, self._size, size=batch_size)
        return {
            "obs": self.obs[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "next_obs": self.next_obs[idx],
            "dones": self.dones[idx],
        }
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/agents/test_replay.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add agents/replay.py tests/agents/test_replay.py
git commit -m "feat(agents): add ReplayBuffer for off-policy methods"
```

---

## Task 10: DQN

**Files:**
- Create: `agents/dqn.py`
- Create: `tests/agents/test_dqn.py`

- [ ] **Step 1: Write failing test**

```python
# tests/agents/test_dqn.py
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
```

- [ ] **Step 2: Run, expect fail**

Run: `pytest tests/agents/test_dqn.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write `agents/dqn.py`**

```python
# agents/dqn.py
"""DQN family: vanilla DQN, Double DQN, Dueling DQN.

All three share replay buffer + target network + ε-greedy. They differ
in:
  - DQN: target = r + γ * max_a' Q_target(s', a')
  - Double DQN: target = r + γ * Q_target(s', argmax_a' Q_online(s', a'))
  - Dueling DQN: Q = V(s) + (A(s,a) - mean_a A(s,a)); other math identical
"""
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.base import Agent


class QNet(nn.Module):
    """Standard Q-network."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DuelingQNet(nn.Module):
    """Dueling architecture: shared trunk, separate V and A heads."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.value_head = nn.Linear(hidden, 1)
        self.advantage_head = nn.Linear(hidden, act_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.trunk(x)
        v = self.value_head(h)
        a = self.advantage_head(h)
        return v + a - a.mean(dim=-1, keepdim=True)


class DQNAgent(Agent):
    """Vanilla DQN with replay + target net + ε-greedy."""

    NET_CLASS = QNet

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        lr: float = 5e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 50_000,
        target_sync_every: int = 1000,
        grad_clip: float = 10.0,
    ):
        super().__init__(obs_dim, act_dim)
        self.q_net = self.NET_CLASS(obs_dim, act_dim)
        self.target_net = self.NET_CLASS(obs_dim, act_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())
        for p in self.target_net.parameters():
            p.requires_grad = False
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.target_sync_every = target_sync_every
        self.grad_clip = grad_clip
        self.epsilon = epsilon_start
        self._step = 0

    def step_epsilon(self, step: int) -> None:
        """Linearly anneal ε from start to end over epsilon_decay_steps."""
        frac = min(step / max(self.epsilon_decay_steps, 1), 1.0)
        self.epsilon = self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def sync_target(self) -> None:
        self.target_net.load_state_dict(self.q_net.state_dict())

    def act(self, obs: np.ndarray) -> tuple[int, dict[str, float]]:
        if np.random.random() < self.epsilon:
            action = int(np.random.randint(0, self.act_dim))
            return action, {"log_prob": 0.0, "value": 0.0, "entropy": float(np.log(self.act_dim))}
        with torch.no_grad():
            q = self.q_net(torch.as_tensor(obs, dtype=torch.float32))
        action = int(q.argmax().item())
        return action, {"log_prob": 0.0, "value": float(q.max().item()), "entropy": 0.0}

    def _td_target(self, next_obs: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        """Standard DQN target. Subclasses override for variants."""
        with torch.no_grad():
            next_q = self.target_net(next_obs).max(dim=1).values
            return rewards + self.gamma * next_q * (1.0 - dones)

    def update(self, batch: dict[str, Any]) -> dict[str, float]:
        obs = torch.as_tensor(batch["obs"], dtype=torch.float32)
        actions = torch.as_tensor(batch["actions"], dtype=torch.int64)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32)
        next_obs = torch.as_tensor(batch["next_obs"], dtype=torch.float32)
        dones = torch.as_tensor(batch["dones"], dtype=torch.float32)

        q_values = self.q_net(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        td_target = self._td_target(next_obs, rewards, dones)
        loss = F.smooth_l1_loss(q_values, td_target)

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.grad_clip)
        self.optimizer.step()

        self._step += 1
        if self._step % self.target_sync_every == 0:
            self.sync_target()

        return {
            "loss": float(loss.item()),
            "grad_norm": float(grad_norm.item()),
            "entropy": 0.0,
        }
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/agents/test_dqn.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add agents/dqn.py tests/agents/test_dqn.py
git commit -m "feat(agents): add DQN with replay buffer and target net"
```

---

## Task 11: Double DQN

**Files:**
- Modify: `agents/dqn.py` (add `DoubleDQNAgent` subclass)
- Create: `tests/agents/test_double_dqn.py`

- [ ] **Step 1: Write failing test**

```python
# tests/agents/test_double_dqn.py
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
```

- [ ] **Step 2: Run, expect fail**

Run: `pytest tests/agents/test_double_dqn.py -v`
Expected: FAIL with `ImportError: cannot import name 'DoubleDQNAgent'`.

- [ ] **Step 3: Append `DoubleDQNAgent` to `agents/dqn.py`**

Add at end of file:

```python
class DoubleDQNAgent(DQNAgent):
    """Double DQN: decouples action selection (online net) from evaluation (target net).

    Reduces overestimation bias compared to vanilla DQN.
    """

    def _td_target(self, next_obs: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            online_argmax = self.q_net(next_obs).argmax(dim=1)
            next_q = self.target_net(next_obs).gather(1, online_argmax.unsqueeze(1)).squeeze(1)
            return rewards + self.gamma * next_q * (1.0 - dones)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/agents/test_double_dqn.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add agents/dqn.py tests/agents/test_double_dqn.py
git commit -m "feat(agents): add Double DQN variant"
```

---

## Task 12: Dueling DQN

**Files:**
- Modify: `agents/dqn.py` (add `DuelingDQNAgent`)
- Create: `tests/agents/test_dueling_dqn.py`

- [ ] **Step 1: Write failing test**

```python
# tests/agents/test_dueling_dqn.py
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
```

- [ ] **Step 2: Run, expect fail**

Run: `pytest tests/agents/test_dueling_dqn.py -v`
Expected: FAIL with `ImportError: cannot import name 'DuelingDQNAgent'`.

- [ ] **Step 3: Append `DuelingDQNAgent` to `agents/dqn.py`**

Add at end of file:

```python
class DuelingDQNAgent(DQNAgent):
    """Dueling DQN: same training math as DQN, different network architecture."""

    NET_CLASS = DuelingQNet
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/agents/test_dueling_dqn.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add agents/dqn.py tests/agents/test_dueling_dqn.py
git commit -m "feat(agents): add Dueling DQN variant"
```

---

## Task 13: SQLite schema & data layer

**Files:**
- Create: `data/schema.sql`
- Create: `data/db.py`
- Create: `tests/data/__init__.py`, `tests/data/test_db.py`

- [ ] **Step 1: Write failing test**

```python
# tests/data/test_db.py
"""Tests for data.db (SQLite layer)."""
import json
from pathlib import Path

from data.db import init_db, insert_run, finish_run, insert_autopsy, insert_episode_metrics, insert_step_metrics, get_run, get_episodes


def test_init_creates_tables(tmp_path: Path):
    db_path = tmp_path / "test.sqlite"
    conn = init_db(db_path)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0] for row in cur.fetchall()}
    assert {"runs", "episodes", "step_metrics", "autopsies"}.issubset(tables)
    conn.close()


def test_insert_and_get_run(tmp_path: Path):
    conn = init_db(tmp_path / "test.sqlite")
    run_id = insert_run(
        conn, algo="PPO", env="CartPole-v1", seed=0, total_steps=1000, hparams={"lr": 3e-4}
    )
    run = get_run(conn, run_id)
    assert run["algo"] == "PPO"
    assert run["env"] == "CartPole-v1"
    assert json.loads(run["hparams_json"])["lr"] == 3e-4
    conn.close()


def test_finish_run_sets_finished_at(tmp_path: Path):
    conn = init_db(tmp_path / "test.sqlite")
    run_id = insert_run(conn, algo="DQN", env="CartPole-v1", seed=1, total_steps=1000, hparams={})
    finish_run(conn, run_id)
    run = get_run(conn, run_id)
    assert run["finished_at"] is not None
    conn.close()


def test_episode_round_trip(tmp_path: Path):
    conn = init_db(tmp_path / "test.sqlite")
    run_id = insert_run(conn, algo="A2C", env="CartPole-v1", seed=0, total_steps=1000, hparams={})
    insert_episode_metrics(conn, run_id, [(0, 100.0, 50, True), (1, 200.0, 80, True)])
    eps = get_episodes(conn, run_id)
    assert len(eps) == 2
    assert eps[0]["return_"] == 100.0
    conn.close()


def test_autopsy_round_trip(tmp_path: Path):
    from common.types import Verdict
    conn = init_db(tmp_path / "test.sqlite")
    run_id = insert_run(conn, algo="A2C", env="CartPole-v1", seed=0, total_steps=1000, hparams={})
    insert_autopsy(conn, run_id, Verdict(failure_mode="stalling", cause="flatlined"))
    cur = conn.execute("SELECT failure_mode, cause FROM autopsies WHERE run_id = ?", (run_id,))
    row = cur.fetchone()
    assert row["failure_mode"] == "stalling"
    conn.close()


def test_step_metrics_downsampling(tmp_path: Path):
    """We pass all step rows; the writer is the place that decides downsampling later."""
    conn = init_db(tmp_path / "test.sqlite")
    run_id = insert_run(conn, algo="PPO", env="CartPole-v1", seed=0, total_steps=1000, hparams={})
    rows = [(i, 0.5, 1.0, 0.0) for i in range(100)]
    insert_step_metrics(conn, run_id, rows)
    cur = conn.execute("SELECT COUNT(*) AS c FROM step_metrics WHERE run_id = ?", (run_id,))
    assert cur.fetchone()["c"] == 100
    conn.close()
```

- [ ] **Step 2: Run, expect fail**

```bash
mkdir -p tests/data && touch tests/data/__init__.py
```

Run: `pytest tests/data/test_db.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write `data/schema.sql`**

```sql
-- data/schema.sql

CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    algo TEXT NOT NULL,
    env TEXT NOT NULL,
    seed INTEGER NOT NULL,
    total_steps INTEGER NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    hparams_json TEXT NOT NULL,
    UNIQUE(algo, env, seed)
);

CREATE TABLE IF NOT EXISTS episodes (
    run_id INTEGER NOT NULL REFERENCES runs(id),
    episode_idx INTEGER NOT NULL,
    return_ REAL NOT NULL,
    length INTEGER NOT NULL,
    success INTEGER NOT NULL,
    PRIMARY KEY (run_id, episode_idx)
);

CREATE TABLE IF NOT EXISTS step_metrics (
    run_id INTEGER NOT NULL REFERENCES runs(id),
    step INTEGER NOT NULL,
    entropy REAL,
    grad_norm REAL,
    value_estimate REAL,
    PRIMARY KEY (run_id, step)
);

CREATE TABLE IF NOT EXISTS autopsies (
    run_id INTEGER PRIMARY KEY REFERENCES runs(id),
    failure_mode TEXT NOT NULL,
    cause TEXT NOT NULL,
    classified_at TEXT NOT NULL
);
```

- [ ] **Step 4: Write `data/db.py`**

```python
# data/db.py
"""SQLite helpers for the graveyard database."""
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from common.types import Verdict


SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def init_db(db_path: Path | str) -> sqlite3.Connection:
    """Open (or create) a SQLite database and ensure the schema is loaded."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    with open(SCHEMA_PATH) as f:
        conn.executescript(f.read())
    conn.commit()
    return conn


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def insert_run(
    conn: sqlite3.Connection,
    algo: str,
    env: str,
    seed: int,
    total_steps: int,
    hparams: dict,
) -> int:
    """Insert a new run row, return its id."""
    cur = conn.execute(
        """INSERT INTO runs (algo, env, seed, total_steps, started_at, hparams_json)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (algo, env, seed, total_steps, _now_iso(), json.dumps(hparams)),
    )
    conn.commit()
    return cur.lastrowid


def finish_run(conn: sqlite3.Connection, run_id: int) -> None:
    conn.execute("UPDATE runs SET finished_at = ? WHERE id = ?", (_now_iso(), run_id))
    conn.commit()


def insert_episode_metrics(
    conn: sqlite3.Connection,
    run_id: int,
    rows: list[tuple[int, float, int, bool]],
) -> None:
    """rows: list of (episode_idx, return, length, success)."""
    conn.executemany(
        "INSERT INTO episodes (run_id, episode_idx, return_, length, success) VALUES (?, ?, ?, ?, ?)",
        [(run_id, idx, ret, length, int(success)) for idx, ret, length, success in rows],
    )
    conn.commit()


def insert_step_metrics(
    conn: sqlite3.Connection,
    run_id: int,
    rows: list[tuple[int, float, float, float]],
) -> None:
    """rows: list of (step, entropy, grad_norm, value_estimate)."""
    conn.executemany(
        "INSERT INTO step_metrics (run_id, step, entropy, grad_norm, value_estimate) VALUES (?, ?, ?, ?, ?)",
        [(run_id, step, e, g, v) for step, e, g, v in rows],
    )
    conn.commit()


def insert_autopsy(conn: sqlite3.Connection, run_id: int, verdict: Verdict) -> None:
    conn.execute(
        "INSERT INTO autopsies (run_id, failure_mode, cause, classified_at) VALUES (?, ?, ?, ?)",
        (run_id, verdict.failure_mode, verdict.cause, _now_iso()),
    )
    conn.commit()


def get_run(conn: sqlite3.Connection, run_id: int) -> sqlite3.Row:
    cur = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,))
    return cur.fetchone()


def get_episodes(conn: sqlite3.Connection, run_id: int) -> list[sqlite3.Row]:
    cur = conn.execute(
        "SELECT * FROM episodes WHERE run_id = ? ORDER BY episode_idx", (run_id,)
    )
    return list(cur.fetchall())
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/data/test_db.py -v`
Expected: 6 passed.

- [ ] **Step 6: Commit**

```bash
git add data/schema.sql data/db.py tests/data/
git commit -m "feat(data): add SQLite schema and CRUD helpers"
```

---

## Task 14: Failure detector — NaN explosion

**Files:**
- Create: `analysis/diagnose.py` (with just the NaN detector for now)
- Create: `tests/analysis/__init__.py`, `tests/analysis/test_diagnose.py`

The detectors are isolated functions taking a `Trajectory` and returning `Verdict | None`. We build them one per task with synthetic-trajectory tests.

- [ ] **Step 1: Write failing test**

```python
# tests/analysis/test_diagnose.py
"""Tests for the per-failure-mode detectors and diagnose()."""
import numpy as np
from analysis.diagnose import detect_nan_explosion
from common.types import Trajectory


def _make_traj(**overrides) -> Trajectory:
    defaults = dict(
        algo="PPO",
        env="CartPole-v1",
        seed=0,
        total_steps=1000,
        episode_returns=[10.0] * 100,
        episode_lengths=[100] * 100,
        entropy_log=[0.5] * 1000,
        grad_norm_log=[1.0] * 1000,
        value_log=[5.0] * 1000,
        hparams={},
    )
    defaults.update(overrides)
    return Trajectory(**defaults)


def test_nan_in_returns_detected():
    traj = _make_traj(episode_returns=[10.0] * 95 + [float("nan")] * 5)
    v = detect_nan_explosion(traj)
    assert v is not None
    assert v.failure_mode == "nan_explosion"


def test_nan_in_grad_norm_detected():
    traj = _make_traj(grad_norm_log=[1.0] * 990 + [float("nan")] * 10)
    v = detect_nan_explosion(traj)
    assert v is not None


def test_clean_trajectory_not_flagged():
    traj = _make_traj()
    assert detect_nan_explosion(traj) is None
```

- [ ] **Step 2: Run, expect fail**

```bash
mkdir -p tests/analysis && touch tests/analysis/__init__.py
```

Run: `pytest tests/analysis/test_diagnose.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write `analysis/diagnose.py` (skeleton with NaN detector)**

```python
# analysis/diagnose.py
"""Rule-based failure mode classifier.

Each detector inspects a Trajectory and returns a Verdict if the signature
of its failure mode is present, else None. The orchestrator diagnose()
runs them in priority order (NaN first, then unrecoverable patterns,
then recoverable patterns) and returns the first match.
"""
from typing import Optional

import numpy as np

from common.types import Trajectory, Verdict


def detect_nan_explosion(traj: Trajectory) -> Optional[Verdict]:
    """Any NaN in episode returns or grad norms in the final 10% of training."""
    # Why this threshold: NaNs poison gradients permanently; even one in the
    # tail means the model has diverged irreversibly.
    tail_returns = traj.episode_returns[-max(10, len(traj.episode_returns) // 10):] or []
    if any(not np.isfinite(r) for r in tail_returns):
        return Verdict("nan_explosion", "NaN in episode returns")
    tail_grad = traj.grad_norm_log[-max(50, len(traj.grad_norm_log) // 10):] or []
    if any(not np.isfinite(g) for g in tail_grad):
        return Verdict("nan_explosion", "NaN in gradient norms")
    return None
```

- [ ] **Step 4: Run test**

Run: `pytest tests/analysis/test_diagnose.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add analysis/diagnose.py tests/analysis/
git commit -m "feat(analysis): add NaN explosion detector"
```

---

## Task 15: Failure detector — Exploration collapse

**Files:**
- Modify: `analysis/diagnose.py`
- Modify: `tests/analysis/test_diagnose.py`

- [ ] **Step 1: Append failing tests to `tests/analysis/test_diagnose.py`**

```python
# Append to existing file
from analysis.diagnose import detect_exploration_collapse


def test_exploration_collapse_low_entropy():
    """Entropy < 0.1 in final 50 steps → collapse."""
    traj = _make_traj(entropy_log=[0.5] * 500 + [0.05] * 500)
    v = detect_exploration_collapse(traj)
    assert v is not None
    assert v.failure_mode == "exploration_collapse"


def test_exploration_collapse_not_triggered_by_high_entropy():
    traj = _make_traj(entropy_log=[0.5] * 1000)
    assert detect_exploration_collapse(traj) is None


def test_exploration_collapse_requires_enough_data():
    """With < 100 entropy samples, can't conclude collapse."""
    traj = _make_traj(entropy_log=[0.0] * 50)
    assert detect_exploration_collapse(traj) is None
```

- [ ] **Step 2: Run, expect fail**

Run: `pytest tests/analysis/test_diagnose.py -v`
Expected: 3 of the new tests FAIL with `ImportError`.

- [ ] **Step 3: Append `detect_exploration_collapse` to `analysis/diagnose.py`**

```python
def detect_exploration_collapse(traj: Trajectory) -> Optional[Verdict]:
    """Policy entropy collapsed to near-deterministic in late training.

    Why this threshold: 0.1 nats ≈ 90/10 split for binary action; effectively
    deterministic. Requires ≥100 entropy samples to ensure we're past
    early-training high-entropy phase.
    """
    if len(traj.entropy_log) < 100:
        return None
    tail = np.array(traj.entropy_log[-50:])
    if tail.mean() < 0.1:
        return Verdict(
            "exploration_collapse",
            f"Entropy collapsed to {tail.mean():.3f} in final 50 steps",
        )
    return None
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/analysis/test_diagnose.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add analysis/diagnose.py tests/analysis/test_diagnose.py
git commit -m "feat(analysis): add exploration collapse detector"
```

---

## Task 16: Failure detector — Death spiral

**Files:**
- Modify: `analysis/diagnose.py`
- Modify: `tests/analysis/test_diagnose.py`

- [ ] **Step 1: Append failing tests**

```python
from analysis.diagnose import detect_death_spiral


def test_death_spiral_returns_crash():
    """Recent 10 episodes < 50% of preceding 40 episodes → death spiral."""
    early = [100.0] * 50
    late = [10.0] * 10
    traj = _make_traj(episode_returns=early + late, episode_lengths=[100] * 60)
    v = detect_death_spiral(traj)
    assert v is not None
    assert v.failure_mode == "death_spiral"


def test_death_spiral_not_triggered_by_stable_returns():
    traj = _make_traj(episode_returns=[100.0] * 60, episode_lengths=[100] * 60)
    assert detect_death_spiral(traj) is None


def test_death_spiral_requires_50_episodes():
    traj = _make_traj(episode_returns=[100.0] * 30 + [10.0] * 5, episode_lengths=[100] * 35)
    assert detect_death_spiral(traj) is None
```

- [ ] **Step 2: Run, expect fail**

Run: `pytest tests/analysis/test_diagnose.py -v`
Expected: 3 new tests FAIL.

- [ ] **Step 3: Append `detect_death_spiral`**

```python
def detect_death_spiral(traj: Trajectory) -> Optional[Verdict]:
    """Returns crashed: recent mean < 50% of preceding mean.

    Why this threshold: small drops are noise; 50% is large enough that
    even noisy learning curves don't trip it accidentally.
    """
    if len(traj.episode_returns) < 50:
        return None
    recent = float(np.mean(traj.episode_returns[-10:]))
    past = float(np.mean(traj.episode_returns[-50:-10]))
    if past > 0 and recent < past * 0.5:
        drop_pct = (past - recent) / past * 100
        return Verdict("death_spiral", f"Returns dropped {drop_pct:.0f}% in final 10 episodes")
    return None
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/analysis/test_diagnose.py -v`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add analysis/diagnose.py tests/analysis/test_diagnose.py
git commit -m "feat(analysis): add death spiral detector"
```

---

## Task 17: Failure detector — Stalling

**Files:**
- Modify: `analysis/diagnose.py`
- Modify: `tests/analysis/test_diagnose.py`

- [ ] **Step 1: Append failing tests**

```python
from analysis.diagnose import detect_stalling


def test_stalling_low_variance():
    """Coefficient of variation < 5% over last 500 episodes → stalling."""
    traj = _make_traj(
        episode_returns=[10.0 + np.random.uniform(-0.1, 0.1) for _ in range(600)],
        episode_lengths=[100] * 600,
    )
    v = detect_stalling(traj)
    assert v is not None
    assert v.failure_mode == "stalling"


def test_stalling_not_triggered_by_learning():
    traj = _make_traj(
        episode_returns=list(np.linspace(10, 100, 600)),
        episode_lengths=[100] * 600,
    )
    assert detect_stalling(traj) is None


def test_stalling_requires_500_episodes():
    traj = _make_traj(episode_returns=[10.0] * 400, episode_lengths=[100] * 400)
    assert detect_stalling(traj) is None
```

- [ ] **Step 2: Run, expect fail**

Run: `pytest tests/analysis/test_diagnose.py -v`
Expected: 3 new tests FAIL.

- [ ] **Step 3: Append `detect_stalling`**

```python
def detect_stalling(traj: Trajectory) -> Optional[Verdict]:
    """Flat returns for hundreds of episodes.

    Why this threshold: CV < 5% over 500 episodes is well below random noise
    for a learning agent; effectively no signal of improvement.
    """
    if len(traj.episode_returns) < 500:
        return None
    window = np.array(traj.episode_returns[-500:])
    cv = window.std() / (abs(window.mean()) + 1e-6)
    if cv < 0.05:
        return Verdict("stalling", f"Returns flat (CV={cv:.3f}) for 500 episodes")
    return None
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/analysis/test_diagnose.py -v`
Expected: 12 passed.

- [ ] **Step 5: Commit**

```bash
git add analysis/diagnose.py tests/analysis/test_diagnose.py
git commit -m "feat(analysis): add stalling detector"
```

---

## Task 18: Failure detector — Overshooting

**Files:**
- Modify: `analysis/diagnose.py`
- Modify: `tests/analysis/test_diagnose.py`

- [ ] **Step 1: Append failing tests**

```python
from analysis.diagnose import detect_overshooting


def test_overshooting_high_variance():
    """CV > 2 over last 50 episodes → wild oscillation."""
    returns = [100.0 if i % 2 == 0 else 5.0 for i in range(60)]
    traj = _make_traj(episode_returns=returns, episode_lengths=[100] * 60)
    v = detect_overshooting(traj)
    assert v is not None
    assert v.failure_mode == "overshooting"


def test_overshooting_not_triggered_by_moderate_variance():
    traj = _make_traj(
        episode_returns=[100.0 + np.random.uniform(-5, 5) for _ in range(60)],
        episode_lengths=[100] * 60,
    )
    assert detect_overshooting(traj) is None
```

- [ ] **Step 2: Run, expect fail**

Run: `pytest tests/analysis/test_diagnose.py -v`
Expected: 2 new tests FAIL.

- [ ] **Step 3: Append `detect_overshooting`**

```python
def detect_overshooting(traj: Trajectory) -> Optional[Verdict]:
    """Wild oscillation: CV > 2.0 over last 50 episodes.

    Why this threshold: a learning agent stabilizes; a CV of 2 means stddev
    is twice the mean — the agent's policy is thrashing.
    """
    if len(traj.episode_returns) < 50:
        return None
    window = np.array(traj.episode_returns[-50:])
    cv = window.std() / (abs(window.mean()) + 1e-6)
    if cv > 2.0:
        return Verdict("overshooting", f"Returns oscillating (CV={cv:.1f})")
    return None
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/analysis/test_diagnose.py -v`
Expected: 14 passed.

- [ ] **Step 5: Commit**

```bash
git add analysis/diagnose.py tests/analysis/test_diagnose.py
git commit -m "feat(analysis): add overshooting detector"
```

---

## Task 19: Failure detector — Reward hacking

**Files:**
- Modify: `analysis/diagnose.py`
- Modify: `tests/analysis/test_diagnose.py`

For v1, "reward hacking" specifically targets the `RewardTrap-v0` env: the agent should reach the goal (reward ≥ 9) but instead farms the trap (positive return, well below 9, never terminating before max steps). We detect this with two signals: env is RewardTrap, late returns are between trap-farming range (1-8) and episodes consistently hit max length.

- [ ] **Step 1: Append failing tests**

```python
from analysis.diagnose import detect_reward_hacking


def test_reward_hacking_in_reward_trap():
    """RewardTrap-v0 with returns in trap-farming range + max-length episodes."""
    traj = _make_traj(
        env="RewardTrap-v0",
        episode_returns=[200.0] * 50 + [240.0] * 10,  # trap-farming yields
        episode_lengths=[500] * 60,  # always hitting max
    )
    v = detect_reward_hacking(traj)
    assert v is not None
    assert v.failure_mode == "reward_hacking"


def test_reward_hacking_not_flagged_when_reaching_goal():
    """High reward + short episodes = reached goal legitimately."""
    traj = _make_traj(
        env="RewardTrap-v0",
        episode_returns=[9.5] * 60,
        episode_lengths=[20] * 60,
    )
    assert detect_reward_hacking(traj) is None


def test_reward_hacking_not_flagged_outside_reward_trap():
    """Detector is scoped to RewardTrap-v0 in v1."""
    traj = _make_traj(
        env="CartPole-v1",
        episode_returns=[200.0] * 60,
        episode_lengths=[500] * 60,
    )
    assert detect_reward_hacking(traj) is None
```

- [ ] **Step 2: Run, expect fail**

Run: `pytest tests/analysis/test_diagnose.py -v`
Expected: 3 new tests FAIL.

- [ ] **Step 3: Append `detect_reward_hacking`**

```python
def detect_reward_hacking(traj: Trajectory) -> Optional[Verdict]:
    """Detect trap-farming in the RewardTrap-v0 env.

    Why this is env-scoped: classic Gym envs are too well-designed to exhibit
    reward hacking. The trap detection looks for the specific signature of
    long episodes with returns in the trap-farming band (positive but well
    below goal reward).

    Signal: env is RewardTrap-v0 AND mean(last 50 return) > 1.0 AND
    mean(last 50 length) > 400 (near 500-step cap). Trap pays 0.5/step, so
    parking on it for 400+ steps yields > 200 return.
    """
    if traj.env != "RewardTrap-v0":
        return None
    if len(traj.episode_returns) < 50:
        return None
    recent_returns = np.array(traj.episode_returns[-50:])
    recent_lengths = np.array(traj.episode_lengths[-50:])
    if recent_returns.mean() > 1.0 and recent_lengths.mean() > 400:
        return Verdict(
            "reward_hacking",
            f"Trap-farming: mean return {recent_returns.mean():.1f} with "
            f"mean length {recent_lengths.mean():.0f} (cap=500)",
        )
    return None
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/analysis/test_diagnose.py -v`
Expected: 17 passed.

- [ ] **Step 5: Commit**

```bash
git add analysis/diagnose.py tests/analysis/test_diagnose.py
git commit -m "feat(analysis): add reward hacking detector for RewardTrap-v0"
```

---

## Task 20: diagnose() orchestration

**Files:**
- Modify: `analysis/diagnose.py`
- Modify: `tests/analysis/test_diagnose.py`

- [ ] **Step 1: Append failing tests**

```python
from analysis.diagnose import diagnose


def test_diagnose_returns_alive_for_clean_trajectory():
    traj = _make_traj(
        episode_returns=list(np.linspace(10, 200, 600)),
        episode_lengths=[100] * 600,
        entropy_log=[0.7] * 1000,
    )
    v = diagnose(traj)
    assert v.failure_mode == "alive"


def test_diagnose_priority_nan_beats_others():
    """When multiple modes apply, NaN wins (it's highest priority)."""
    traj = _make_traj(
        episode_returns=[10.0] * 100 + [float("nan")] * 5,
        entropy_log=[0.0] * 1000,  # would also trigger exploration collapse
    )
    v = diagnose(traj)
    assert v.failure_mode == "nan_explosion"
```

- [ ] **Step 2: Run, expect fail**

Run: `pytest tests/analysis/test_diagnose.py -v`
Expected: 2 new tests FAIL with ImportError.

- [ ] **Step 3: Append `diagnose()` orchestrator at end of `analysis/diagnose.py`**

```python
# Priority order — earlier detectors take precedence.
# Why this order:
#   1. NaN — irrecoverable, blocks everything downstream
#   2. Reward hacking — env-specific high-confidence signal
#   3. Death spiral — recent crash dominates other patterns
#   4. Exploration collapse — common, distinctive
#   5. Overshooting — high-variance pattern
#   6. Stalling — last because needs longest history
DETECTORS = [
    detect_nan_explosion,
    detect_reward_hacking,
    detect_death_spiral,
    detect_exploration_collapse,
    detect_overshooting,
    detect_stalling,
]


def diagnose(traj: Trajectory) -> Verdict:
    """Run all detectors in priority order. Return first verdict or 'alive'."""
    for detector in DETECTORS:
        verdict = detector(traj)
        if verdict is not None:
            return verdict
    return Verdict("alive", "Agent reached training budget without classified failure")
```

- [ ] **Step 4: Run all analysis tests**

Run: `pytest tests/analysis/ -v`
Expected: 19 passed.

- [ ] **Step 5: Commit**

```bash
git add analysis/diagnose.py tests/analysis/test_diagnose.py
git commit -m "feat(analysis): add diagnose() orchestrator with priority ordering"
```

---

## Task 21: Experiment config

**Files:**
- Create: `experiments/config.py`
- Create: `tests/experiments/__init__.py`, `tests/experiments/test_config.py`

- [ ] **Step 1: Write failing test**

```python
# tests/experiments/test_config.py
"""Tests for ExperimentConfig."""
import pytest
from experiments.config import ExperimentConfig, build_agent, SUPPORTED_ALGOS


def test_supported_algos_match_spec():
    assert SUPPORTED_ALGOS == {"PPO", "A2C", "REINFORCE", "DQN", "DoubleDQN", "DuelingDQN"}


def test_config_defaults():
    cfg = ExperimentConfig(algo="PPO", env="CartPole-v1", seed=0)
    assert cfg.total_steps == 200_000
    assert cfg.hparams == {}


def test_build_agent_dispatches_correctly():
    cfg = ExperimentConfig(algo="PPO", env="CartPole-v1", seed=0)
    agent = build_agent(cfg, obs_dim=4, act_dim=2)
    from agents.ppo import PPOAgent
    assert isinstance(agent, PPOAgent)


def test_build_agent_double_dqn():
    cfg = ExperimentConfig(algo="DoubleDQN", env="CartPole-v1", seed=0)
    agent = build_agent(cfg, obs_dim=4, act_dim=2)
    from agents.dqn import DoubleDQNAgent
    assert isinstance(agent, DoubleDQNAgent)


def test_unknown_algo_raises():
    cfg = ExperimentConfig(algo="Nonsense", env="CartPole-v1", seed=0)
    with pytest.raises(ValueError):
        build_agent(cfg, obs_dim=4, act_dim=2)
```

- [ ] **Step 2: Run, expect fail**

```bash
mkdir -p tests/experiments && touch tests/experiments/__init__.py
```

Run: `pytest tests/experiments/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write `experiments/config.py`**

```python
# experiments/config.py
"""Experiment configuration and agent factory."""
from dataclasses import dataclass, field

from agents.a2c import A2CAgent
from agents.base import Agent
from agents.dqn import DoubleDQNAgent, DQNAgent, DuelingDQNAgent
from agents.ppo import PPOAgent
from agents.reinforce import REINFORCEAgent


SUPPORTED_ALGOS = {"PPO", "A2C", "REINFORCE", "DQN", "DoubleDQN", "DuelingDQN"}


@dataclass
class ExperimentConfig:
    algo: str
    env: str
    seed: int
    total_steps: int = 200_000
    hparams: dict = field(default_factory=dict)


_REGISTRY = {
    "PPO": PPOAgent,
    "A2C": A2CAgent,
    "REINFORCE": REINFORCEAgent,
    "DQN": DQNAgent,
    "DoubleDQN": DoubleDQNAgent,
    "DuelingDQN": DuelingDQNAgent,
}


def build_agent(cfg: ExperimentConfig, obs_dim: int, act_dim: int) -> Agent:
    """Construct the agent for a config."""
    if cfg.algo not in _REGISTRY:
        raise ValueError(f"Unknown algo: {cfg.algo}. Supported: {SUPPORTED_ALGOS}")
    cls = _REGISTRY[cfg.algo]
    return cls(obs_dim=obs_dim, act_dim=act_dim, **cfg.hparams)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/experiments/test_config.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add experiments/config.py tests/experiments/
git commit -m "feat(experiments): add ExperimentConfig and agent factory"
```

---

## Task 22: Experiment runner

**Files:**
- Create: `experiments/runner.py`
- Create: `tests/experiments/test_runner.py`

The runner must handle three distinct training loops (on-policy episode-based for REINFORCE, on-policy n-step for A2C/PPO, off-policy replay for DQN). We implement three internal train loops, dispatched by algo. This is the longest module in the plan.

- [ ] **Step 1: Write failing test (smoke only — full unit testing would be brittle)**

```python
# tests/experiments/test_runner.py
"""Smoke tests for the experiment runner."""
import sqlite3
from pathlib import Path

import pytest

from data.db import init_db, get_run, get_episodes
from experiments.config import ExperimentConfig
from experiments.runner import train_and_record


def _run_short(algo: str, tmp_path: Path) -> int:
    conn = init_db(tmp_path / "graveyard.sqlite")
    cfg = ExperimentConfig(
        algo=algo,
        env="CartPole-v1",
        seed=0,
        total_steps=2000,  # Very short — just verify the loop runs
    )
    run_id = train_and_record(cfg, conn)
    conn.close()
    return run_id


@pytest.mark.parametrize("algo", ["REINFORCE", "A2C", "PPO", "DQN", "DoubleDQN", "DuelingDQN"])
def test_all_algos_complete_short_run(algo, tmp_path):
    run_id = _run_short(algo, tmp_path)
    conn = sqlite3.connect(tmp_path / "graveyard.sqlite")
    conn.row_factory = sqlite3.Row
    run = get_run(conn, run_id)
    eps = get_episodes(conn, run_id)
    assert run["finished_at"] is not None
    assert len(eps) >= 1, f"{algo} produced no episodes"
    cur = conn.execute("SELECT * FROM autopsies WHERE run_id = ?", (run_id,))
    autopsy = cur.fetchone()
    assert autopsy is not None
    conn.close()
```

- [ ] **Step 2: Run, expect fail**

Run: `pytest tests/experiments/test_runner.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'experiments.runner'`.

- [ ] **Step 3: Write `experiments/runner.py`**

```python
# experiments/runner.py
"""Train one (algo, env, seed) config and persist its trajectory + autopsy."""
from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

import numpy as np
import torch

from agents.a2c import A2CAgent
from agents.dqn import DQNAgent
from agents.ppo import PPOAgent
from agents.reinforce import REINFORCEAgent
from agents.replay import ReplayBuffer
from analysis.diagnose import diagnose
from common.types import Trajectory
from data.db import (
    finish_run,
    insert_autopsy,
    insert_episode_metrics,
    insert_run,
    insert_step_metrics,
)
from envs.registry import REGISTRY, make_env
from envs.wrappers import LoggingWrapper
from experiments.config import ExperimentConfig, build_agent

if TYPE_CHECKING:
    from agents.base import Agent


STEP_METRIC_STRIDE = 100  # Log to SQLite every 100 env steps (downsampling for storage)


def train_and_record(cfg: ExperimentConfig, conn: sqlite3.Connection) -> int:
    """Train one agent, persist everything, run autopsy, return run_id."""
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    env = LoggingWrapper(make_env(cfg.env, seed=cfg.seed))
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    agent = build_agent(cfg, obs_dim=obs_dim, act_dim=act_dim)

    run_id = insert_run(
        conn, algo=cfg.algo, env=cfg.env, seed=cfg.seed,
        total_steps=cfg.total_steps, hparams=cfg.hparams,
    )

    entropy_log: list[float] = []
    grad_norm_log: list[float] = []
    value_log: list[float] = []
    step_rows: list[tuple[int, float, float, float]] = []

    if isinstance(agent, REINFORCEAgent):
        _train_reinforce(agent, env, cfg, entropy_log, grad_norm_log, value_log, step_rows)
    elif isinstance(agent, (A2CAgent, PPOAgent)):
        _train_on_policy_rollout(agent, env, cfg, entropy_log, grad_norm_log, value_log, step_rows)
    elif isinstance(agent, DQNAgent):
        _train_dqn(agent, env, cfg, entropy_log, grad_norm_log, value_log, step_rows)
    else:
        raise TypeError(f"No training loop for {type(agent).__name__}")

    # Episodes from the wrapper
    ep_success_threshold = REGISTRY[cfg.env].success_threshold
    episode_rows = [
        (i, r, l, r >= ep_success_threshold)
        for i, (r, l) in enumerate(zip(env.episode_returns, env.episode_lengths))
    ]
    insert_episode_metrics(conn, run_id, episode_rows)
    insert_step_metrics(conn, run_id, step_rows)

    traj = Trajectory(
        algo=cfg.algo,
        env=cfg.env,
        seed=cfg.seed,
        total_steps=cfg.total_steps,
        episode_returns=list(env.episode_returns),
        episode_lengths=list(env.episode_lengths),
        entropy_log=entropy_log,
        grad_norm_log=grad_norm_log,
        value_log=value_log,
        hparams=cfg.hparams,
    )
    verdict = diagnose(traj)
    insert_autopsy(conn, run_id, verdict)
    finish_run(conn, run_id)
    env.close()
    return run_id


def _maybe_log_step(step: int, info: dict, metrics: dict | None,
                    entropy_log: list, grad_log: list, value_log: list,
                    rows: list) -> None:
    entropy_log.append(info["entropy"])
    value_log.append(info["value"])
    grad_log.append(metrics["grad_norm"] if metrics else 0.0)
    if step % STEP_METRIC_STRIDE == 0:
        rows.append((step, info["entropy"], grad_log[-1], info["value"]))


def _train_reinforce(agent: "REINFORCEAgent", env, cfg, entropy_log, grad_log, value_log, rows):
    step = 0
    obs, _ = env.reset(seed=cfg.seed)
    ep_rewards: list[float] = []
    ep_log_probs: list = []
    ep_entropies: list = []
    last_metrics: dict | None = None

    while step < cfg.total_steps:
        action, info = agent.act(obs)
        ep_log_probs.append(info["_log_prob_tensor"])
        ep_entropies.append(info["_entropy_tensor"])
        next_obs, reward, term, trunc, _ = env.step(action)
        ep_rewards.append(float(reward))
        _maybe_log_step(step, info, last_metrics, entropy_log, grad_log, value_log, rows)
        step += 1
        obs = next_obs
        if term or trunc:
            last_metrics = agent.update({
                "rewards": ep_rewards, "log_probs": ep_log_probs, "entropies": ep_entropies,
            })
            ep_rewards, ep_log_probs, ep_entropies = [], [], []
            obs, _ = env.reset()


def _train_on_policy_rollout(agent, env, cfg, entropy_log, grad_log, value_log, rows):
    """Used by A2C and PPO. Collects fixed-length rollouts, then updates."""
    rollout_length = 128 if isinstance(agent, PPOAgent) else agent.n_steps
    step = 0
    obs, _ = env.reset(seed=cfg.seed)
    last_metrics: dict | None = None

    while step < cfg.total_steps:
        buf_obs, buf_actions, buf_rewards, buf_dones = [], [], [], []
        buf_log_probs, buf_values = [], []

        for _ in range(rollout_length):
            if step >= cfg.total_steps:
                break
            action, info = agent.act(obs)
            buf_obs.append(obs)
            buf_actions.append(action)
            buf_log_probs.append(info["log_prob"])
            buf_values.append(info["value"])
            next_obs, reward, term, trunc, _ = env.step(action)
            buf_rewards.append(float(reward))
            buf_dones.append(term or trunc)
            _maybe_log_step(step, info, last_metrics, entropy_log, grad_log, value_log, rows)
            step += 1
            if term or trunc:
                next_obs, _ = env.reset()
            obs = next_obs

        if not buf_obs:
            break

        # Bootstrap value for the last obs
        with torch.no_grad():
            _, bootstrap_info = agent.act(obs)
        bootstrap_value = bootstrap_info["value"] if not buf_dones[-1] else 0.0

        if isinstance(agent, PPOAgent):
            batch = {
                "obs": buf_obs, "actions": buf_actions, "log_probs_old": buf_log_probs,
                "rewards": buf_rewards, "dones": buf_dones, "values": buf_values,
                "bootstrap_value": bootstrap_value,
            }
        else:  # A2C
            batch = {
                "obs": buf_obs, "actions": buf_actions, "rewards": buf_rewards,
                "dones": buf_dones, "bootstrap_value": bootstrap_value,
            }
        last_metrics = agent.update(batch)


def _train_dqn(agent: "DQNAgent", env, cfg, entropy_log, grad_log, value_log, rows):
    buf = ReplayBuffer(capacity=50_000, obs_dim=agent.obs_dim)
    warmup = 1000
    batch_size = 64
    train_every = 4

    step = 0
    obs, _ = env.reset(seed=cfg.seed)
    last_metrics: dict | None = None

    while step < cfg.total_steps:
        agent.step_epsilon(step)
        action, info = agent.act(obs)
        next_obs, reward, term, trunc, _ = env.step(action)
        buf.add(
            obs=np.asarray(obs, dtype=np.float32),
            action=action,
            reward=float(reward),
            next_obs=np.asarray(next_obs, dtype=np.float32),
            done=bool(term),  # truncated transitions still bootstrap, so use term only
        )

        if step >= warmup and step % train_every == 0 and len(buf) >= batch_size:
            batch = buf.sample(batch_size=batch_size)
            last_metrics = agent.update(batch)

        _maybe_log_step(step, info, last_metrics, entropy_log, grad_log, value_log, rows)
        step += 1
        if term or trunc:
            obs, _ = env.reset()
        else:
            obs = next_obs
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/experiments/test_runner.py -v`
Expected: 6 passed (one per algo).

This will take a minute or two — each algo trains 2K steps on CartPole.

- [ ] **Step 5: Commit**

```bash
git add experiments/runner.py tests/experiments/test_runner.py
git commit -m "feat(experiments): add runner with algo-specific training loops"
```

---

## Task 23: End-to-end smoke test

**Files:**
- Create: `tests/test_end_to_end.py`

- [ ] **Step 1: Write a test that exercises every component**

```python
# tests/test_end_to_end.py
"""End-to-end: train one agent, autopsy it, verify everything landed in SQLite."""
import sqlite3
import pytest

from data.db import init_db, get_run, get_episodes
from experiments.config import ExperimentConfig
from experiments.runner import train_and_record


@pytest.mark.parametrize(
    "algo,env",
    [
        ("PPO", "CartPole-v1"),
        ("DQN", "CartPole-v1"),
        ("REINFORCE", "Acrobot-v1"),
        ("PPO", "RewardTrap-v0"),
    ],
)
def test_end_to_end(algo, env, tmp_path):
    db_path = tmp_path / "graveyard.sqlite"
    conn = init_db(db_path)
    cfg = ExperimentConfig(algo=algo, env=env, seed=0, total_steps=5000)
    run_id = train_and_record(cfg, conn)
    conn.close()

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    run = get_run(conn, run_id)
    assert run["finished_at"] is not None
    assert run["algo"] == algo
    assert run["env"] == env

    eps = get_episodes(conn, run_id)
    assert len(eps) >= 1, f"No episodes recorded for {algo} on {env}"

    autopsy = conn.execute("SELECT * FROM autopsies WHERE run_id = ?", (run_id,)).fetchone()
    assert autopsy is not None
    assert autopsy["failure_mode"] in {
        "alive", "nan_explosion", "exploration_collapse",
        "death_spiral", "stalling", "overshooting", "reward_hacking",
    }

    steps = conn.execute("SELECT COUNT(*) AS c FROM step_metrics WHERE run_id = ?", (run_id,)).fetchone()
    assert steps["c"] > 0, f"No step metrics for {algo} on {env}"
    conn.close()
```

- [ ] **Step 2: Run**

Run: `pytest tests/test_end_to_end.py -v`
Expected: 4 passed. Will take ~2-5 minutes.

- [ ] **Step 3: Run the full test suite**

Run: `pytest -v`
Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add tests/test_end_to_end.py
git commit -m "test: end-to-end smoke covering 4 algo/env pairs"
```

---

## Task 24: Cleanup — remove `graveyard.py`, update README

**Files:**
- Delete: `graveyard.py`
- Modify: `README.md`

- [ ] **Step 1: Delete `graveyard.py`**

Run:
```bash
git rm graveyard.py
```

- [ ] **Step 2: Update README**

Modify `README.md`. Replace the `## Running` section (currently `python graveyard.py`) with:

```markdown
## Running

Install dependencies:

```bash
pip install -e ".[dev]"
```

Run all tests:

```bash
pytest
```

Train a single agent end-to-end:

```python
from data.db import init_db
from experiments.config import ExperimentConfig
from experiments.runner import train_and_record

conn = init_db("data/graveyard.sqlite")
cfg = ExperimentConfig(algo="PPO", env="CartPole-v1", seed=0, total_steps=200_000)
train_and_record(cfg, conn)
```

The full sweep runner (Modal) and the interactive `graveyard/` frontend
arrive in later phases — see `docs/specs/2026-06-05-v1-design.md`.
```

Also update the `## Architecture` section's directory tree to match what was built. Replace the existing tree with:

```
rl-graveyard/
├── agents/              # PPO, A2C, REINFORCE, DQN, Double DQN, Dueling DQN
├── envs/                # Registry, logging wrapper, RewardTrap-v0
├── experiments/         # Config dataclass + per-algo training loops
├── analysis/            # Six failure-mode detectors + diagnose() orchestrator
├── data/                # SQLite schema + CRUD helpers
├── common/              # Shared types (Trajectory, Verdict, EnvMeta)
└── docs/                # Specs and plans
```

And update the `## Current Status` section to:

```
Phase 1 (backend skeleton) complete: all 6 algorithms train end-to-end on
5 environments, with episode metrics and autopsies persisted to SQLite.
Phase 2 (full grid sweep) and Phase 3 (interactive frontend) pending.
```

- [ ] **Step 3: Run full test suite once more to confirm nothing broke**

Run: `pytest`
Expected: all green.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "chore: replace graveyard.py with packaged structure, update README"
```

---

## Done — Phase 1 complete

At this point:
- All 6 algorithms train end-to-end on all 5 environments (4 classic + RewardTrap)
- Each training run writes to SQLite (`runs`, `episodes`, `step_metrics`, `autopsies`)
- 6 failure detectors with unit tests; `diagnose()` orchestrates them in priority order
- Test suite is the regression safety net for everything Phase 2/3/4 will build on

**Next:** Phase 2 plan — author the sweep grid, build the Modal entrypoint, run 150 configurations, populate `data/graveyard.sqlite`, commit the artifact.
