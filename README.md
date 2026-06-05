<p align="center">
  <img src="banner.png" alt="RL Graveyard Banner" width="100%">
</p>

# RL Graveyard: A Systematic Catalog of Reinforcement Learning Failure Modes

An empirical study of where and how reinforcement learning algorithms fail across diverse environments.

## Motivation

The reinforcement learning literature overwhelmingly reports success curves and convergence plots. The failures, which constitute the majority of training runs in practice, are rarely documented or analyzed. This project attempts to correct that imbalance by training hundreds of RL agents, classifying their exact failure modes, and visualizing the results.

## Research Question

Which RL algorithms fail on which environments, and why? Can we predict failure from training dynamics?

## Failure Taxonomy

| Code | Name | Description |
|------|------|-------------|
| RH | Reward Hacking | Agent finds unintended shortcut to high reward |
| CF | Catastrophic Forgetting | Agent forgets earlier skills |
| EC | Exploration Collapse | Policy collapses to single action, entropy flatlines |
| DS | Death Spiral | Loss diverges, returns crash |
| NE | NaN Explosion | Gradients or losses become NaN |
| SI | Simulation Overfitting | Works in training distribution, fails on slight perturbation |
| ST | Stalling | Plateaus indefinitely at suboptimal performance |
| OS | Overshooting | Oscillates around optimal, never converges |

## Methodology

For each algorithm-environment pair, we run 5 seeds for 1M steps, logging:
- Episode returns
- Policy entropy
- Value function estimates
- Gradient norms
- Action distributions
- Environment-specific success metrics

Failure classification uses a rule-based diagnostic applied to the training trajectory post-hoc.

## Architecture

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

## Current Status

Phase 1 (backend skeleton) complete: all 6 algorithms (PPO, A2C, REINFORCE, DQN,
Double DQN, Dueling DQN) train end-to-end on 5 environments (CartPole-v1,
Acrobot-v1, MountainCar-v0, LunarLander-v3, RewardTrap-v0), with episode metrics
and autopsies persisted to SQLite. Phase 2 (full grid sweep on Modal) and
Phase 3 (interactive frontend) pending.

## Dependencies

```
torch>=2.0.0
gymnasium[box2d]>=0.29.0
numpy>=1.24.0
pyyaml>=6.0
```

## Running

Install dependencies (requires Python 3.10+):

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

## Citation

If you use this work, please cite:
```
@software{rl_graveyard_2026,
  author = {Vardhan, Manas},
  title = {RL Graveyard: A Systematic Catalog of RL Failure Modes},
  year = {2026},
  url = {https://github.com/ManasVardhan/rl-graveyard}
}
```

## License

MIT
