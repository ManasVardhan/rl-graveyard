# RL Graveyard -- Where Agents Go to Die

A visual catalog of every way reinforcement learning algorithms fail. Trains 500+ agents across environments and surfaces the beautiful, tragic ways they collapse.

## The Hook

"I trained 500 RL agents and killed 487 of them. Here is where they died."

## Concept

Every RL paper shows the success curve. Nobody shows the death curve. This project:

1. Trains 10 algorithms (PPO, SAC, TD3, A2C, DQN, etc.) across 50+ Gymnasium/MuJoCo/Procgen environments
2. Logs exact failure mode: reward hacking, catastrophic forgetting, exploration collapse, death spiral, NaN explosion
3. Builds an interactive 3D graveyard where each dead agent is a tombstone with cause of death
4. Surfaces which algorithms die where, and why

## Stack

- CleanRL for fast, readable algorithm implementations
- WandB for experiment tracking
- Three.js for the 3D graveyard visualization
- Python + PyTorch for training pipeline

## Architecture

```
rl-graveyard/
├── agents/              # CleanRL-style algorithm implementations
├── envs/                # Environment wrappers + custom failure detectors
├── graveyard/           # 3D visualization (Three.js + D3.js)
├── experiments/         # Training configs per algorithm-env pair
├── analysis/            # Failure mode classification pipeline
└── data/                # SQLite DB of all training runs
```

## Failure Taxonomy

| Code | Name | Description |
|------|------|-------------|
| RH | Reward Hacking | Agent finds unintended shortcut to high reward |
| CF | Catastrophic Forgetting | Agent forgets earlier skills |
| EC | Exploration Collapse | Policy collapses to single action |
| DS | Death Spiral | Loss diverges, entropy crashes |
| NE | NaN Explosion | Gradients or losses become NaN |
| SI | Sim-overfitting | Works in sim, fails on tiny env change |
| ST | Stalling | Plateaus forever at suboptimal |
| OS | Overshooting | Oscillates around optimal, never settles |

## Training Protocol

Each algorithm x environment pair gets 5 seeds x 1M steps. We log:
- Episode returns
- Policy entropy
- Value function estimates
- Gradient norms
- Action distributions
- Environment-specific success metrics

## Visualization

Interactive 3D graveyard where:
- X axis: Algorithm family
- Y axis: Environment difficulty
- Z axis: Failure mode category
- Each tombstone: one dead agent, hover for autopsy

## Viral Mechanics

- Auto-generate "death certificates" for agents (shareable PNGs)
- Leaderboard: "Most Survivable Algorithm" by environment category
- Weekly "autopsy reports" on X
- Users can submit their own dead agents

## License

MIT -- build in public, share the carnage.
