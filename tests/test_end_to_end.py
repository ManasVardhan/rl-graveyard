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
