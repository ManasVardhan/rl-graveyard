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
