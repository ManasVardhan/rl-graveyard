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


def test_default_hparams_are_logged(tmp_path):
    """Even when cfg.hparams is empty, the agent's defaults are persisted."""
    import json
    from data.db import init_db

    conn = init_db(tmp_path / "graveyard.sqlite")
    cfg = ExperimentConfig(algo="PPO", env="CartPole-v1", seed=0, total_steps=500)
    run_id = train_and_record(cfg, conn)
    run = get_run(conn, run_id)
    hparams = json.loads(run["hparams_json"])
    # PPO defaults: lr=3e-4, gamma=0.99, lam=0.95, clip_eps=0.2 — must all be present
    assert hparams["lr"] == 3e-4
    assert hparams["gamma"] == 0.99
    assert hparams["lam"] == 0.95
    assert hparams["clip_eps"] == 0.2
    conn.close()


def test_explicit_hparams_override_defaults(tmp_path):
    """cfg.hparams takes precedence over agent defaults."""
    import json
    from data.db import init_db

    conn = init_db(tmp_path / "graveyard.sqlite")
    cfg = ExperimentConfig(
        algo="PPO", env="CartPole-v1", seed=0, total_steps=500,
        hparams={"lr": 1e-2},
    )
    run_id = train_and_record(cfg, conn)
    run = get_run(conn, run_id)
    hparams = json.loads(run["hparams_json"])
    assert hparams["lr"] == 1e-2
    assert hparams["gamma"] == 0.99  # untouched default still present
    conn.close()
