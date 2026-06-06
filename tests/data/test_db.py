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
