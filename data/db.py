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
