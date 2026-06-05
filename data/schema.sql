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
