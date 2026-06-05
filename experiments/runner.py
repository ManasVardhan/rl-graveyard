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
