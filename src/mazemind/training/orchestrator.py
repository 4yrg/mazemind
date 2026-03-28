"""Training orchestrator with episode management and metric collection."""

from __future__ import annotations

from typing import Optional, Generator

import numpy as np

from mazemind.agents.base_agent import BaseAgent
from mazemind.agents.dyna_q import DynaQAgent
from mazemind.agents.sarsa import SarsaAgent
from mazemind.envs.maze_parser import MazeData
from mazemind.envs.micromouse_env import MicromouseEnv
from mazemind.utils.metrics import EpisodeMetrics, TrainingMetrics


def train_agent(
    agent: BaseAgent,
    env: MicromouseEnv,
    n_episodes: int = 500,
    max_steps: int = 1000,
    alpha: float = 0.1,
    gamma: float = 0.99,
    seed: Optional[int] = None,
    agent_name: str = "",
    maze_name: str = "",
) -> TrainingMetrics:
    if seed is not None:
        np.random.seed(seed)
        import random as _random
        _random.seed(seed)

    metrics = TrainingMetrics(agent_name=agent_name, maze_name=maze_name)

    for ep in range(n_episodes):
        state = env.reset()
        si = env.state_to_index(state)
        total_reward = 0.0
        done = False

        if isinstance(agent, SarsaAgent):
            action = agent.select_action(si)

        for step in range(max_steps):
            if not isinstance(agent, SarsaAgent):
                action = agent.select_action(si)

            result = env.step(action)
            nsi = env.state_to_index(result.state)
            total_reward += result.reward

            if isinstance(agent, SarsaAgent):
                next_action = agent.select_action(nsi)
                agent.update(
                    si, action, result.reward, nsi, alpha, gamma, result.done,
                    next_action=next_action,
                )
                action = next_action
            else:
                agent.update(si, action, result.reward, nsi, alpha, gamma, result.done)

            si = nsi

            if result.done:
                break

        agent.decay_epsilon()

        metrics.add_episode(EpisodeMetrics(
            episode=ep,
            total_reward=total_reward,
            steps=step + 1,
            success=result.done,
            epsilon=agent.epsilon,
        ))

    return metrics


def train_agent_generator(
    agent: BaseAgent,
    env: MicromouseEnv,
    n_episodes: int = 500,
    max_steps: int = 1000,
    alpha: float = 0.1,
    gamma: float = 0.99,
    seed: Optional[int] = None,
    agent_name: str = "",
    maze_name: str = "",
) -> Generator[tuple[int, EpisodeMetrics, BaseAgent, MicromouseEnv], None, TrainingMetrics]:
    if seed is not None:
        np.random.seed(seed)
        import random as _random
        _random.seed(seed)

    metrics = TrainingMetrics(agent_name=agent_name, maze_name=maze_name)

    for ep in range(n_episodes):
        state = env.reset()
        si = env.state_to_index(state)
        total_reward = 0.0
        done = False

        if isinstance(agent, SarsaAgent):
            action = agent.select_action(si)

        for step in range(max_steps):
            if not isinstance(agent, SarsaAgent):
                action = agent.select_action(si)

            result = env.step(action)
            nsi = env.state_to_index(result.state)
            total_reward += result.reward

            if isinstance(agent, SarsaAgent):
                next_action = agent.select_action(nsi)
                agent.update(
                    si, action, result.reward, nsi, alpha, gamma, result.done,
                    next_action=next_action,
                )
                action = next_action
            else:
                agent.update(si, action, result.reward, nsi, alpha, gamma, result.done)

            si = nsi
            if result.done:
                break

        agent.decay_epsilon()

        ep_metrics = EpisodeMetrics(
            episode=ep,
            total_reward=total_reward,
            steps=step + 1,
            success=result.done,
            epsilon=agent.epsilon,
        )
        metrics.add_episode(ep_metrics)
        yield ep, ep_metrics, agent, env

    return metrics


def extract_optimal_path(
    agent: BaseAgent,
    env: MicromouseEnv,
    max_steps: int = 500,
) -> list[tuple[int, int]]:
    state = env.reset()
    si = env.state_to_index(state)
    path = [state]
    visited = {state}

    for _ in range(max_steps):
        action = agent.select_greedy_action(si)
        result = env.step(action)
        path.append(result.state)

        if result.done:
            break

        if result.state in visited:
            break
        visited.add(result.state)

        si = env.state_to_index(result.state)

    return path
