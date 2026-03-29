"""Training orchestrator with episode management and metric collection."""

from __future__ import annotations

from typing import Optional, Generator

import numpy as np

from mazemind.agents.base_agent import BaseAgent
from mazemind.agents.dyna_q import DynaQAgent
from mazemind.agents.sarsa import SarsaAgent
from mazemind.envs.maze_parser import MazeData
from mazemind.envs.micromouse_env import MicromouseEnv
from mazemind.utils.metrics import EpisodeMetrics, EpisodeSnapshot, TrainingMetrics


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


def train_with_snapshots(
    agent: BaseAgent,
    env: MicromouseEnv,
    n_episodes: int = 500,
    max_steps: int = 1000,
    alpha: float = 0.1,
    gamma: float = 0.99,
    seed: Optional[int] = None,
    agent_name: str = "",
    maze_name: str = "",
    snapshot_episodes: Optional[list[int]] = None,
) -> tuple[TrainingMetrics, list[EpisodeSnapshot], list[list[tuple[int, int]]], list[int]]:
    if seed is not None:
        np.random.seed(seed)
        import random as _random
        _random.seed(seed)

    if snapshot_episodes is None:
        snapshot_episodes = [0, 1, 5, 10, 25, 50, 100, 200, 499]
    snapshot_set = set(snapshot_episodes)

    metrics = TrainingMetrics(agent_name=agent_name, maze_name=maze_name)
    snapshots: list[EpisodeSnapshot] = []
    all_trajectories: list[list[tuple[int, int]]] = []
    cumulative_visits = np.zeros((env.maze.size, env.maze.size))
    exploration_history: list[int] = []

    is_dyna = isinstance(agent, DynaQAgent)

    for ep in range(n_episodes):
        state = env.reset()
        si = env.state_to_index(state)
        total_reward = 0.0
        trajectory = [state]

        if isinstance(agent, SarsaAgent):
            action = agent.select_action(si)

        for step in range(max_steps):
            if not isinstance(agent, SarsaAgent):
                action = agent.select_action(si)

            result = env.step(action)
            nsi = env.state_to_index(result.state)
            total_reward += result.reward
            trajectory.append(result.state)

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

        for pos in trajectory:
            cumulative_visits[pos[0], pos[1]] += 1

        ep_metrics = EpisodeMetrics(
            episode=ep,
            total_reward=total_reward,
            steps=step + 1,
            success=result.done,
            epsilon=agent.epsilon,
        )
        metrics.add_episode(ep_metrics)
        all_trajectories.append(trajectory)

        explored_cells = int(np.count_nonzero(cumulative_visits))
        exploration_history.append(explored_cells)

        if ep in snapshot_set:
            model_size = len(agent.model) if is_dyna else 0
            planning = agent.n_planning_steps if is_dyna else 0
            snapshots.append(EpisodeSnapshot(
                episode=ep,
                path=trajectory,
                visit_counts=cumulative_visits.copy(),
                model_size=model_size,
                planning_steps=planning,
                q_table_snapshot=agent.q_table.copy(),
                success=result.done,
                steps=step + 1,
                reward=total_reward,
                epsilon=agent.epsilon,
            ))

    return metrics, snapshots, all_trajectories, exploration_history


def train_both_generator(
    dq_agent: DynaQAgent,
    ss_agent: SarsaAgent,
    dq_env: MicromouseEnv,
    ss_env: MicromouseEnv,
    n_episodes: int = 500,
    max_steps: int = 1000,
    alpha: float = 0.1,
    gamma: float = 0.99,
    seed: Optional[int] = None,
) -> Generator[tuple[int, DynaQAgent, MicromouseEnv, SarsaAgent, MicromouseEnv, EpisodeMetrics, EpisodeMetrics, list, list], None, None]:
    if seed is not None:
        np.random.seed(seed)
        _random.seed(seed)

    for ep in range(n_episodes):
        dq_state = dq_env.reset()
        dq_si = dq_env.state_to_index(dq_state)
        dq_total_reward = 0.0
        dq_trajectory = [dq_state]

        dq_action = dq_agent.select_action(dq_si)
        for dq_step in range(max_steps):
            dq_action = dq_agent.select_action(dq_si)
            dq_result = dq_env.step(dq_action)
            dq_nsi = dq_env.state_to_index(dq_result.state)
            dq_total_reward += dq_result.reward
            dq_trajectory.append(dq_result.state)
            dq_agent.update(dq_si, dq_action, dq_result.reward, dq_nsi, alpha, gamma, dq_result.done)
            dq_si = dq_nsi
            if dq_result.done:
                break
        dq_agent.decay_epsilon()

        ss_state = ss_env.reset()
        ss_si = ss_env.state_to_index(ss_state)
        ss_total_reward = 0.0
        ss_trajectory = [ss_state]

        ss_action = ss_agent.select_action(ss_si)
        for ss_step in range(max_steps):
            ss_result = ss_env.step(ss_action)
            ss_nsi = ss_env.state_to_index(ss_result.state)
            ss_total_reward += ss_result.reward
            ss_trajectory.append(ss_result.state)
            ss_next_action = ss_agent.select_action(ss_nsi)
            ss_agent.update(ss_si, ss_action, ss_result.reward, ss_nsi, alpha, gamma, ss_result.done,
                            next_action=ss_next_action)
            ss_si = ss_nsi
            ss_action = ss_next_action
            if ss_result.done:
                break
        ss_agent.decay_epsilon()

        dq_ep_metrics = EpisodeMetrics(
            episode=ep, total_reward=dq_total_reward,
            steps=dq_step + 1, success=dq_result.done, epsilon=dq_agent.epsilon,
        )
        ss_ep_metrics = EpisodeMetrics(
            episode=ep, total_reward=ss_total_reward,
            steps=ss_step + 1, success=ss_result.done, epsilon=ss_agent.epsilon,
        )

        yield ep, dq_agent, dq_env, ss_agent, ss_env, dq_ep_metrics, ss_ep_metrics, dq_trajectory, ss_trajectory
