"""Tests for training orchestrator."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import numpy as np
from mazemind.envs.maze_parser import parse_maze_file, list_maze_files
from mazemind.envs.micromouse_env import MicromouseEnv
from mazemind.agents.dyna_q import DynaQAgent
from mazemind.agents.sarsa import SarsaAgent
from mazemind.training.orchestrator import train_agent, extract_optimal_path
from mazemind.utils.metrics import TrainingMetrics, ComparisonResult

MAZE_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'mazes', 'classic')


@pytest.fixture
def maze():
    return parse_maze_file(os.path.join(MAZE_DIR, "AAMC16Maze.txt"))


class TestTrainingOrchestrator:
    def test_train_dyna_q(self, maze):
        agent = DynaQAgent(n_planning_steps=5)
        env = MicromouseEnv(maze)
        metrics = train_agent(agent, env, n_episodes=10, max_steps=100,
                              seed=42, agent_name='Dyna-Q')
        assert len(metrics.episodes) == 10
        assert metrics.agent_name == 'Dyna-Q'

    def test_train_sarsa(self, maze):
        agent = SarsaAgent()
        env = MicromouseEnv(maze)
        metrics = train_agent(agent, env, n_episodes=10, max_steps=100,
                              seed=42, agent_name='SARSA')
        assert len(metrics.episodes) == 10
        assert metrics.agent_name == 'SARSA'

    def test_metrics_have_rewards(self, maze):
        agent = DynaQAgent()
        env = MicromouseEnv(maze)
        metrics = train_agent(agent, env, n_episodes=5, max_steps=50, seed=42)
        assert len(metrics.rewards) == 5
        assert all(isinstance(r, float) for r in metrics.rewards)

    def test_metrics_have_steps(self, maze):
        agent = DynaQAgent()
        env = MicromouseEnv(maze)
        metrics = train_agent(agent, env, n_episodes=5, max_steps=50, seed=42)
        assert len(metrics.steps) == 5
        assert all(s > 0 for s in metrics.steps)

    def test_extract_optimal_path(self, maze):
        agent = DynaQAgent()
        env = MicromouseEnv(maze)
        train_agent(agent, env, n_episodes=5, max_steps=100, seed=42)
        path = extract_optimal_path(agent, MicromouseEnv(maze))
        assert len(path) > 0
        assert path[0] == maze.start

    def test_seed_reproducibility(self, maze):
        a1 = DynaQAgent()
        e1 = MicromouseEnv(maze)
        m1 = train_agent(a1, e1, n_episodes=10, max_steps=100, seed=123)

        a2 = DynaQAgent()
        e2 = MicromouseEnv(maze)
        m2 = train_agent(a2, e2, n_episodes=10, max_steps=100, seed=123)

        np.testing.assert_array_equal(m1.rewards, m2.rewards)
        np.testing.assert_array_equal(m1.steps, m2.steps)


class TestMetrics:
    def test_summary(self, maze):
        agent = DynaQAgent()
        env = MicromouseEnv(maze)
        metrics = train_agent(agent, env, n_episodes=10, max_steps=100, seed=42)
        summary = metrics.summary()
        assert 'agent' in summary
        assert 'total_episodes' in summary
        assert 'mean_reward' in summary
        assert summary['total_episodes'] == 10

    def test_comparison_result(self, maze):
        dq = DynaQAgent()
        de = MicromouseEnv(maze)
        dm = train_agent(dq, de, n_episodes=10, max_steps=100, seed=42)

        ss = SarsaAgent()
        se = MicromouseEnv(maze)
        sm = train_agent(ss, se, n_episodes=10, max_steps=100, seed=42)

        comp = ComparisonResult(dyna_q_metrics=dm, sarsa_metrics=sm)
        result = comp.statistical_test('rewards')
        assert 't_statistic' in result
        assert 'p_value' in result or 't_p_value' in result
