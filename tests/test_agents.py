"""Tests for agents."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import numpy as np
from mazemind.agents.dyna_q import DynaQAgent
from mazemind.agents.sarsa import SarsaAgent


class TestDynaQAgent:
    @pytest.fixture
    def agent(self):
        return DynaQAgent(n_planning_steps=5)

    def test_initialization(self, agent):
        assert agent.q_table.shape == (256, 4)
        assert agent.n_planning_steps == 5
        assert len(agent.model) == 0

    def test_select_action_returns_int(self, agent):
        action = agent.select_action(0)
        assert isinstance(action, int)
        assert 0 <= action < 4

    def test_greedy_action(self, agent):
        agent.q_table[0] = [1, 2, 3, 0]
        action = agent.select_greedy_action(0)
        assert action == 2

    def test_epsilon_decay(self, agent):
        initial_eps = agent.epsilon
        agent.decay_epsilon()
        assert agent.epsilon < initial_eps
        assert agent.epsilon >= agent.epsilon_min

    def test_update_changes_q_table(self, agent):
        initial_q = agent.q_table[0, 0].copy()
        agent.update(0, 0, 10.0, 1, 0.1, 0.99, False)
        assert agent.q_table[0, 0] != initial_q

    def test_update_adds_to_model(self, agent):
        agent.update(0, 0, -1.0, 1, 0.1, 0.99, False)
        assert (0, 0) in agent.model
        assert agent.model[(0, 0)] == (-1.0, 1)

    def test_planning_updates_q_table(self, agent):
        agent.update(0, 0, -1.0, 1, 0.1, 0.99, False)
        agent.update(1, 1, -1.0, 2, 0.1, 0.99, False)
        initial_q = agent.q_table.copy()
        agent.update(2, 2, -1.0, 3, 0.1, 0.99, False)
        # Planning should have updated some Q-values
        assert len(agent.model) >= 2

    def test_done_terminal_update(self, agent):
        agent.update(0, 0, 100.0, 1, 0.1, 0.99, True)
        assert agent.q_table[0, 0] == pytest.approx(10.0)

    def test_reset_clears_model(self, agent):
        agent.update(0, 0, -1.0, 1, 0.1, 0.99, False)
        agent.reset()
        assert len(agent.model) == 0
        assert np.all(agent.q_table == 0)


class TestSarsaAgent:
    @pytest.fixture
    def agent(self):
        return SarsaAgent()

    def test_initialization(self, agent):
        assert agent.q_table.shape == (256, 4)

    def test_select_action_returns_int(self, agent):
        action = agent.select_action(0)
        assert isinstance(action, int)
        assert 0 <= action < 4

    def test_update_with_next_action(self, agent):
        initial_q = agent.q_table[0, 0].copy()
        agent.update(0, 0, -1.0, 1, 0.1, 0.99, False, next_action=1)
        assert agent.q_table[0, 0] != initial_q

    def test_update_without_next_action_done(self, agent):
        initial_q = agent.q_table[0, 0].copy()
        agent.update(0, 0, 100.0, 1, 0.1, 0.99, True)
        assert agent.q_table[0, 0] != initial_q

    def test_done_terminal_update(self, agent):
        agent.update(0, 0, 100.0, 1, 0.1, 0.99, True)
        assert agent.q_table[0, 0] == pytest.approx(10.0)

    def test_sarsa_uses_next_action_value(self, agent):
        agent.q_table[1, 0] = 5.0
        agent.q_table[1, 1] = 10.0
        agent.update(0, 0, 0.0, 1, 0.1, 0.99, False, next_action=1)
        expected = 0.0 + 0.1 * (0.0 + 0.99 * 10.0 - 0.0)
        assert agent.q_table[0, 0] == pytest.approx(expected)
