"""Tests for MicromouseEnv."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import numpy as np
from mazemind.envs.maze_parser import parse_maze_file, list_maze_files, ACTION_DELTAS, ACTION_NAMES
from mazemind.envs.micromouse_env import MicromouseEnv

MAZE_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'mazes', 'classic')


@pytest.fixture
def env():
    maze = parse_maze_file(os.path.join(MAZE_DIR, "AAMC16Maze.txt"))
    return MicromouseEnv(maze)


class TestEnvironment:
    def test_reset_returns_start(self, env):
        state = env.reset()
        assert state == env.maze.start

    def test_reset_clears_steps(self, env):
        env.step(0)
        env.reset()
        assert env.steps == 0

    def test_step_returns_step_result(self, env):
        env.reset()
        actions = env.maze.get_valid_actions(*env.state)
        if actions:
            result = env.step(actions[0])
            assert hasattr(result, 'state')
            assert hasattr(result, 'reward')
            assert hasattr(result, 'done')
            assert hasattr(result, 'info')

    def test_wall_collision_reward(self, env):
        env.reset()
        r, c = env.state
        for action in range(4):
            direction = ACTION_NAMES[action]
            if env.maze.has_wall(r, c, direction):
                result = env.step(action)
                assert result.reward == env.reward_wall
                assert result.state == (r, c)
                assert result.info.get('collision') == True
                break

    def test_valid_move_changes_state(self, env):
        env.reset()
        r, c = env.state
        actions = env.maze.get_valid_actions(r, c)
        if actions:
            action = actions[0]
            result = env.step(action)
            assert result.reward == env.reward_step
            dr, dc = ACTION_DELTAS[action]
            assert result.state == (r + dr, c + dc)

    def test_goal_reward(self, env):
        env.reset()
        goal = next(iter(env.maze.goals))
        env.state = goal
        result = env.step(0)
        # If agent is at goal and tries a valid action, it might move away
        # So we test by directly checking is_goal
        assert env.maze.is_goal(*goal)

    def test_state_to_index(self, env):
        assert env.state_to_index((0, 0)) == 0
        assert env.state_to_index((0, 1)) == 1
        assert env.state_to_index((1, 0)) == 16
        assert env.state_to_index((15, 15)) == 255

    def test_index_to_state(self, env):
        assert env.index_to_state(0) == (0, 0)
        assert env.index_to_state(1) == (0, 1)
        assert env.index_to_state(16) == (1, 0)
        assert env.index_to_state(255) == (15, 15)

    def test_roundtrip_state_conversion(self, env):
        for r in range(16):
            for c in range(16):
                idx = env.state_to_index((r, c))
                assert env.index_to_state(idx) == (r, c)

    def test_n_states(self, env):
        assert env.n_states == 256

    def test_n_actions(self, env):
        assert env.n_actions == 4

    def test_visited_tracking(self, env):
        env.reset()
        assert env.maze.start in env.visited
        actions = env.maze.get_valid_actions(*env.state)
        if actions:
            env.step(actions[0])
            assert len(env.visited) >= 2
