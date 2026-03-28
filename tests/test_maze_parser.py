"""Tests for maze parser."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
from mazemind.envs.maze_parser import (
    parse_maze_file, list_maze_files, MazeData, ACTION_DELTAS, ACTION_NAMES,
)

MAZE_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'mazes', 'classic')


@pytest.fixture
def maze():
    maze_path = os.path.join(MAZE_DIR, "AAMC16Maze.txt")
    assert os.path.exists(maze_path), f"Maze file not found: {maze_path}"
    return parse_maze_file(maze_path)


class TestMazeParser:
    def test_parse_returns_maze_data(self, maze):
        assert isinstance(maze, MazeData)
        assert maze.size == 16

    def test_start_is_tuple(self, maze):
        assert isinstance(maze.start, tuple)
        assert len(maze.start) == 2

    def test_goals_non_empty(self, maze):
        assert len(maze.goals) > 0

    def test_walls_dimensions(self, maze):
        assert len(maze.walls) == 16
        assert all(len(row) == 16 for row in maze.walls)

    def test_walls_have_directions(self, maze):
        for r in range(16):
            for c in range(16):
                walls = maze.walls[r][c]
                assert set(walls.keys()) == {"N", "E", "S", "W"}

    def test_boundary_walls(self, maze):
        for c in range(16):
            assert maze.has_wall(0, c, "S"), f"No south wall at (0,{c})"
            assert maze.has_wall(15, c, "N"), f"No north wall at (15,{c})"
        for r in range(16):
            assert maze.has_wall(r, 0, "W"), f"No west wall at ({r},0)"
            assert maze.has_wall(r, 15, "E"), f"No east wall at ({r},15)"

    def test_valid_actions_returns_list(self, maze):
        actions = maze.get_valid_actions(0, 0)
        assert isinstance(actions, list)
        assert all(a in range(4) for a in actions)

    def test_goal_detection(self, maze):
        for g in maze.goals:
            assert maze.is_goal(*g)
        assert not maze.is_goal(0, 0)

    def test_maze_is_solvable(self, maze):
        from collections import deque
        q = deque([maze.start])
        visited = {maze.start}
        found = False
        while q:
            r, c = q.popleft()
            if (r, c) in maze.goals:
                found = True
                break
            for a in maze.get_valid_actions(r, c):
                dr, dc = ACTION_DELTAS[a]
                nr, nc = r + dr, c + dc
                if (nr, nc) not in visited:
                    visited.add((nr, nc))
                    q.append((nr, nc))
        assert found, f"Maze {maze.name} is not solvable"

    def test_action_deltas_consistency(self, maze):
        for action, (dr, dc) in ACTION_DELTAS.items():
            direction = ACTION_NAMES[action]
            r, c = 1, 1
            if not maze.has_wall(r, c, direction):
                nr, nc = r + dr, c + dc
                assert 0 <= nr < 16 and 0 <= nc < 16, \
                    f"Action {action} leads out of bounds from ({r},{c})"
