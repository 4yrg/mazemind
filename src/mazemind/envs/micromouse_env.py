"""Gym-like RL environment for Micromouse maze navigation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from mazemind.envs.maze_parser import (
    MazeData,
    ACTION_DELTAS,
    ACTION_NAMES,
)


@dataclass
class StepResult:
    state: tuple[int, int]
    reward: float
    done: bool
    info: dict


class MicromouseEnv:
    def __init__(
        self,
        maze: MazeData,
        reward_step: float = -1.0,
        reward_goal: float = 100.0,
        reward_wall: float = -1.0,
    ):
        self.maze = maze
        self.reward_step = reward_step
        self.reward_goal = reward_goal
        self.reward_wall = reward_wall
        self.state: tuple[int, int] = maze.start
        self.steps = 0
        self.visited: set[tuple[int, int]] = set()

    def reset(self) -> tuple[int, int]:
        self.state = self.maze.start
        self.steps = 0
        self.visited = {self.maze.start}
        return self.state

    def step(self, action: int) -> StepResult:
        self.steps += 1
        row, col = self.state
        direction = ACTION_NAMES[action]

        if self.maze.has_wall(row, col, direction):
            return StepResult(
                state=self.state,
                reward=self.reward_wall,
                done=False,
                info={"collision": True, "steps": self.steps},
            )

        dr, dc = ACTION_DELTAS[action]
        nr, nc = row + dr, col + dc

        if not (0 <= nr < self.maze.size and 0 <= nc < self.maze.size):
            return StepResult(
                state=self.state,
                reward=self.reward_wall,
                done=False,
                info={"collision": True, "steps": self.steps},
            )

        self.state = (nr, nc)
        self.visited.add(self.state)

        if self.maze.is_goal(nr, nc):
            return StepResult(
                state=self.state,
                reward=self.reward_goal,
                done=True,
                info={"success": True, "steps": self.steps},
            )

        return StepResult(
            state=self.state,
            reward=self.reward_step,
            done=False,
            info={"steps": self.steps},
        )

    def get_state(self) -> tuple[int, int]:
        return self.state

    def state_to_index(self, state: tuple[int, int]) -> int:
        return state[0] * self.maze.size + state[1]

    def index_to_state(self, index: int) -> tuple[int, int]:
        return (index // self.maze.size, index % self.maze.size)

    @property
    def n_states(self) -> int:
        return self.maze.size * self.maze.size

    @property
    def n_actions(self) -> int:
        return 4

    def get_visit_counts(self) -> np.ndarray:
        counts = np.zeros((self.maze.size, self.maze.size))
        for r, c in self.visited:
            counts[r][c] += 1
        return counts
