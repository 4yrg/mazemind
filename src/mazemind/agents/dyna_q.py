"""Dyna-Q agent: model-based RL with planning from simulated experiences."""

from __future__ import annotations

import random as _random

import numpy as np

from mazemind.agents.base_agent import BaseAgent


class DynaQAgent(BaseAgent):
    def __init__(
        self,
        n_states: int = 256,
        n_actions: int = 4,
        n_planning_steps: int = 10,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
    ):
        super().__init__(n_states, n_actions, epsilon, epsilon_min, epsilon_decay)
        self.n_planning_steps = n_planning_steps
        self.model: dict[tuple[int, int], tuple[float, int]] = {}

    def update(
        self,
        state_index: int,
        action: int,
        reward: float,
        next_state_index: int,
        alpha: float,
        gamma: float,
        done: bool,
        **kwargs,
    ) -> None:
        current_q = self.q_table[state_index, action]

        if done:
            target = reward
        else:
            max_future_q = np.max(self.q_table[next_state_index])
            target = reward + gamma * max_future_q

        self.q_table[state_index, action] += alpha * (target - current_q)

        self.model[(state_index, action)] = (reward, next_state_index)

        if not done:
            for _ in range(self.n_planning_steps):
                if not self.model:
                    break
                (sim_s, sim_a), (sim_r, sim_ns) = _random.choice(
                    list(self.model.items())
                )
                sim_current_q = self.q_table[sim_s, sim_a]
                sim_max_future_q = np.max(self.q_table[sim_ns])
                self.q_table[sim_s, sim_a] += alpha * (
                    sim_r + gamma * sim_max_future_q - sim_current_q
                )

    def reset(self):
        super().reset()
        self.model = {}
