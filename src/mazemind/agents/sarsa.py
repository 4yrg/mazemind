"""SARSA agent: model-free on-policy temporal difference learning."""

from __future__ import annotations

import numpy as np

from mazemind.agents.base_agent import BaseAgent


class SarsaAgent(BaseAgent):
    def __init__(
        self,
        n_states: int = 256,
        n_actions: int = 4,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
    ):
        super().__init__(n_states, n_actions, epsilon, epsilon_min, epsilon_decay)

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
        next_action = kwargs.get("next_action", None)

        current_q = self.q_table[state_index, action]

        if done:
            target = reward
        else:
            next_q = self.q_table[next_state_index, next_action]
            target = reward + gamma * next_q

        self.q_table[state_index, action] += alpha * (target - current_q)
