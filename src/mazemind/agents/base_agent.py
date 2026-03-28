"""Abstract base agent with ε-greedy action selection."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseAgent(ABC):
    def __init__(
        self,
        n_states: int = 256,
        n_actions: int = 4,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table = np.zeros((n_states, n_actions))
        self.rng = np.random.default_rng()

    def select_action(self, state_index: int) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return int(np.argmax(self.q_table[state_index]))

    def select_greedy_action(self, state_index: int) -> int:
        return int(np.argmax(self.q_table[state_index]))

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_q_values(self, state_index: int) -> np.ndarray:
        return self.q_table[state_index].copy()

    def get_max_q(self, state_index: int) -> float:
        return float(np.max(self.q_table[state_index]))

    def reset(self):
        self.q_table = np.zeros((self.n_states, self.n_actions))
        self.epsilon = 1.0

    @abstractmethod
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
        ...
