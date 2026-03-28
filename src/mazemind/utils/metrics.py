"""Metrics collection and statistical analysis."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats


@dataclass
class EpisodeMetrics:
    episode: int
    total_reward: float
    steps: int
    success: bool
    epsilon: float


@dataclass
class TrainingMetrics:
    agent_name: str
    maze_name: str
    episodes: list[EpisodeMetrics] = field(default_factory=list)

    def add_episode(self, ep: EpisodeMetrics):
        self.episodes.append(ep)

    @property
    def rewards(self) -> np.ndarray:
        return np.array([e.total_reward for e in self.episodes])

    @property
    def steps(self) -> np.ndarray:
        return np.array([e.steps for e in self.episodes])

    @property
    def successes(self) -> np.ndarray:
        return np.array([e.success for e in self.episodes], dtype=float)

    @property
    def epsilons(self) -> np.ndarray:
        return np.array([e.epsilon for e in self.episodes])

    def success_rate(self, window: int = 50) -> np.ndarray:
        s = self.successes
        if len(s) < window:
            return np.array([s.mean()])
        return np.convolve(s, np.ones(window) / window, mode="valid")

    def avg_reward(self, window: int = 50) -> np.ndarray:
        r = self.rewards
        if len(r) < window:
            return np.array([r.mean()])
        return np.convolve(r, np.ones(window) / window, mode="valid")

    def episodes_to_convergence(self, threshold: float = 0.95, window: int = 50) -> int | None:
        sr = self.success_rate(window)
        for i, rate in enumerate(sr):
            if rate >= threshold:
                return i + window
        return None

    def total_steps_to_convergence(self, threshold: float = 0.95, window: int = 50) -> int | None:
        conv_ep = self.episodes_to_convergence(threshold, window)
        if conv_ep is None:
            return None
        return int(self.steps[:conv_ep].sum())

    def summary(self) -> dict:
        return {
            "agent": self.agent_name,
            "maze": self.maze_name,
            "total_episodes": len(self.episodes),
            "mean_reward": float(self.rewards.mean()),
            "std_reward": float(self.rewards.std()),
            "mean_steps": float(self.steps.mean()),
            "total_successes": int(self.successes.sum()),
            "success_rate": float(self.successes.mean()),
            "episodes_to_convergence": self.episodes_to_convergence(),
            "total_steps_to_convergence": self.total_steps_to_convergence(),
        }


@dataclass
class EpisodeSnapshot:
    episode: int
    path: list[tuple[int, int]]
    visit_counts: np.ndarray
    model_size: int
    planning_steps: int
    q_table_snapshot: np.ndarray
    success: bool
    steps: int
    reward: float
    epsilon: float


@dataclass
class ComparisonResult:
    dyna_q_metrics: TrainingMetrics
    sarsa_metrics: TrainingMetrics

    def statistical_test(self, metric: str = "rewards") -> dict:
        if metric == "rewards":
            a = self.dyna_q_metrics.rewards
            b = self.sarsa_metrics.rewards
        elif metric == "steps":
            a = self.dyna_q_metrics.steps
            b = self.sarsa_metrics.steps
        else:
            raise ValueError(f"Unknown metric: {metric}")

        t_stat, p_value = stats.ttest_ind(a, b)
        u_stat, u_pvalue = stats.mannwhitneyu(a, b, alternative="two-sided")

        return {
            "metric": metric,
            "dyna_q_mean": float(a.mean()),
            "sarsa_mean": float(b.mean()),
            "dyna_q_std": float(a.std()),
            "sarsa_std": float(b.std()),
            "t_statistic": float(t_stat),
            "t_p_value": float(p_value),
            "u_statistic": float(u_stat),
            "u_p_value": float(u_pvalue),
            "significant_at_005": p_value < 0.05,
        }

    def speedup_factor(self) -> float | None:
        dq_conv = self.dyna_q_metrics.total_steps_to_convergence()
        ss_conv = self.sarsa_metrics.total_steps_to_convergence()
        if dq_conv is None or ss_conv is None or dq_conv == 0:
            return None
        return ss_conv / dq_conv
