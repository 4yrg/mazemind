"""Metrics plotting: learning curves, comparison charts, and diagrams."""

from __future__ import annotations

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from mazemind.utils.metrics import TrainingMetrics, ComparisonResult


def plot_learning_curve(
    metrics: TrainingMetrics,
    ax: Optional[Axes] = None,
    window: int = 50,
    show_raw: bool = True,
    color: str = "#3498db",
    label: Optional[str] = None,
) -> tuple[Figure, Axes]:
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    else:
        fig = ax.get_figure()

    episodes = np.arange(len(metrics.episodes))
    rewards = metrics.rewards
    label = label or metrics.agent_name

    if show_raw:
        ax.plot(episodes, rewards, alpha=0.15, color=color, linewidth=0.5)

    if len(rewards) >= window:
        smoothed = metrics.avg_reward(window)
        smooth_episodes = episodes[window - 1:]
        ax.plot(smooth_episodes, smoothed, color=color, linewidth=2, label=label)
        std = np.array([rewards[max(0, i - window + 1):i + 1].std()
                        for i in range(window - 1, len(rewards))])
        ax.fill_between(
            smooth_episodes,
            smoothed - std,
            smoothed + std,
            alpha=0.2, color=color,
        )

    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Cumulative Reward", fontsize=11)
    ax.set_title("Learning Curve", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    return fig, ax


def plot_success_rate(
    metrics: TrainingMetrics,
    ax: Optional[Axes] = None,
    window: int = 50,
    color: str = "#2ecc71",
    label: Optional[str] = None,
) -> tuple[Figure, Axes]:
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    else:
        fig = ax.get_figure()

    label = label or metrics.agent_name

    if len(metrics.episodes) >= window:
        sr = metrics.success_rate(window)
        episodes = np.arange(window - 1, len(metrics.episodes))
        ax.plot(episodes, sr * 100, color=color, linewidth=2, label=label)

    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Success Rate (%)", fontsize=11)
    ax.set_title("Success Rate Over Training", fontsize=12, fontweight="bold")
    ax.set_ylim(-5, 105)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    return fig, ax


def plot_steps_per_episode(
    metrics: TrainingMetrics,
    ax: Optional[Axes] = None,
    window: int = 50,
    color: str = "#e74c3c",
    label: Optional[str] = None,
) -> tuple[Figure, Axes]:
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    else:
        fig = ax.get_figure()

    label = label or metrics.agent_name

    if len(metrics.steps) >= window:
        smoothed = np.convolve(metrics.steps, np.ones(window) / window, mode="valid")
        episodes = np.arange(window - 1, len(metrics.episodes))
        ax.plot(episodes, smoothed, color=color, linewidth=2, label=label)

    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Steps", fontsize=11)
    ax.set_title("Steps Per Episode", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    return fig, ax


def plot_comparison_learning_curves(
    comparison: ComparisonResult,
    window: int = 50,
) -> Figure:
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    plot_learning_curve(
        comparison.dyna_q_metrics, ax=ax, window=window,
        color="#3498db", label="Dyna-Q",
    )
    plot_learning_curve(
        comparison.sarsa_metrics, ax=ax, window=window,
        color="#e74c3c", label="SARSA",
    )
    ax.set_title("Learning Curves: Dyna-Q vs SARSA", fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_comparison_success_rates(
    comparison: ComparisonResult,
    window: int = 50,
) -> Figure:
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    plot_success_rate(
        comparison.dyna_q_metrics, ax=ax, window=window,
        color="#3498db", label="Dyna-Q",
    )
    plot_success_rate(
        comparison.sarsa_metrics, ax=ax, window=window,
        color="#e74c3c", label="SARSA",
    )
    ax.set_title("Success Rate: Dyna-Q vs SARSA", fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_convergence_bar_chart(
    comparison: ComparisonResult,
) -> Figure:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    dq_summary = comparison.dyna_q_metrics.summary()
    ss_summary = comparison.sarsa_metrics.summary()

    agents = ["Dyna-Q", "SARSA"]
    colors = ["#3498db", "#e74c3c"]

    dq_conv = dq_summary["episodes_to_convergence"] or 0
    ss_conv = ss_summary["episodes_to_convergence"] or 0
    axes[0].bar(agents, [dq_conv, ss_conv], color=colors)
    axes[0].set_title("Episodes to Convergence", fontweight="bold")
    axes[0].set_ylabel("Episodes")

    dq_steps = dq_summary["total_steps_to_convergence"] or 0
    ss_steps = ss_summary["total_steps_to_convergence"] or 0
    axes[1].bar(agents, [dq_steps, ss_steps], color=colors)
    axes[1].set_title("Total Steps to Convergence", fontweight="bold")
    axes[1].set_ylabel("Steps")

    axes[2].bar(agents,
                [dq_summary["success_rate"] * 100, ss_summary["success_rate"] * 100],
                color=colors)
    axes[2].set_title("Overall Success Rate", fontweight="bold")
    axes[2].set_ylabel("Success Rate (%)")
    axes[2].set_ylim(0, 105)

    plt.tight_layout()
    return fig


def plot_epsilon_decay(
    comparison: ComparisonResult,
) -> Figure:
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    episodes = np.arange(len(comparison.dyna_q_metrics.episodes))
    ax.plot(episodes, comparison.dyna_q_metrics.epsilons,
            color="#3498db", linewidth=2, label="Dyna-Q")
    ax.plot(episodes, comparison.sarsa_metrics.epsilons,
            color="#e74c3c", linewidth=2, label="SARSA")
    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Epsilon (exploration rate)", fontsize=11)
    ax.set_title("Exploration Decay Over Training", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_q_value_distribution(
    q_table_dq: np.ndarray,
    q_table_sarsa: np.ndarray,
) -> Figure:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    dq_vals = q_table_dq.flatten()
    dq_vals = dq_vals[dq_vals != 0]
    ss_vals = q_table_sarsa.flatten()
    ss_vals = ss_vals[ss_vals != 0]

    if len(dq_vals) > 0:
        axes[0].hist(dq_vals, bins=50, color="#3498db", alpha=0.7, edgecolor="white")
    axes[0].set_title("Dyna-Q Q-Value Distribution", fontweight="bold")
    axes[0].set_xlabel("Q-Value")
    axes[0].set_ylabel("Frequency")

    if len(ss_vals) > 0:
        axes[1].hist(ss_vals, bins=50, color="#e74c3c", alpha=0.7, edgecolor="white")
    axes[1].set_title("SARSA Q-Value Distribution", fontweight="bold")
    axes[1].set_xlabel("Q-Value")
    axes[1].set_ylabel("Frequency")

    plt.tight_layout()
    return fig


def plot_radar_comparison(
    comparison: ComparisonResult,
) -> Figure:
    dq = comparison.dyna_q_metrics.summary()
    ss = comparison.sarsa_metrics.summary()

    categories = [
        "Success Rate",
        "Avg Reward\n(normalized)",
        "Speed\n(inv steps)",
        "Convergence\nSpeed",
        "Sample\nEfficiency",
    ]

    max_steps = max(dq["mean_steps"], ss["mean_steps"], 1)
    max_reward = max(abs(dq["mean_reward"]), abs(ss["mean_reward"]), 1)

    dq_conv = dq["episodes_to_convergence"] or 500
    ss_conv = ss["episodes_to_convergence"] or 500
    max_conv = max(dq_conv, ss_conv, 1)

    dq_values = [
        dq["success_rate"],
        (dq["mean_reward"] + 100) / (max_reward + 100),
        1 - dq["mean_steps"] / max_steps,
        1 - dq_conv / max_conv,
        1 - (dq["total_steps_to_convergence"] or 5000) / max(
            dq["total_steps_to_convergence"] or 5000,
            ss["total_steps_to_convergence"] or 5000, 1),
    ]
    ss_values = [
        ss["success_rate"],
        (ss["mean_reward"] + 100) / (max_reward + 100),
        1 - ss["mean_steps"] / max_steps,
        1 - ss_conv / max_conv,
        1 - (ss["total_steps_to_convergence"] or 5000) / max(
            dq["total_steps_to_convergence"] or 5000,
            ss["total_steps_to_convergence"] or 5000, 1),
    ]

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    dq_values += dq_values[:1]
    ss_values += ss_values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(1, 1, figsize=(8, 8), subplot_kw=dict(polar=True))

    ax.fill(angles, dq_values, alpha=0.2, color="#3498db")
    ax.plot(angles, dq_values, color="#3498db", linewidth=2, label="Dyna-Q")

    ax.fill(angles, ss_values, alpha=0.2, color="#e74c3c")
    ax.plot(angles, ss_values, color="#e74c3c", linewidth=2, label="SARSA")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_title("Performance Radar: Dyna-Q vs SARSA",
                 fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1), fontsize=11)

    return fig


def plot_step_distribution(
    comparison: ComparisonResult,
) -> Figure:
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.hist(comparison.dyna_q_metrics.steps, bins=30, alpha=0.6,
            color="#3498db", label="Dyna-Q", edgecolor="white")
    ax.hist(comparison.sarsa_metrics.steps, bins=30, alpha=0.6,
            color="#e74c3c", label="SARSA", edgecolor="white")
    ax.set_xlabel("Steps per Episode", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title("Step Count Distribution", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig
