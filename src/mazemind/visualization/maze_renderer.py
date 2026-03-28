"""Matplotlib-based maze rendering with agent path visualization."""

from __future__ import annotations

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from mazemind.envs.maze_parser import MazeData, ACTION_DELTAS


WALL_COLOR = "#2c3e50"
PATH_COLOR = "#3498db"
AGENT_COLOR = "#e74c3c"
START_COLOR = "#2ecc71"
GOAL_COLOR = "#f39c12"
VISITED_CMAP = "YlOrRd"


def _build_wall_grid(maze: MazeData, scale: int = 10) -> np.ndarray:
    n = maze.size
    h = n * scale + (n + 1) * 2
    w = n * scale + (n + 1) * 2
    grid = np.ones((h, w, 3))

    def cell_origin(r, c):
        return ((r + 1) * 2 + r * scale, (c + 1) * 2 + c * scale)

    for r in range(n):
        for c in range(n):
            top, left = cell_origin(r, c)
            walls = maze.walls[r][c]

            grid[top:top + scale, left:left + scale] = [0.96, 0.96, 0.98]

            if walls["N"]:
                y = top - 2
                grid[y:y + 2, left - 2:left + scale + 2] = [0.17, 0.24, 0.31]

            if walls["S"]:
                y = top + scale
                grid[y:y + 2, left - 2:left + scale + 2] = [0.17, 0.24, 0.31]

            if walls["W"]:
                x = left - 2
                grid[top - 2:top + scale + 2, x:x + 2] = [0.17, 0.24, 0.31]

            if walls["E"]:
                x = left + scale
                grid[top - 2:top + scale + 2, x:x + 2] = [0.17, 0.24, 0.31]

    grid[0:2, :] = [0.17, 0.24, 0.31]
    grid[-2:, :] = [0.17, 0.24, 0.31]
    grid[:, 0:2] = [0.17, 0.24, 0.31]
    grid[:, -2:] = [0.17, 0.24, 0.31]

    return grid


def render_maze(
    maze: MazeData,
    ax: Optional[Axes] = None,
    title: str = "",
    path: Optional[list[tuple[int, int]]] = None,
    agent_pos: Optional[tuple[int, int]] = None,
    visit_counts: Optional[np.ndarray] = None,
    show_walls: bool = True,
    cell_size: float = 1.0,
) -> tuple[Figure, Axes]:
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    else:
        fig = ax.get_figure()

    n = maze.size
    scale = 10

    if show_walls:
        wall_grid = _build_wall_grid(maze, scale)
        ax.imshow(wall_grid, origin="lower", interpolation="nearest")
    else:
        blank = np.ones((n * scale + (n + 1) * 2, n * scale + (n + 1) * 2, 3))
        ax.imshow(blank, origin="lower", interpolation="nearest")

    def cell_to_pixel(r, c):
        top = (r + 1) * 2 + r * scale + scale // 2
        left = (c + 1) * 2 + c * scale + scale // 2
        return left, top

    if visit_counts is not None:
        vmax = max(visit_counts.max(), 1)
        for r in range(n):
            for c in range(n):
                if visit_counts[r][c] > 0:
                    intensity = visit_counts[r][c] / vmax
                    color = plt.cm.get_cmap(VISITED_CMAP)(intensity)
                    px, py = cell_to_pixel(r, c)
                    rect = plt.Rectangle(
                        (px - scale // 2, py - scale // 2), scale, scale,
                        facecolor=color, alpha=0.5,
                    )
                    ax.add_patch(rect)

    for gr, gc in maze.goals:
        px, py = cell_to_pixel(gr, gc)
        rect = plt.Rectangle(
            (px - scale // 2 + 1, py - scale // 2 + 1), scale - 2, scale - 2,
            facecolor=GOAL_COLOR, alpha=0.6, edgecolor="none",
        )
        ax.add_patch(rect)
        ax.text(px, py, "G", ha="center", va="center",
                fontsize=6, fontweight="bold", color="white", zorder=6)

    sr, sc = maze.start
    px, py = cell_to_pixel(sr, sc)
    rect = plt.Rectangle(
        (px - scale // 2 + 1, py - scale // 2 + 1), scale - 2, scale - 2,
        facecolor=START_COLOR, alpha=0.6, edgecolor="none",
    )
    ax.add_patch(rect)
    ax.text(px, py, "S", ha="center", va="center",
            fontsize=6, fontweight="bold", color="white", zorder=6)

    if path and len(path) > 1:
        path_x = [cell_to_pixel(r, c)[0] for r, c in path]
        path_y = [cell_to_pixel(r, c)[1] for r, c in path]
        ax.plot(path_x, path_y, color=PATH_COLOR, linewidth=1.5,
                alpha=0.8, zorder=4)
        ax.plot(path_x, path_y, "o", color=PATH_COLOR,
                markersize=2, alpha=0.5, zorder=4)

    if agent_pos is not None:
        px, py = cell_to_pixel(*agent_pos)
        circle = plt.Circle(
            (px, py), scale // 3,
            facecolor=AGENT_COLOR, edgecolor="darkred",
            linewidth=1, zorder=7,
        )
        ax.add_patch(circle)

    if title:
        ax.set_title(title, fontsize=11, fontweight="bold")

    ax.axis("off")
    ax.set_aspect("equal")

    return fig, ax


def render_maze_comparison(
    maze: MazeData,
    path_left: Optional[list[tuple[int, int]]] = None,
    path_right: Optional[list[tuple[int, int]]] = None,
    title_left: str = "Agent 1",
    title_right: str = "Agent 2",
    visit_left: Optional[np.ndarray] = None,
    visit_right: Optional[np.ndarray] = None,
) -> Figure:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    render_maze(maze, ax=ax1, title=title_left,
                path=path_left, visit_counts=visit_left)
    render_maze(maze, ax=ax2, title=title_right,
                path=path_right, visit_counts=visit_right)
    plt.tight_layout()
    return fig


def render_training_snapshot(
    maze: MazeData,
    episode: int,
    trajectory: list[tuple[int, int]],
    visit_counts: np.ndarray,
    agent_name: str = "",
    model_size: int = 0,
    planning_steps: int = 0,
    success: bool = False,
    steps: int = 0,
    reward: float = 0.0,
    ax: Optional[Axes] = None,
) -> tuple[Figure, Axes]:
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    else:
        fig = ax.get_figure()

    n = maze.size
    scale = 10
    wall_grid = _build_wall_grid(maze, scale)
    ax.imshow(wall_grid, origin="lower", interpolation="nearest")

    def cell_to_pixel(r, c):
        top = (r + 1) * 2 + r * scale + scale // 2
        left = (c + 1) * 2 + c * scale + scale // 2
        return left, top

    vmax = max(visit_counts.max(), 1)
    for r in range(n):
        for c in range(n):
            if visit_counts[r][c] > 0:
                intensity = visit_counts[r][c] / vmax
                color = plt.cm.get_cmap(VISITED_CMAP)(intensity)
                px, py = cell_to_pixel(r, c)
                rect = plt.Rectangle(
                    (px - scale // 2, py - scale // 2), scale, scale,
                    facecolor=color, alpha=0.4,
                )
                ax.add_patch(rect)

    for gr, gc in maze.goals:
        px, py = cell_to_pixel(gr, gc)
        rect = plt.Rectangle(
            (px - scale // 2 + 1, py - scale // 2 + 1), scale - 2, scale - 2,
            facecolor=GOAL_COLOR, alpha=0.6, edgecolor="none",
        )
        ax.add_patch(rect)
        ax.text(px, py, "G", ha="center", va="center",
                fontsize=6, fontweight="bold", color="white", zorder=6)

    sr, sc = maze.start
    px, py = cell_to_pixel(sr, sc)
    rect = plt.Rectangle(
        (px - scale // 2 + 1, py - scale // 2 + 1), scale - 2, scale - 2,
        facecolor=START_COLOR, alpha=0.6, edgecolor="none",
    )
    ax.add_patch(rect)
    ax.text(px, py, "S", ha="center", va="center",
            fontsize=6, fontweight="bold", color="white", zorder=6)

    if len(trajectory) > 1:
        path_x = [cell_to_pixel(r, c)[0] for r, c in trajectory]
        path_y = [cell_to_pixel(r, c)[1] for r, c in trajectory]
        ax.plot(path_x, path_y, color=PATH_COLOR, linewidth=1.5, alpha=0.8, zorder=4)
        ax.plot(path_x, path_y, "o", color=PATH_COLOR, markersize=1.5, alpha=0.4, zorder=4)

    if trajectory:
        px, py = cell_to_pixel(*trajectory[-1])
        circle = plt.Circle(
            (px, py), scale // 3,
            facecolor=AGENT_COLOR, edgecolor="darkred",
            linewidth=1, zorder=7,
        )
        ax.add_patch(circle)

    status = "SUCCESS" if success else "FAIL"
    title_parts = [f"{agent_name} - Episode {episode}"]
    title_parts.append(f"[{status}] Steps: {steps} | Reward: {reward:.0f}")
    if planning_steps > 0:
        title_parts.append(f"Model: {model_size} transitions | Planning: {planning_steps} steps/real step")

    ax.set_title("\n".join(title_parts), fontsize=9, fontweight="bold")
    ax.axis("off")
    ax.set_aspect("equal")

    return fig, ax


def render_discovery_comparison(
    maze: MazeData,
    dq_episode: int,
    dq_trajectory: list[tuple[int, int]],
    dq_visits: np.ndarray,
    dq_model_size: int,
    dq_success: bool,
    dq_steps: int,
    dq_reward: float,
    ss_episode: int,
    ss_trajectory: list[tuple[int, int]],
    ss_visits: np.ndarray,
    ss_success: bool,
    ss_steps: int,
    ss_reward: float,
) -> Figure:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    render_training_snapshot(
        maze, dq_episode, dq_trajectory, dq_visits,
        agent_name="Dyna-Q", model_size=dq_model_size, planning_steps=10,
        success=dq_success, steps=dq_steps, reward=dq_reward, ax=ax1,
    )

    render_training_snapshot(
        maze, ss_episode, ss_trajectory, ss_visits,
        agent_name="SARSA", model_size=0, planning_steps=0,
        success=ss_success, steps=ss_steps, reward=ss_reward, ax=ax2,
    )

    plt.tight_layout()
    return fig
