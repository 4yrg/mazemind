"""Matplotlib-based maze rendering with agent path visualization."""

from __future__ import annotations
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from mazemind.envs.maze_parser import MazeData, ACTION_DELTAS

WALL_COLOR = "#2c3e50"
PATH_COLOR = "#3498db"
AGENT_COLOR = "#e74c3c"
START_COLOR = "#2ecc71"
GOAL_COLOR = "#f39c12"
VISITED_CMAP = "YlOrRd"

WT = 2
CS = 10


def _build_maze_image(maze: MazeData) -> np.ndarray:
    n = maze.size
    grid_h = n * CS + (n + 1) * WT
    grid_w = n * CS + (n + 1) * WT
    img = np.ones((grid_h, grid_w, 3))

    def cell_top(r):
        return r * (CS + WT) + WT

    def cell_left(c):
        return c * (CS + WT) + WT

    def wall_row(r):
        return r * (CS + WT) + CS + WT

    def wall_col(c):
        return c * (CS + WT) + CS + WT

    for r in range(n):
        for c in range(n):
            ct = cell_top(r)
            cl = cell_left(c)
            img[ct:ct + CS, cl:cl + CS] = [0.96, 0.96, 0.98]

            if maze.walls[r][c]["N"]:
                wy = wall_row(r - 1) if r > 0 else 0
                img[wy:wy + WT, cl - WT:cl + CS + WT] = [0.17, 0.24, 0.31]

            if maze.walls[r][c]["S"]:
                wy = wall_row(r)
                img[wy:wy + WT, cl - WT:cl + CS + WT] = [0.17, 0.24, 0.31]

            if maze.walls[r][c]["W"]:
                wx = wall_col(c - 1) if c > 0 else 0
                img[ct - WT:ct + CS + WT, wx:wx + WT] = [0.17, 0.24, 0.31]

            if maze.walls[r][c]["E"]:
                wx = wall_col(c)
                img[ct - WT:ct + CS + WT, wx:wx + WT] = [0.17, 0.24, 0.31]

    return img


def _cell_to_pixel(r, c):
    py = r * (CS + WT) + WT + CS // 2
    px = c * (CS + WT) + WT + CS // 2
    return px, py


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

    if show_walls:
        img = _build_maze_image(maze)
    else:
        grid_h = n * CS + (n + 1) * WT
        grid_w = n * CS + (n + 1) * WT
        img = np.ones((grid_h, grid_w, 3))

    ax.imshow(img, origin="upper", interpolation="nearest")

    if visit_counts is not None:
        vmax = max(visit_counts.max(), 1)
        for r in range(n):
            for c in range(n):
                if visit_counts[r][c] > 0:
                    intensity = visit_counts[r][c] / vmax
                    color = plt.cm.get_cmap(VISITED_CMAP)(intensity)
                    px, py = _cell_to_pixel(r, c)
                    rect = plt.Rectangle(
                        (px - CS // 2, py - CS // 2), CS, CS,
                        facecolor=color, alpha=0.5, zorder=2,
                    )
                    ax.add_patch(rect)

    for gr, gc in maze.goals:
        px, py = _cell_to_pixel(gr, gc)
        rect = plt.Rectangle(
            (px - CS // 2 + 1, py - CS // 2 + 1), CS - 2, CS - 2,
            facecolor=GOAL_COLOR, alpha=0.7, edgecolor="none", zorder=3,
        )
        ax.add_patch(rect)
        ax.text(px, py, "G", ha="center", va="center",
                fontsize=6, fontweight="bold", color="white", zorder=4)

    sr, sc = maze.start
    px, py = _cell_to_pixel(sr, sc)
    rect = plt.Rectangle(
        (px - CS // 2 + 1, py - CS // 2 + 1), CS - 2, CS - 2,
        facecolor=START_COLOR, alpha=0.7, edgecolor="none", zorder=3,
    )
    ax.add_patch(rect)
    ax.text(px, py, "S", ha="center", va="center",
            fontsize=6, fontweight="bold", color="white", zorder=4)

    if path and len(path) > 1:
        path_x = [_cell_to_pixel(r, c)[0] for r, c in path]
        path_y = [_cell_to_pixel(r, c)[1] for r, c in path]
        ax.plot(path_x, path_y, color=PATH_COLOR, linewidth=2,
                alpha=0.8, zorder=5)
        ax.plot(path_x, path_y, "o", color=PATH_COLOR,
                markersize=2, alpha=0.5, zorder=5)

    if agent_pos is not None:
        px, py = _cell_to_pixel(*agent_pos)
        circle = plt.Circle(
            (px, py), CS // 3,
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
    img = _build_maze_image(maze)
    ax.imshow(img, origin="upper", interpolation="nearest")

    vmax = max(visit_counts.max(), 1)
    for r in range(n):
        for c in range(n):
            if visit_counts[r][c] > 0:
                intensity = visit_counts[r][c] / vmax
                color = plt.cm.get_cmap(VISITED_CMAP)(intensity)
                px, py = _cell_to_pixel(r, c)
                rect = plt.Rectangle(
                    (px - CS // 2, py - CS // 2), CS, CS,
                    facecolor=color, alpha=0.4, zorder=2,
                )
                ax.add_patch(rect)

    for gr, gc in maze.goals:
        px, py = _cell_to_pixel(gr, gc)
        rect = plt.Rectangle(
            (px - CS // 2 + 1, py - CS // 2 + 1), CS - 2, CS - 2,
            facecolor=GOAL_COLOR, alpha=0.7, edgecolor="none", zorder=3,
        )
        ax.add_patch(rect)
        ax.text(px, py, "G", ha="center", va="center",
                fontsize=6, fontweight="bold", color="white", zorder=4)

    sr, sc = maze.start
    px, py = _cell_to_pixel(sr, sc)
    rect = plt.Rectangle(
        (px - CS // 2 + 1, py - CS // 2 + 1), CS - 2, CS - 2,
        facecolor=START_COLOR, alpha=0.7, edgecolor="none", zorder=3,
    )
    ax.add_patch(rect)
    ax.text(px, py, "S", ha="center", va="center",
            fontsize=6, fontweight="bold", color="white", zorder=4)

    if len(trajectory) > 1:
        path_x = [_cell_to_pixel(r, c)[0] for r, c in trajectory]
        path_y = [_cell_to_pixel(r, c)[1] for r, c in trajectory]
        ax.plot(path_x, path_y, color=PATH_COLOR, linewidth=1.5, alpha=0.8, zorder=5)
        ax.plot(path_x, path_y, "o", color=PATH_COLOR, markersize=1.5, alpha=0.4, zorder=5)

    if trajectory:
        px, py = _cell_to_pixel(*trajectory[-1])
        circle = plt.Circle(
            (px, py), CS // 3,
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
