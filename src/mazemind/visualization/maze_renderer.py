"""Matplotlib-based maze rendering with agent path visualization."""

from __future__ import annotations
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from mazemind.envs.maze_parser import MazeData, ACTION_DELTAS

PATH_COLOR = "#3498db"
AGENT_COLOR = "#e74c3c"
START_COLOR = "#2ecc71"
GOAL_COLOR = "#f39c12"
VISITED_CMAP = "YlOrRd"

SCALE = 8


def _build_pixel_grid(maze: MazeData) -> np.ndarray:
    n = maze.size
    s = SCALE
    rows = 2 * n + 1
    cols = 2 * n + 1
    h = rows * s
    w = cols * s
    img = np.ones((h, w, 3))
    dark = np.array([0.17, 0.24, 0.31])

    def paint(pr, pc):
        img[pr * s:(pr + 1) * s, pc * s:(pc + 1) * s] = dark

    for r in range(n):
        for c in range(n):
            tr = 2 * (n - 1 - r)
            tc = 2 * c

            paint(tr, tc)

            if maze.walls[r][c]["N"]:
                paint(tr, tc + 1)

            if c == n - 1 and maze.walls[r][c]["E"]:
                paint(tr, tc + 2)

            if maze.walls[r][c]["W"]:
                paint(tr + 1, tc)

            if r == 0 and maze.walls[r][c]["S"]:
                paint(tr + 1, tc + 1)

            if r == 0 and c == n - 1 and maze.walls[r][c]["E"]:
                paint(tr + 1, tc + 2)

    for c in range(n):
        if maze.walls[0][c]["S"]:
            paint(2 * n, 2 * c)
            paint(2 * n, 2 * c + 1)
        if c == n - 1 and maze.walls[0][c]["E"]:
            paint(2 * n, 2 * c + 2)

    return img


def _cell_to_pixel(r, c, n):
    s = SCALE
    pr = (2 * (n - 1 - r) + 1) * s + s // 2
    pc = (2 * c + 1) * s + s // 2
    return pc, pr


def _grid_dims(n):
    s = SCALE
    return (2 * n + 1) * s


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
    s = SCALE
    grid_px = _grid_dims(n)

    if show_walls:
        img = _build_pixel_grid(maze)
    else:
        img = np.ones((grid_px, grid_px, 3))

    ax.imshow(img, origin="upper", interpolation="nearest")

    if visit_counts is not None:
        vmax = max(visit_counts.max(), 1)
        for r in range(n):
            for c in range(n):
                if visit_counts[r][c] > 0:
                    intensity = visit_counts[r][c] / vmax
                    color = plt.cm.get_cmap(VISITED_CMAP)(intensity)
                    px, py = _cell_to_pixel(r, c, n)
                    rect = plt.Rectangle(
                        (px - s // 2, py - s // 2), s, s,
                        facecolor=color, alpha=0.5, zorder=2,
                    )
                    ax.add_patch(rect)

    for gr, gc in maze.goals:
        px, py = _cell_to_pixel(gr, gc, n)
        rect = plt.Rectangle(
            (px - s // 2 + 1, py - s // 2 + 1), s - 2, s - 2,
            facecolor=GOAL_COLOR, alpha=0.7, edgecolor="none", zorder=3,
        )
        ax.add_patch(rect)
        ax.text(px, py, "G", ha="center", va="center",
                fontsize=5, fontweight="bold", color="white", zorder=4)

    sr, sc = maze.start
    px, py = _cell_to_pixel(sr, sc, n)
    rect = plt.Rectangle(
        (px - s // 2 + 1, py - s // 2 + 1), s - 2, s - 2,
        facecolor=START_COLOR, alpha=0.7, edgecolor="none", zorder=3,
    )
    ax.add_patch(rect)
    ax.text(px, py, "S", ha="center", va="center",
            fontsize=5, fontweight="bold", color="white", zorder=4)

    if path and len(path) > 1:
        path_x = [_cell_to_pixel(r, c, n)[0] for r, c in path]
        path_y = [_cell_to_pixel(r, c, n)[1] for r, c in path]
        ax.plot(path_x, path_y, color=PATH_COLOR, linewidth=1.5, alpha=0.8, zorder=5)

    if agent_pos is not None:
        px, py = _cell_to_pixel(*agent_pos, n)
        circle = plt.Circle(
            (px, py), s // 3,
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
    s = SCALE
    img = _build_pixel_grid(maze)
    ax.imshow(img, origin="upper", interpolation="nearest")

    vmax = max(visit_counts.max(), 1)
    for r in range(n):
        for c in range(n):
            if visit_counts[r][c] > 0:
                intensity = visit_counts[r][c] / vmax
                color = plt.cm.get_cmap(VISITED_CMAP)(intensity)
                px, py = _cell_to_pixel(r, c, n)
                rect = plt.Rectangle(
                    (px - s // 2, py - s // 2), s, s,
                    facecolor=color, alpha=0.4, zorder=2,
                )
                ax.add_patch(rect)

    for gr, gc in maze.goals:
        px, py = _cell_to_pixel(gr, gc, n)
        rect = plt.Rectangle(
            (px - s // 2 + 1, py - s // 2 + 1), s - 2, s - 2,
            facecolor=GOAL_COLOR, alpha=0.7, edgecolor="none", zorder=3,
        )
        ax.add_patch(rect)
        ax.text(px, py, "G", ha="center", va="center",
                fontsize=5, fontweight="bold", color="white", zorder=4)

    sr, sc = maze.start
    px, py = _cell_to_pixel(sr, sc, n)
    rect = plt.Rectangle(
        (px - s // 2 + 1, py - s // 2 + 1), s - 2, s - 2,
        facecolor=START_COLOR, alpha=0.7, edgecolor="none", zorder=3,
    )
    ax.add_patch(rect)
    ax.text(px, py, "S", ha="center", va="center",
            fontsize=5, fontweight="bold", color="white", zorder=4)

    if len(trajectory) > 1:
        path_x = [_cell_to_pixel(r, c, n)[0] for r, c in trajectory]
        path_y = [_cell_to_pixel(r, c, n)[1] for r, c in trajectory]
        ax.plot(path_x, path_y, color=PATH_COLOR, linewidth=1.5, alpha=0.8, zorder=5)

    if trajectory:
        px, py = _cell_to_pixel(*trajectory[-1], n)
        circle = plt.Circle(
            (px, py), s // 3,
            facecolor=AGENT_COLOR, edgecolor="darkred",
            linewidth=1, zorder=7,
        )
        ax.add_patch(circle)

    status = "SUCCESS" if success else "FAIL"
    title_parts = [f"{agent_name} - Episode {episode}"]
    title_parts.append(f"[{status}] Steps: {steps} | Reward: {reward:.0f}")
    if planning_steps > 0:
        title_parts.append(f"Model: {model_size} | Planning: {planning_steps}/step")

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
