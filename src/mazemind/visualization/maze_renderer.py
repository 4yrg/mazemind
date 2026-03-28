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
WALL_COLOR = "#2c3e50"
VISITED_CMAP = "YlOrRd"


def _maze_to_ascii(maze: MazeData) -> list[str]:
    n = maze.size
    lines = []
    for row in range(n - 1, -1, -1):
        top = ""
        mid = ""
        for col in range(n):
            top += "o"
            if maze.walls[row][col]["N"]:
                top += "---"
            else:
                top += "   "
            if maze.walls[row][col]["W"]:
                mid += "| . "
            else:
                mid += "  . "
            mid += " "
        top += "o"
        mid += "|"
        lines.append(top)
        lines.append(mid)
    bottom = "o"
    for col in range(n):
        if maze.walls[0][col]["S"]:
            bottom += "---o"
        else:
            bottom += "   o"
    lines.append(bottom)
    return lines


def _render_ascii(ax: Axes, maze: MazeData, U: float = 40.0):
    lines = _maze_to_ascii(maze)
    n = maze.size
    num_lines = len(lines)

    for y, line in enumerate(lines):
        for x, ch in enumerate(line):
            if ch == "o":
                ax.plot(x * U / 4, (num_lines - 1 - y) * U / 4, "o",
                        color=WALL_COLOR, markersize=3, zorder=2)

    for y, line in enumerate(lines):
        if y % 2 == 0:
            for x in range(len(line)):
                if line[x:x + 3] == "---":
                    vx1 = x * U / 4
                    vx2 = (x + 3) * U / 4
                    vy = (num_lines - 1 - y) * U / 4
                    ax.plot([vx1, vx2], [vy, vy], color=WALL_COLOR,
                            linewidth=2.5, solid_capstyle="round", zorder=1)

    for y, line in enumerate(lines):
        if y % 2 == 1:
            for x, ch in enumerate(line):
                if ch == "|":
                    vx = x * U / 4
                    vy1 = (num_lines - 1 - y) * U / 4
                    vy2 = (num_lines - y) * U / 4
                    ax.plot([vx, vx], [vy1, vy2], color=WALL_COLOR,
                            linewidth=2.5, solid_capstyle="round", zorder=1)

    for gr, gc in maze.goals:
        text_row = 2 * (n - 1 - gr) + 1
        text_col = 4 * gc + 2
        cx = text_col * U / 4
        cy = (num_lines - 1 - text_row + 0.5) * U / 4
        rect = plt.Rectangle(
            (cx - U / 8, cy - U / 8), U / 4, U / 4,
            facecolor=GOAL_COLOR, alpha=0.7, edgecolor="none", zorder=3,
        )
        ax.add_patch(rect)
        ax.text(cx, cy, "G", ha="center", va="center",
                fontsize=max(6, int(U / 8)), fontweight="bold",
                color="white", zorder=4)

    sr, sc = maze.start
    text_row = 2 * (n - 1 - sr) + 1
    text_col = 4 * sc + 2
    cx = text_col * U / 4
    cy = (num_lines - 1 - text_row + 0.5) * U / 4
    rect = plt.Rectangle(
        (cx - U / 8, cy - U / 8), U / 4, U / 4,
        facecolor=START_COLOR, alpha=0.7, edgecolor="none", zorder=3,
    )
    ax.add_patch(rect)
    ax.text(cx, cy, "S", ha="center", va="center",
            fontsize=max(6, int(U / 8)), fontweight="bold",
            color="white", zorder=4)

    return U, num_lines


def _cell_to_xy(r, c, n, num_lines, U):
    text_row = 2 * (n - 1 - r) + 1
    text_col = 4 * c + 2
    x = text_col * U / 4
    y = (num_lines - 1 - text_row + 0.5) * U / 4
    return x, y


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
    U = 40.0
    num_lines = 2 * n + 1

    if show_walls:
        _render_ascii(ax, maze, U)

    if visit_counts is not None:
        vmax = max(visit_counts.max(), 1)
        for r in range(n):
            for c in range(n):
                if visit_counts[r][c] > 0:
                    intensity = visit_counts[r][c] / vmax
                    color = plt.cm.get_cmap(VISITED_CMAP)(intensity)
                    cx, cy = _cell_to_xy(r, c, n, num_lines, U)
                    rect = plt.Rectangle(
                        (cx - U / 8, cy - U / 8), U / 4, U / 4,
                        facecolor=color, alpha=0.5, zorder=2,
                    )
                    ax.add_patch(rect)

    if path and len(path) > 1:
        px = [_cell_to_xy(r, c, n, num_lines, U)[0] for r, c in path]
        py = [_cell_to_xy(r, c, n, num_lines, U)[1] for r, c in path]
        ax.plot(px, py, color=PATH_COLOR, linewidth=2, alpha=0.8, zorder=5)

    if agent_pos is not None:
        cx, cy = _cell_to_xy(*agent_pos, n, num_lines, U)
        circle = plt.Circle(
            (cx, cy), U / 8,
            facecolor=AGENT_COLOR, edgecolor="darkred",
            linewidth=1.5, zorder=7,
        )
        ax.add_patch(circle)

    if title:
        ax.set_title(title, fontsize=11, fontweight="bold")

    total_w = (4 * n + 1) * U / 4
    total_h = (2 * n + 1) * U / 4
    ax.set_xlim(-U / 4, total_w + U / 4)
    ax.set_ylim(-U / 4, total_h + U / 4)
    ax.set_aspect("equal")
    ax.axis("off")

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
    U = 40.0
    num_lines = 2 * n + 1

    _render_ascii(ax, maze, U)

    vmax = max(visit_counts.max(), 1)
    for r in range(n):
        for c in range(n):
            if visit_counts[r][c] > 0:
                intensity = visit_counts[r][c] / vmax
                color = plt.cm.get_cmap(VISITED_CMAP)(intensity)
                cx, cy = _cell_to_xy(r, c, n, num_lines, U)
                rect = plt.Rectangle(
                    (cx - U / 8, cy - U / 8), U / 4, U / 4,
                    facecolor=color, alpha=0.4, zorder=2,
                )
                ax.add_patch(rect)

    if len(trajectory) > 1:
        px = [_cell_to_xy(r, c, n, num_lines, U)[0] for r, c in trajectory]
        py = [_cell_to_xy(r, c, n, num_lines, U)[1] for r, c in trajectory]
        ax.plot(px, py, color=PATH_COLOR, linewidth=1.5, alpha=0.8, zorder=5)

    if trajectory:
        cx, cy = _cell_to_xy(*trajectory[-1], n, num_lines, U)
        circle = plt.Circle(
            (cx, cy), U / 8,
            facecolor=AGENT_COLOR, edgecolor="darkred",
            linewidth=1.5, zorder=7,
        )
        ax.add_patch(circle)

    status = "SUCCESS" if success else "FAIL"
    title_parts = [f"{agent_name} - Episode {episode}"]
    title_parts.append(f"[{status}] Steps: {steps} | Reward: {reward:.0f}")
    if planning_steps > 0:
        title_parts.append(f"Model: {model_size} | Planning: {planning_steps}/step")

    ax.set_title("\n".join(title_parts), fontsize=9, fontweight="bold")

    total_w = (4 * n + 1) * U / 4
    total_h = (2 * n + 1) * U / 4
    ax.set_xlim(-U / 4, total_w + U / 4)
    ax.set_ylim(-U / 4, total_h + U / 4)
    ax.set_aspect("equal")
    ax.axis("off")

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
