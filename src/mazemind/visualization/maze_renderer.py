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
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_aspect("equal")
    ax.invert_yaxis()

    if visit_counts is not None:
        vmax = max(visit_counts.max(), 1)
        for r in range(n):
            for c in range(n):
                if visit_counts[r][c] > 0:
                    intensity = visit_counts[r][c] / vmax
                    color = plt.cm.get_cmap(VISITED_CMAP)(intensity)
                    ax.add_patch(plt.Rectangle(
                        (c - 0.5, r - 0.5), 1, 1,
                        facecolor=color, alpha=0.5,
                    ))

    if show_walls:
        for r in range(n):
            for c in range(n):
                walls = maze.walls[r][c]
                x, y = c, r
                if walls["N"]:
                    ax.plot([x - 0.5, x + 0.5], [y - 0.5, y - 0.5],
                            color=WALL_COLOR, linewidth=2)
                if walls["S"]:
                    ax.plot([x - 0.5, x + 0.5], [y + 0.5, y + 0.5],
                            color=WALL_COLOR, linewidth=2)
                if walls["W"]:
                    ax.plot([x - 0.5, x - 0.5], [y - 0.5, y + 0.5],
                            color=WALL_COLOR, linewidth=2)
                if walls["E"]:
                    ax.plot([x + 0.5, x + 0.5], [y - 0.5, y + 0.5],
                            color=WALL_COLOR, linewidth=2)

    for gr, gc in maze.goals:
        ax.add_patch(plt.Rectangle(
            (gc - 0.4, gr - 0.4), 0.8, 0.8,
            facecolor=GOAL_COLOR, alpha=0.6, edgecolor="none",
        ))
        ax.text(gc, gr, "G", ha="center", va="center",
                fontsize=8, fontweight="bold", color="white")

    sr, sc = maze.start
    ax.add_patch(plt.Rectangle(
        (sc - 0.4, sr - 0.4), 0.8, 0.8,
        facecolor=START_COLOR, alpha=0.6, edgecolor="none",
    ))
    ax.text(sc, sr, "S", ha="center", va="center",
            fontsize=8, fontweight="bold", color="white")

    if path and len(path) > 1:
        path_y = [p[0] for p in path]
        path_x = [p[1] for p in path]
        ax.plot(path_x, path_y, color=PATH_COLOR, linewidth=2,
                alpha=0.7, zorder=3)
        ax.plot(path_x, path_y, "o", color=PATH_COLOR,
                markersize=3, alpha=0.5, zorder=3)

    if agent_pos is not None:
        ar, ac = agent_pos
        ax.add_patch(plt.Circle(
            (ac, ar), 0.3,
            facecolor=AGENT_COLOR, edgecolor="darkred",
            linewidth=2, zorder=5,
        ))

    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")

    ax.set_xticks([])
    ax.set_yticks([])

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
