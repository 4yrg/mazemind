"""Matplotlib-based maze rendering using vertex-edge system."""

from __future__ import annotations
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection

from mazemind.envs.maze_parser import MazeData, ACTION_DELTAS

PATH_COLOR = "#3498db"
AGENT_COLOR = "#e74c3c"
START_COLOR = "#2ecc71"
GOAL_COLOR = "#f39c12"
WALL_COLOR = "#2c3e50"
VISITED_CMAP = "YlOrRd"


def _cell_vertex(r, c, n):
    """Map maze cell (r,c) to vertex grid positions.

    Vertex grid: (2n+1) rows x (2n+1) cols.
    Vertex row 0 = top of maze (north), vertex row 2n = bottom (south).
    Maze row 0 = south, maze row n-1 = north.
    """
    vr_top = 2 * (n - 1 - r)
    vr_bot = vr_top + 2
    vc_left = 2 * c
    vc_right = vc_left + 2
    return vr_top, vr_bot, vc_left, vc_right


def _draw_walls(ax: Axes, maze: MazeData, U: float):
    n = maze.size
    h_segs = []
    v_segs = []

    for r in range(n):
        for c in range(n):
            vr_top, vr_bot, vc_left, vc_right = _cell_vertex(r, c, n)

            if maze.walls[r][c]["N"]:
                h_segs.append([(vc_left * U, vr_top * U),
                               (vc_right * U, vr_top * U)])

            if maze.walls[r][c]["S"]:
                h_segs.append([(vc_left * U, vr_bot * U),
                               (vc_right * U, vr_bot * U)])

            if maze.walls[r][c]["W"]:
                v_segs.append([(vc_left * U, vr_top * U),
                               (vc_left * U, vr_bot * U)])

            if maze.walls[r][c]["E"]:
                v_segs.append([(vc_right * U, vr_top * U),
                               (vc_right * U, vr_bot * U)])

    lw = max(2, U * 0.12)

    if h_segs:
        lc = LineCollection(h_segs, colors=WALL_COLOR, linewidths=lw,
                            capstyle="butt", zorder=1)
        ax.add_collection(lc)

    if v_segs:
        lc = LineCollection(v_segs, colors=WALL_COLOR, linewidths=lw,
                            capstyle="butt", zorder=1)
        ax.add_collection(lc)

    for vr in range(2 * n + 1):
        for vc in range(2 * n + 1):
            ax.plot(vc * U, vr * U, "s", color=WALL_COLOR,
                    markersize=lw * 0.9, zorder=2)


def _cell_center(r, c, n, U):
    vr_top, _, vc_left, _ = _cell_vertex(r, c, n)
    x = (vc_left + 1) * U
    y = (vr_top + 1) * U
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

    if show_walls:
        _draw_walls(ax, maze, U)

    if visit_counts is not None:
        vmax = max(visit_counts.max(), 1)
        for r in range(n):
            for c in range(n):
                if visit_counts[r][c] > 0:
                    intensity = visit_counts[r][c] / vmax
                    color = plt.cm.get_cmap(VISITED_CMAP)(intensity)
                    cx, cy = _cell_center(r, c, n, U)
                    rect = plt.Rectangle(
                        (cx - U * 0.4, cy - U * 0.4), U * 0.8, U * 0.8,
                        facecolor=color, alpha=0.5, zorder=3,
                    )
                    ax.add_patch(rect)

    for gr, gc in maze.goals:
        cx, cy = _cell_center(gr, gc, n, U)
        rect = plt.Rectangle(
            (cx - U * 0.35, cy - U * 0.35), U * 0.7, U * 0.7,
            facecolor=GOAL_COLOR, alpha=0.7, edgecolor="none", zorder=4,
        )
        ax.add_patch(rect)
        ax.text(cx, cy, "G", ha="center", va="center",
                fontsize=max(6, int(U / 6)), fontweight="bold",
                color="white", zorder=5)

    sr, sc = maze.start
    cx, cy = _cell_center(sr, sc, n, U)
    rect = plt.Rectangle(
        (cx - U * 0.35, cy - U * 0.35), U * 0.7, U * 0.7,
        facecolor=START_COLOR, alpha=0.7, edgecolor="none", zorder=4,
    )
    ax.add_patch(rect)
    ax.text(cx, cy, "S", ha="center", va="center",
            fontsize=max(6, int(U / 6)), fontweight="bold",
            color="white", zorder=5)

    if path and len(path) > 1:
        px = [_cell_center(r, c, n, U)[0] for r, c in path]
        py = [_cell_center(r, c, n, U)[1] for r, c in path]
        ax.plot(px, py, color=PATH_COLOR, linewidth=2, alpha=0.8, zorder=6)

    if agent_pos is not None:
        cx, cy = _cell_center(*agent_pos, n, U)
        circle = plt.Circle(
            (cx, cy), U * 0.3,
            facecolor=AGENT_COLOR, edgecolor="darkred",
            linewidth=1.5, zorder=7,
        )
        ax.add_patch(circle)

    if title:
        ax.set_title(title, fontsize=11, fontweight="bold")

    total_w = (2 * n + 1) * U
    total_h = (2 * n + 1) * U
    ax.set_xlim(-U * 0.5, total_w + U * 0.5)
    ax.set_ylim(-U * 0.5, total_h + U * 0.5)
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

    _draw_walls(ax, maze, U)

    vmax = max(visit_counts.max(), 1)
    for r in range(n):
        for c in range(n):
            if visit_counts[r][c] > 0:
                intensity = visit_counts[r][c] / vmax
                color = plt.cm.get_cmap(VISITED_CMAP)(intensity)
                cx, cy = _cell_center(r, c, n, U)
                rect = plt.Rectangle(
                    (cx - U * 0.4, cy - U * 0.4), U * 0.8, U * 0.8,
                    facecolor=color, alpha=0.4, zorder=3,
                )
                ax.add_patch(rect)

    if len(trajectory) > 1:
        px = [_cell_center(r, c, n, U)[0] for r, c in trajectory]
        py = [_cell_center(r, c, n, U)[1] for r, c in trajectory]
        ax.plot(px, py, color=PATH_COLOR, linewidth=1.5, alpha=0.8, zorder=6)

    if trajectory:
        cx, cy = _cell_center(*trajectory[-1], n, U)
        circle = plt.Circle(
            (cx, cy), U * 0.3,
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

    total_w = (2 * n + 1) * U
    total_h = (2 * n + 1) * U
    ax.set_xlim(-U * 0.5, total_w + U * 0.5)
    ax.set_ylim(-U * 0.5, total_h + U * 0.5)
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
