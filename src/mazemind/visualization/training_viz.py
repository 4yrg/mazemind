"""Live training process visualization: Q-table heatmaps, policy grids, training panels."""

from __future__ import annotations
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection

from mazemind.envs.maze_parser import MazeData, ACTION_NAMES, ACTION_DELTAS


WALL_COLOR = "#2c3e50"
PATH_COLOR = "#3498db"
AGENT_COLOR = "#e74c3c"
START_COLOR = "#2ecc71"
GOAL_COLOR = "#f39c12"

ARROW_SYMBOLS = {0: "\u2191", 1: "\u2192", 2: "\u2193", 3: "\u2190"}


def _cell_vertex(r, c, n):
    vr_top = 2 * (n - 1 - r)
    vr_bot = vr_top + 2
    vc_left = 2 * c
    vc_right = vc_left + 2
    return vr_top, vr_bot, vc_left, vc_right


def _cell_center(r, c, n, U):
    vr_top, _, vc_left, _ = _cell_vertex(r, c, n)
    return (vc_left + 1) * U, (vr_top + 1) * U


def _draw_maze_walls(ax: Axes, maze: MazeData, U: float):
    n = maze.size
    h_segs = []
    v_segs = []
    for r in range(n):
        for c in range(n):
            vr_top, vr_bot, vc_left, vc_right = _cell_vertex(r, c, n)
            if maze.walls[r][c]["N"]:
                h_segs.append([(vc_left * U, vr_top * U), (vc_right * U, vr_top * U)])
            if maze.walls[r][c]["S"]:
                h_segs.append([(vc_left * U, vr_bot * U), (vc_right * U, vr_bot * U)])
            if maze.walls[r][c]["W"]:
                v_segs.append([(vc_left * U, vr_top * U), (vc_left * U, vr_bot * U)])
            if maze.walls[r][c]["E"]:
                v_segs.append([(vc_right * U, vr_top * U), (vc_right * U, vr_bot * U)])
    lw = max(2, U * 0.12)
    if h_segs:
        ax.add_collection(LineCollection(h_segs, colors=WALL_COLOR, linewidths=lw, capstyle="butt", zorder=3))
    if v_segs:
        ax.add_collection(LineCollection(v_segs, colors=WALL_COLOR, linewidths=lw, capstyle="butt", zorder=3))
    for vr in range(2 * n + 1):
        for vc in range(2 * n + 1):
            ax.plot(vc * U, vr * U, "s", color=WALL_COLOR, markersize=lw * 0.9, zorder=4)


def _set_ax_limits(ax: Axes, n: int, U: float):
    total_w = (2 * n + 1) * U
    total_h = (2 * n + 1) * U
    ax.set_xlim(-U * 0.5, total_w + U * 0.5)
    ax.set_ylim(-U * 0.5, total_h + U * 0.5)
    ax.set_aspect("equal")
    ax.axis("off")


def render_q_table_heatmap(
    q_table: np.ndarray,
    maze: MazeData,
    ax: Optional[Axes] = None,
    title: str = "Q-Table (max Q per cell)",
) -> tuple[Figure, Axes]:
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    else:
        fig = ax.get_figure()

    n = maze.size
    U = 30.0
    max_q = np.zeros((n, n))
    for r in range(n):
        for c in range(n):
            si = r * n + c
            max_q[r][c] = np.max(q_table[si])

    vmax = max(abs(max_q.max()), abs(max_q.min()), 1e-6)
    for r in range(n):
        for c in range(n):
            cx, cy = _cell_center(r, c, n, U)
            val = max_q[r][c]
            norm_val = (val + vmax) / (2 * vmax)
            color = plt.cm.coolwarm(norm_val)
            rect = plt.Rectangle(
                (cx - U * 0.4, cy - U * 0.4), U * 0.8, U * 0.8,
                facecolor=color, alpha=0.8, edgecolor="none", zorder=1,
            )
            ax.add_patch(rect)
            if abs(val) > 0.01:
                fontsize = max(3, int(U / 8))
                ax.text(cx, cy, f"{val:.1f}", ha="center", va="center",
                        fontsize=fontsize, fontweight="bold",
                        color="white" if abs(val) > vmax * 0.5 else "black", zorder=2)

    for gr, gc in maze.goals:
        cx, cy = _cell_center(gr, gc, n, U)
        ax.add_patch(plt.Rectangle(
            (cx - U * 0.4, cy - U * 0.4), U * 0.8, U * 0.8,
            facecolor=GOAL_COLOR, alpha=0.3, edgecolor="gold", linewidth=1, zorder=2,
        ))

    _draw_maze_walls(ax, maze, U)
    _set_ax_limits(ax, n, U)
    if title:
        ax.set_title(title, fontsize=9, fontweight="bold")
    return fig, ax


def render_policy_grid(
    q_table: np.ndarray,
    maze: MazeData,
    ax: Optional[Axes] = None,
    title: str = "Policy (best action per cell)",
) -> tuple[Figure, Axes]:
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    else:
        fig = ax.get_figure()

    n = maze.size
    U = 30.0

    for r in range(n):
        for c in range(n):
            si = r * n + c
            cx, cy = _cell_center(r, c, n, U)
            best_action = int(np.argmax(q_table[si]))
            best_q = q_table[si][best_action]
            dr, dc = ACTION_DELTAS[best_action]

            if abs(best_q) > 0.01:
                color = "#27ae60" if best_q > 0 else "#c0392b"
                alpha = min(abs(best_q) / 50.0, 1.0)
                alpha = max(alpha, 0.3)
                ax.annotate(
                    "",
                    xy=(cx + dc * U * 0.3, cy + dr * U * 0.3),
                    xytext=(cx - dc * U * 0.3, cy - dr * U * 0.3),
                    arrowprops=dict(
                        arrowstyle="->", color=color,
                        lw=max(1, U / 15), alpha=alpha,
                    ),
                    zorder=2,
                )
            else:
                ax.text(cx, cy, ".", ha="center", va="center",
                        fontsize=max(4, int(U / 6)), color="gray", zorder=2)

    for gr, gc in maze.goals:
        cx, cy = _cell_center(gr, gc, n, U)
        ax.add_patch(plt.Rectangle(
            (cx - U * 0.4, cy - U * 0.4), U * 0.8, U * 0.8,
            facecolor=GOAL_COLOR, alpha=0.3, edgecolor="gold", linewidth=1, zorder=2,
        ))

    _draw_maze_walls(ax, maze, U)
    _set_ax_limits(ax, n, U)
    if title:
        ax.set_title(title, fontsize=9, fontweight="bold")
    return fig, ax


def render_training_panel(
    maze: MazeData,
    q_table: np.ndarray,
    trajectory: list[tuple[int, int]],
    visit_counts: np.ndarray,
    agent_name: str = "",
    episode: int = 0,
    steps: int = 0,
    reward: float = 0.0,
    epsilon: float = 0.0,
    success: bool = False,
    model_size: int = 0,
) -> Figure:
    fig = plt.figure(figsize=(14, 5))

    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1], wspace=0.3)

    ax_grid = fig.add_subplot(gs[0, 0])
    n = maze.size
    U = 25.0

    vmax = max(visit_counts.max(), 1)
    for r in range(n):
        for c in range(n):
            if visit_counts[r][c] > 0:
                cx, cy = _cell_center(r, c, n, U)
                intensity = visit_counts[r][c] / vmax
                color = plt.cm.get_cmap("YlOrRd")(intensity)
                ax_grid.add_patch(plt.Rectangle(
                    (cx - U * 0.4, cy - U * 0.4), U * 0.8, U * 0.8,
                    facecolor=color, alpha=0.5, zorder=1,
                ))

    for gr, gc in maze.goals:
        cx, cy = _cell_center(gr, gc, n, U)
        ax_grid.add_patch(plt.Rectangle(
            (cx - U * 0.35, cy - U * 0.35), U * 0.7, U * 0.7,
            facecolor=GOAL_COLOR, alpha=0.6, edgecolor="none", zorder=2,
        ))
        ax_grid.text(cx, cy, "G", ha="center", va="center",
                     fontsize=4, fontweight="bold", color="white", zorder=3)

    if len(trajectory) > 1:
        px = [_cell_center(r, c, n, U)[0] for r, c in trajectory]
        py = [_cell_center(r, c, n, U)[1] for r, c in trajectory]
        ax_grid.plot(px, py, color=PATH_COLOR, linewidth=1, alpha=0.7, zorder=3)

    if trajectory:
        cx, cy = _cell_center(*trajectory[-1], n, U)
        ax_grid.add_patch(plt.Circle(
            (cx, cy), U * 0.25, facecolor=AGENT_COLOR,
            edgecolor="darkred", linewidth=1, zorder=5,
        ))

    _draw_maze_walls(ax_grid, maze, U)
    _set_ax_limits(ax_grid, n, U)
    ax_grid.set_title("Grid + Path", fontsize=8, fontweight="bold")

    ax_q = fig.add_subplot(gs[0, 1])
    max_q = np.zeros((n, n))
    for r in range(n):
        for c in range(n):
            si = r * n + c
            max_q[r][c] = np.max(q_table[si])
    vmax_q = max(abs(max_q.max()), abs(max_q.min()), 1e-6)
    for r in range(n):
        for c in range(n):
            cx, cy = _cell_center(r, c, n, U)
            val = max_q[r][c]
            norm_val = (val + vmax_q) / (2 * vmax_q)
            color = plt.cm.coolwarm(norm_val)
            ax_q.add_patch(plt.Rectangle(
                (cx - U * 0.4, cy - U * 0.4), U * 0.8, U * 0.8,
                facecolor=color, alpha=0.8, edgecolor="none", zorder=1,
            ))
    for gr, gc in maze.goals:
        cx, cy = _cell_center(gr, gc, n, U)
        ax_q.add_patch(plt.Rectangle(
            (cx - U * 0.4, cy - U * 0.4), U * 0.8, U * 0.8,
            facecolor=GOAL_COLOR, alpha=0.2, edgecolor="gold", linewidth=0.5, zorder=2,
        ))
    _draw_maze_walls(ax_q, maze, U)
    _set_ax_limits(ax_q, n, U)
    ax_q.set_title("Q-Table (max Q)", fontsize=8, fontweight="bold")

    ax_pol = fig.add_subplot(gs[0, 2])
    for r in range(n):
        for c in range(n):
            si = r * n + c
            cx, cy = _cell_center(r, c, n, U)
            best_action = int(np.argmax(q_table[si]))
            best_q = q_table[si][best_action]
            dr, dc = ACTION_DELTAS[best_action]
            if abs(best_q) > 0.01:
                color = "#27ae60" if best_q > 0 else "#c0392b"
                alpha = min(abs(best_q) / 50.0, 1.0)
                alpha = max(alpha, 0.3)
                ax_pol.annotate(
                    "",
                    xy=(cx + dc * U * 0.3, cy + dr * U * 0.3),
                    xytext=(cx - dc * dc * 0.3, cy - dr * U * 0.3),
                    arrowprops=dict(arrowstyle="->", color=color,
                                    lw=max(1, U / 15), alpha=alpha),
                    zorder=2,
                )
            else:
                ax_pol.text(cx, cy, ".", ha="center", va="center",
                            fontsize=3, color="gray", zorder=2)
    for gr, gc in maze.goals:
        cx, cy = _cell_center(gr, gc, n, U)
        ax_pol.add_patch(plt.Rectangle(
            (cx - U * 0.4, cy - U * 0.4), U * 0.8, U * 0.8,
            facecolor=GOAL_COLOR, alpha=0.2, edgecolor="gold", linewidth=0.5, zorder=2,
        ))
    _draw_maze_walls(ax_pol, maze, U)
    _set_ax_limits(ax_pol, n, U)
    ax_pol.set_title("Policy (arrows)", fontsize=8, fontweight="bold")

    status = "SUCCESS" if success else ""
    info = f"{agent_name} | Ep {episode} | Steps: {steps} | Reward: {reward:.0f} | \u03b5: {epsilon:.3f}"
    if model_size > 0:
        info += f" | Model: {model_size}"
    if status:
        info += f" | {status}"
    fig.suptitle(info, fontsize=10, fontweight="bold",
                 color="#27ae60" if success else "#333")

    return fig


def render_side_by_side_training(
    maze: MazeData,
    dq_q_table: np.ndarray,
    ss_q_table: np.ndarray,
    dq_trajectory: list[tuple[int, int]],
    ss_trajectory: list[tuple[int, int]],
    dq_visits: np.ndarray,
    ss_visits: np.ndarray,
    episode: int,
    dq_steps: int = 0,
    ss_steps: int = 0,
    dq_reward: float = 0.0,
    ss_reward: float = 0.0,
    dq_epsilon: float = 0.0,
    ss_epsilon: float = 0.0,
    dq_success: bool = False,
    ss_success: bool = False,
    dq_model_size: int = 0,
) -> Figure:
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.3)

    n = maze.size
    U = 22.0

    def draw_panel(ax, q_table, trajectory, visits, title):
        vmax = max(visits.max(), 1)
        for r in range(n):
            for c in range(n):
                if visits[r][c] > 0:
                    cx, cy = _cell_center(r, c, n, U)
                    intensity = visits[r][c] / vmax
                    color = plt.cm.get_cmap("YlOrRd")(intensity)
                    ax.add_patch(plt.Rectangle(
                        (cx - U * 0.4, cy - U * 0.4), U * 0.8, U * 0.8,
                        facecolor=color, alpha=0.4, zorder=1,
                    ))
        for gr, gc in maze.goals:
            cx, cy = _cell_center(gr, gc, n, U)
            ax.add_patch(plt.Rectangle(
                (cx - U * 0.35, cy - U * 0.35), U * 0.7, U * 0.7,
                facecolor=GOAL_COLOR, alpha=0.6, edgecolor="none", zorder=2,
            ))
            ax.text(cx, cy, "G", ha="center", va="center",
                    fontsize=3, fontweight="bold", color="white", zorder=3)
        if len(trajectory) > 1:
            px = [_cell_center(r, c, n, U)[0] for r, c in trajectory]
            py = [_cell_center(r, c, n, U)[1] for r, c in trajectory]
            ax.plot(px, py, color=PATH_COLOR, linewidth=0.8, alpha=0.7, zorder=3)
        if trajectory:
            cx, cy = _cell_center(*trajectory[-1], n, U)
            ax.add_patch(plt.Circle(
                (cx, cy), U * 0.2, facecolor=AGENT_COLOR,
                edgecolor="darkred", linewidth=0.8, zorder=5,
            ))
        _draw_maze_walls(ax, maze, U)
        _set_ax_limits(ax, n, U)
        ax.set_title(title, fontsize=8, fontweight="bold")

    def draw_qtable(ax, q_table, title):
        max_q = np.zeros((n, n))
        for r in range(n):
            for c in range(n):
                max_q[r][c] = np.max(q_table[r * n + c])
        vmax_q = max(abs(max_q.max()), abs(max_q.min()), 1e-6)
        for r in range(n):
            for c in range(n):
                cx, cy = _cell_center(r, c, n, U)
                val = max_q[r][c]
                norm_val = (val + vmax_q) / (2 * vmax_q)
                color = plt.cm.coolwarm(norm_val)
                ax.add_patch(plt.Rectangle(
                    (cx - U * 0.4, cy - U * 0.4), U * 0.8, U * 0.8,
                    facecolor=color, alpha=0.8, edgecolor="none", zorder=1,
                ))
        _draw_maze_walls(ax, maze, U)
        _set_ax_limits(ax, n, U)
        ax.set_title(title, fontsize=8, fontweight="bold")

    def draw_policy(ax, q_table, title):
        for r in range(n):
            for c in range(n):
                si = r * n + c
                cx, cy = _cell_center(r, c, n, U)
                best_action = int(np.argmax(q_table[si]))
                best_q = q_table[si][best_action]
                dr, dc = ACTION_DELTAS[best_action]
                if abs(best_q) > 0.01:
                    color = "#27ae60" if best_q > 0 else "#c0392b"
                    alpha = min(abs(best_q) / 50.0, 1.0)
                    alpha = max(alpha, 0.3)
                    ax.annotate(
                        "",
                        xy=(cx + dc * U * 0.3, cy + dr * U * 0.3),
                        xytext=(cx - dc * U * 0.3, cy - dr * U * 0.3),
                        arrowprops=dict(arrowstyle="->", color=color,
                                        lw=max(0.5, U / 20), alpha=alpha),
                        zorder=2,
                    )
        _draw_maze_walls(ax, maze, U)
        _set_ax_limits(ax, n, U)
        ax.set_title(title, fontsize=8, fontweight="bold")

    ax1 = fig.add_subplot(gs[0, 0])
    draw_panel(ax1, dq_q_table, dq_trajectory, dq_visits, "Dyna-Q: Grid")

    ax2 = fig.add_subplot(gs[0, 1])
    draw_qtable(ax2, dq_q_table, "Dyna-Q: Q-Table")

    ax3 = fig.add_subplot(gs[0, 2])
    draw_policy(ax3, dq_q_table, "Dyna-Q: Policy")

    ax4 = fig.add_subplot(gs[1, 0])
    draw_panel(ax4, ss_q_table, ss_trajectory, ss_visits, "SARSA: Grid")

    ax5 = fig.add_subplot(gs[1, 1])
    draw_qtable(ax5, ss_q_table, "SARSA: Q-Table")

    ax6 = fig.add_subplot(gs[1, 2])
    draw_policy(ax6, ss_q_table, "SARSA: Policy")

    dq_status = "OK" if dq_success else ""
    ss_status = "OK" if ss_success else ""
    fig.suptitle(
        f"Episode {episode}\n"
        f"Dyna-Q: steps={dq_steps} reward={dq_reward:.0f} eps={dq_epsilon:.3f} model={dq_model_size} {dq_status}   |   "
        f"SARSA: steps={ss_steps} reward={ss_reward:.0f} eps={ss_epsilon:.3f} {ss_status}",
        fontsize=10, fontweight="bold",
    )

    return fig


def render_playback_frame(
    maze: MazeData,
    q_table: np.ndarray,
    trajectory: list[tuple[int, int]],
    step_idx: int,
    visit_counts: np.ndarray,
    agent_name: str = "",
    episode: int = 0,
    success: bool = False,
    steps: int = 0,
    reward: float = 0.0,
    epsilon: float = 0.0,
    model_size: int = 0,
) -> Figure:
    fig = plt.figure(figsize=(15, 4.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1], wspace=0.3)

    n = maze.size
    U = 25.0

    ax_grid = fig.add_subplot(gs[0, 0])
    vmax = max(visit_counts.max(), 1)
    for r in range(n):
        for c in range(n):
            if visit_counts[r][c] > 0:
                cx, cy = _cell_center(r, c, n, U)
                intensity = visit_counts[r][c] / vmax
                color = plt.cm.get_cmap("YlOrRd")(intensity)
                ax_grid.add_patch(plt.Rectangle(
                    (cx - U * 0.4, cy - U * 0.4), U * 0.8, U * 0.8,
                    facecolor=color, alpha=0.5, zorder=1,
                ))

    for gr, gc in maze.goals:
        cx, cy = _cell_center(gr, gc, n, U)
        ax_grid.add_patch(plt.Rectangle(
            (cx - U * 0.35, cy - U * 0.35), U * 0.7, U * 0.7,
            facecolor=GOAL_COLOR, alpha=0.6, edgecolor="none", zorder=2,
        ))
        ax_grid.text(cx, cy, "G", ha="center", va="center",
                     fontsize=4, fontweight="bold", color="white", zorder=3)

    if len(trajectory) > 1 and step_idx > 0:
        path_slice = trajectory[:step_idx + 1]
        px = [_cell_center(r, c, n, U)[0] for r, c in path_slice]
        py = [_cell_center(r, c, n, U)[1] for r, c in path_slice]
        ax_grid.plot(px, py, color=PATH_COLOR, linewidth=1.5, alpha=0.8, zorder=3)

    if step_idx < len(trajectory):
        cx, cy = _cell_center(*trajectory[step_idx], n, U)
        ax_grid.add_patch(plt.Circle(
            (cx, cy), U * 0.25, facecolor=AGENT_COLOR,
            edgecolor="darkred", linewidth=1, zorder=5,
        ))

    _draw_maze_walls(ax_grid, maze, U)
    _set_ax_limits(ax_grid, n, U)
    ax_grid.set_title(f"Step {step_idx}/{len(trajectory)-1}", fontsize=9, fontweight="bold")

    ax_q = fig.add_subplot(gs[0, 1])
    max_q = np.zeros((n, n))
    for r in range(n):
        for c in range(n):
            max_q[r][c] = np.max(q_table[r * n + c])
    vmax_q = max(abs(max_q.max()), abs(max_q.min()), 1e-6)
    for r in range(n):
        for c in range(n):
            cx, cy = _cell_center(r, c, n, U)
            val = max_q[r][c]
            norm_val = (val + vmax_q) / (2 * vmax_q)
            color = plt.cm.coolwarm(norm_val)
            ax_q.add_patch(plt.Rectangle(
                (cx - U * 0.4, cy - U * 0.4), U * 0.8, U * 0.8,
                facecolor=color, alpha=0.8, edgecolor="none", zorder=1,
            ))
    for gr, gc in maze.goals:
        cx, cy = _cell_center(gr, gc, n, U)
        ax_q.add_patch(plt.Rectangle(
            (cx - U * 0.4, cy - U * 0.4), U * 0.8, U * 0.8,
            facecolor=GOAL_COLOR, alpha=0.2, edgecolor="gold", linewidth=0.5, zorder=2,
        ))
    _draw_maze_walls(ax_q, maze, U)
    _set_ax_limits(ax_q, n, U)
    ax_q.set_title("Q-Table (max Q)", fontsize=9, fontweight="bold")

    ax_pol = fig.add_subplot(gs[0, 2])
    for r in range(n):
        for c in range(n):
            si = r * n + c
            cx, cy = _cell_center(r, c, n, U)
            best_a = int(np.argmax(q_table[si]))
            best_q = q_table[si][best_a]
            dr, dc = ACTION_DELTAS[best_a]
            if abs(best_q) > 0.01:
                color = "#27ae60" if best_q > 0 else "#c0392b"
                alpha = min(abs(best_q) / 50.0, 1.0)
                alpha = max(alpha, 0.3)
                ax_pol.annotate(
                    "",
                    xy=(cx + dc * U * 0.3, cy + dr * U * 0.3),
                    xytext=(cx - dc * U * 0.3, cy - dr * U * 0.3),
                    arrowprops=dict(arrowstyle="->", color=color,
                                    lw=max(0.5, U / 20), alpha=alpha),
                    zorder=2,
                )
    _draw_maze_walls(ax_pol, maze, U)
    _set_ax_limits(ax_pol, n, U)
    ax_pol.set_title("Policy", fontsize=9, fontweight="bold")

    status = "SUCCESS" if success else ""
    info = f"{agent_name} | Episode {episode} | Step {step_idx}/{len(trajectory)-1}"
    if status:
        info += f" | {status}"
    info += f" | Steps: {steps} | Reward: {reward:.0f} | \u03b5: {epsilon:.3f}"
    if model_size > 0:
        info += f" | Model: {model_size}"
    fig.suptitle(info, fontsize=9, fontweight="bold",
                 color="#27ae60" if success else "#333")

    return fig
