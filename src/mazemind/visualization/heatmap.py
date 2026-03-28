"""State-visitation heatmaps and Q-value visualizations."""

from __future__ import annotations
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from mazemind.envs.maze_parser import MazeData, ACTION_NAMES, ACTION_DELTAS


def _overlay_walls(ax: Axes, maze: MazeData, linewidth: float = 1.5):
    n = maze.size
    for r in range(n):
        for c in range(n):
            walls = maze.walls[r][c]
            x, y = c, r
            if walls["N"]:
                ax.plot([x - 0.5, x + 0.5], [y - 0.5, y - 0.5],
                        color="black", linewidth=linewidth)
            if walls["S"]:
                ax.plot([x - 0.5, x + 0.5], [y + 0.5, y + 0.5],
                        color="black", linewidth=linewidth)
            if walls["W"]:
                ax.plot([x - 0.5, x - 0.5], [y - 0.5, y + 0.5],
                        color="black", linewidth=linewidth)
            if walls["E"]:
                ax.plot([x + 0.5, x + 0.5], [y - 0.5, y + 0.5],
                        color="black", linewidth=linewidth)


def render_heatmap(
    data: np.ndarray,
    ax: Optional[Axes] = None,
    title: str = "State Visitation Heatmap",
    cmap: str = "hot",
    annotate: bool = False,
    maze: Optional[MazeData] = None,
) -> tuple[Figure, Axes]:
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    else:
        fig = ax.get_figure()

    im = ax.imshow(data, cmap=cmap, interpolation="nearest", aspect="equal")

    if annotate:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data[i, j]
                if val > 0:
                    color = "white" if val > data.max() * 0.5 else "black"
                    ax.text(j, i, f"{int(val)}", ha="center", va="center",
                            fontsize=6, color=color)

    if maze is not None:
        _overlay_walls(ax, maze)

    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])

    return fig, ax


def render_q_value_map(
    q_table: np.ndarray,
    maze: MazeData,
    ax: Optional[Axes] = None,
    title: str = "Q-Value Map",
) -> tuple[Figure, Axes]:
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    else:
        fig = ax.get_figure()

    n = maze.size
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_aspect("equal")
    ax.invert_yaxis()

    for r in range(n):
        for c in range(n):
            si = r * n + c
            q_vals = q_table[si]

            for action in range(4):
                if maze.has_wall(r, c, ACTION_NAMES[action]):
                    continue
                dr, dc = ACTION_DELTAS[action]
                nr, nc = r + dr, c + dc
                if not (0 <= nr < n and 0 <= nc < n):
                    continue

                q = q_vals[action]
                color = "green" if q > 0 else "red"
                alpha = min(abs(q) / 50.0, 1.0)
                alpha = max(alpha, 0.1)

                ax.annotate(
                    "",
                    xy=(c + dc * 0.35, r + dr * 0.35),
                    xytext=(c, r),
                    arrowprops=dict(
                        arrowstyle="->",
                        color=color,
                        alpha=alpha,
                        lw=max(abs(q) / 10.0, 0.5),
                    ),
                )

    _overlay_walls(ax, maze)

    for gr, gc in maze.goals:
        ax.add_patch(plt.Rectangle(
            (gc - 0.4, gr - 0.4), 0.8, 0.8,
            facecolor="#f39c12", alpha=0.6, edgecolor="none",
        ))
        ax.text(gc, gr, "G", ha="center", va="center",
                fontsize=8, fontweight="bold", color="white")

    sr, sc = maze.start
    ax.add_patch(plt.Rectangle(
        (sc - 0.4, sr - 0.4), 0.8, 0.8,
        facecolor="#2ecc71", alpha=0.6, edgecolor="none",
    ))
    ax.text(sc, sr, "S", ha="center", va="center",
            fontsize=8, fontweight="bold", color="white")

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])

    return fig, ax


def render_heatmap_comparison(
    data_left: np.ndarray,
    data_right: np.ndarray,
    title_left: str = "Agent 1",
    title_right: str = "Agent 2",
    cmap: str = "hot",
    maze: Optional[MazeData] = None,
) -> Figure:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    render_heatmap(data_left, ax=ax1, title=title_left, cmap=cmap, maze=maze)
    render_heatmap(data_right, ax=ax2, title=title_right, cmap=cmap, maze=maze)
    plt.tight_layout()
    return fig


def render_model_knowledge(
    maze: MazeData,
    model: dict[tuple[int, int], tuple[float, int]],
    ax: Optional[Axes] = None,
    title: str = "Dyna-Q Internal Model Knowledge",
) -> tuple[Figure, Axes]:
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    else:
        fig = ax.get_figure()

    n = maze.size
    knowledge = np.zeros((n, n))

    for (si, _action), (_reward, _nsi) in model.items():
        r, c = si // n, si % n
        knowledge[r][c] += 1

    im = ax.imshow(knowledge, cmap="Purples", interpolation="nearest", aspect="equal")
    _overlay_walls(ax, maze)

    for gr, gc in maze.goals:
        ax.add_patch(plt.Rectangle(
            (gc - 0.4, gr - 0.4), 0.8, 0.8,
            facecolor="#f39c12", alpha=0.6, edgecolor="none",
        ))
        ax.text(gc, gr, "G", ha="center", va="center",
                fontsize=8, fontweight="bold", color="white")

    sr, sc = maze.start
    ax.add_patch(plt.Rectangle(
        (sc - 0.4, sr - 0.4), 0.8, 0.8,
        facecolor="#2ecc71", alpha=0.6, edgecolor="none",
    ))
    ax.text(sc, sr, "S", ha="center", va="center",
            fontsize=8, fontweight="bold", color="white")

    fig.colorbar(im, ax=ax, shrink=0.8, label="Known actions per cell")
    ax.set_title(f"{title}\n({len(model)} state-action pairs known)", fontsize=11, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])

    return fig, ax


def render_exploration_timeline(
    snapshots: list,
    maze: MazeData,
    agent_name: str = "",
) -> Figure:
    n_snaps = len(snapshots)
    cols = min(n_snaps, 5)
    rows = (n_snaps + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if n_snaps == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    cmap = "Blues" if "Dyna" in agent_name else "Reds"

    for i, snap in enumerate(snapshots):
        ax = axes[i]
        n = maze.size
        im = ax.imshow(snap.visit_counts, cmap=cmap, interpolation="nearest", aspect="equal")
        _overlay_walls(ax, maze, linewidth=0.5)

        for gr, gc in maze.goals:
            ax.add_patch(plt.Rectangle((gc - 0.3, gr - 0.3), 0.6, 0.6,
                                        facecolor="#f39c12", alpha=0.5, edgecolor="none"))

        explored = int(np.count_nonzero(snap.visit_counts))
        extra = f" | Model: {snap.model_size}" if snap.model_size > 0 else ""
        ax.set_title(f"Ep {snap.episode}\n{explored}/256 cells{extra}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    for i in range(n_snaps, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(f"{agent_name} Exploration Timeline", fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig
