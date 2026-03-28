"""Parse ASCII text maze files from micromouseonline/mazefiles."""

from __future__ import annotations

import os
import random as _random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


MAZE_REPO_URL = "https://raw.githubusercontent.com/micromouseonline/mazefiles/master"
CLASSIC_SIZE = 16

ACTION_NAMES = ["N", "E", "S", "W"]
NORTH, EAST, SOUTH, WEST = 0, 1, 2, 3
ACTION_DELTAS = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}


@dataclass
class MazeData:
    walls: list[list[dict[str, bool]]]
    start: tuple[int, int]
    goals: set[tuple[int, int]]
    size: int
    name: str = ""

    def has_wall(self, row: int, col: int, direction: str) -> bool:
        return self.walls[row][col][direction]

    def get_valid_actions(self, row: int, col: int) -> list[int]:
        valid = []
        for action, (dr, dc) in ACTION_DELTAS.items():
            nr, nc = row + dr, col + dc
            direction = ACTION_NAMES[action]
            if self.has_wall(row, col, direction):
                continue
            if 0 <= nr < self.size and 0 <= nc < self.size:
                valid.append(action)
        return valid

    def is_goal(self, row: int, col: int) -> bool:
        return (row, col) in self.goals


def _extract_maze_rows(raw_lines: list[str]) -> list[str]:
    rows = []
    for line in raw_lines:
        stripped = line.rstrip("\n\r")
        if not stripped:
            continue
        if stripped.startswith(("o", "|")):
            rows.append(stripped)
        elif rows:
            break
    return rows


def _find_cell_walls(rows: list[str], row: int, col: int) -> dict[str, bool]:
    middle_row = len(rows) - 2 * row - 2
    middle_col = 2 + 4 * col
    north_row = middle_row - 1
    south_row = middle_row + 1
    west_col = middle_col - 2
    east_col = middle_col + 2

    north = rows[north_row][middle_col] == "-"
    east = rows[middle_row][east_col] == "|"
    south = rows[south_row][middle_col] == "-"
    west = rows[middle_row][west_col] == "|"

    return {"N": north, "E": east, "S": south, "W": west}


def _find_tagged_cells(rows: list[str], tag: str) -> set[tuple[int, int]]:
    found = set()
    for i, row in enumerate(reversed(rows[1::2])):
        for j, column in enumerate(row[2::4]):
            if column == tag:
                found.add((i, j))
    return found


def parse_maze_file(filepath: str | Path) -> MazeData:
    filepath = Path(filepath)
    raw_lines = filepath.read_text().split("\n")
    rows = _extract_maze_rows(raw_lines)

    if len(rows) != CLASSIC_SIZE * 2 + 1:
        raise ValueError(
            f"Expected {CLASSIC_SIZE * 2 + 1} rows, got {len(rows)} in {filepath.name}"
        )

    walls: list[list[dict[str, bool]]] = []
    for r in range(CLASSIC_SIZE):
        row_walls = []
        for c in range(CLASSIC_SIZE):
            row_walls.append(_find_cell_walls(rows, r, c))
        walls.append(row_walls)

    start_cells = _find_tagged_cells(rows, "S")
    goal_cells = _find_tagged_cells(rows, "G")

    if not start_cells:
        start = (0, 0)
    else:
        start = next(iter(start_cells))

    if not goal_cells:
        goal_cells = {(7, 7), (7, 8), (8, 7), (8, 8)}

    return MazeData(
        walls=walls,
        start=start,
        goals=goal_cells,
        size=CLASSIC_SIZE,
        name=filepath.stem,
    )


def list_maze_files(directory: str | Path) -> list[Path]:
    directory = Path(directory)
    return sorted(directory.glob("*.txt"))


def load_random_maze(directory: str | Path) -> MazeData:
    files = list_maze_files(directory)
    if not files:
        raise FileNotFoundError(f"No .txt maze files found in {directory}")
    return parse_maze_file(_random.choice(files))


def download_mazes(directory: str | Path, category: str = "classic",
                   names: Optional[list[str]] = None) -> list[Path]:
    import urllib.request

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    if names is None:
        names = ["13ye.txt", "50.txt", "86.txt", "88.txt", "87sin.txt"]

    downloaded = []
    for name in names:
        url = f"{MAZE_REPO_URL}/{category}/{name}"
        dest = directory / name
        if dest.exists():
            downloaded.append(dest)
            continue
        try:
            urllib.request.urlretrieve(url, str(dest))
            downloaded.append(dest)
        except Exception as e:
            print(f"Failed to download {name}: {e}")

    return downloaded
