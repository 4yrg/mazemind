# Architecture Overview

## Design Principles

- **Modularity**: Each component (parser, environment, agent, training, visualization) is independent
- **Strategy Pattern**: Agents implement a common interface (`BaseAgent`) with swappable algorithms
- **Separation of Concerns**: Environment physics separate from learning logic
- **Tabular Methods Only**: No neural networks - all values stored in explicit NumPy arrays

## Module Structure

```
src/mazemind/
├── envs/                    # Environment layer
│   ├── maze_parser.py       # Parse ASCII maze files to wall data
│   └── micromouse_env.py    # Gym-like step/reset/reward interface
├── agents/                  # RL agent layer
│   ├── base_agent.py        # Abstract base: Q-table, ε-greedy, interface
│   ├── dyna_q.py            # Dyna-Q: Q-learning + internal model + planning
│   └── sarsa.py             # SARSA: on-policy TD update
├── training/                # Orchestration layer
│   └── orchestrator.py      # Episode loop, snapshots, path extraction
├── visualization/           # Rendering layer
│   ├── maze_renderer.py     # Vertex-edge maze drawing
│   ├── heatmap.py           # Q-value and visitation heatmaps
│   ├── metrics_plotter.py   # Learning curves and comparison charts
│   └── training_viz.py      # Live training process visualization
└── utils/                   # Utilities
    └── metrics.py           # Dataclasses for metrics and snapshots
```

## Data Flow

```
Maze File (.txt)
    │
    ▼
maze_parser.py ──► MazeData (walls, start, goals)
    │
    ▼
micromouse_env.py ──► MicromouseEnv (reset, step, reward)
    │
    ▼
orchestrator.py ──► train_with_snapshots()
    │                   │
    │                   ├─► DynaQAgent.update()  ── Q-table + model
    │                   ├─► SarsaAgent.update()  ── Q-table
    │                   └─► EpisodeSnapshot      ── trajectory, visits, Q-table copy
    │
    ▼
visualization/ ──► render_maze(), render_heatmap(), render_training_panel()
    │
    ▼
app.py / notebooks ──► Display
```

## Key Data Structures

### MazeData
```python
walls[row][col] = {"N": bool, "E": bool, "S": bool, "W": bool}
start = (row, col)
goals = {(row, col), ...}
size = 16
```

### Q-Table
```python
q_table[state_index, action] = float  # shape: (256, 4)
# state_index = row * 16 + col
# action: 0=N, 1=E, 2=S, 3=W
```

### EpisodeSnapshot
```python
episode: int
path: list[tuple[int, int]]      # trajectory
visit_counts: np.ndarray         # cumulative visits (16x16)
model_size: int                  # Dyna-Q only
q_table_snapshot: np.ndarray     # copy of Q-table
success: bool
steps: int
reward: float
```

## Algorithm Implementations

### Dyna-Q Update
```python
# 1. Direct RL update (off-policy Q-learning)
q_table[s, a] += alpha * (reward + gamma * max(q_table[s']) - q_table[s, a])

# 2. Store in model
model[(s, a)] = (reward, s')

# 3. Planning: N simulated updates
for _ in range(n_planning):
    (sim_s, sim_a), (sim_r, sim_s') = random_sample(model)
    q_table[sim_s, sim_a] += alpha * (sim_r + gamma * max(q_table[sim_s']) - q_table[sim_s, sim_a])
```

### SARSA Update
```python
# On-policy: uses actual next action
q_table[s, a] += alpha * (reward + gamma * q_table[s', a'] - q_table[s, a])
# where a' is chosen by ε-greedy policy for state s'
```

## Visualization Pipeline

1. **Maze rendering**: Vertex-edge system. Each cell maps to corner vertices. Walls drawn as `LineCollection` segments between vertices. Square dots at corners ensure connected walls.

2. **Heatmaps**: `imshow()` of 16x16 data arrays with wall overlay drawn as lines on top.

3. **Training viz**: `render_training_panel()` combines maze grid + Q-table heatmap + policy arrows + metrics text into a single figure. `render_side_by_side_training()` places two panels side by side.
