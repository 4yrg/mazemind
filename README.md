# Mazemind: Tabular Reinforcement Learning for Micromouse Maze Pathfinding

A pedagogical system evaluating **Dyna-Q** (model-based) and **SARSA** (model-free) reinforcement learning algorithms in autonomous robot pathfinding through standardized Micromouse mazes.

## Overview

Mazemind implements two tabular RL algorithms that navigate a 16x16 Micromouse maze to find the center goal. The system provides real-time visualization of how each agent explores the environment, builds its Q-table, and converges to an optimal policy.

| Algorithm | Type | Update Rule | Key Difference |
|-----------|------|-------------|----------------|
| **Dyna-Q** | Model-based / Off-policy | `Q(s,a) += α[r + γ·max Q(s',a') - Q(s,a)]` + N planning steps from internal model | Builds a world model and simulates extra experiences |
| **SARSA** | Model-free / On-policy | `Q(s,a) += α[r + γ·Q(s',a') - Q(s,a)]` | Uses the actual next action chosen by the policy |

## Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/4yrg/mazemind.git
cd mazemind
python -m venv .venv
```

### 2. Activate Virtual Environment

```bash
# Git Bash / Linux / macOS
source .venv/Scripts/activate

# Windows CMD
.venv\Scripts\activate.bat

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

### 4. Register Jupyter Kernel

```bash
python -m ipykernel install --user --name mazemind --display-name "Python (mazemind)"
```

### 5. Run the Streamlit UI

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Configure maze and hyperparameters in the sidebar, click **Run Training**, then explore the results across 5 visualization tabs.

### 6. Run the Notebooks

```bash
jupyter notebook notebooks/
```

Select the **Python (mazemind)** kernel when opening each notebook.

## Project Structure

```
mazemind/
├── app.py                          # Streamlit UI (side-by-side comparison)
├── configs/
│   └── hyperparams.json            # Default hyperparameters
├── data/
│   └── mazes/
│       ├── classic/                # 512 classic 16x16 Micromouse mazes
│       └── training/               # Training maze files
├── docs/
│   ├── setup.md                    # Installation guide
│   ├── notebooks.md                # Notebook usage guide
│   ├── app.md                      # Streamlit UI guide
│   └── architecture.md             # Code architecture
├── notebooks/
│   ├── 01_dyna_q_training.ipynb    # Dyna-Q training with visualizations
│   ├── 02_sarsa_training.ipynb     # SARSA training with visualizations
│   └── 03_comparison.ipynb         # Head-to-head comparison
├── src/mazemind/
│   ├── envs/
│   │   ├── maze_parser.py          # Parse ASCII maze files
│   │   └── micromouse_env.py       # Gym-like RL environment
│   ├── agents/
│   │   ├── base_agent.py           # Abstract agent with ε-greedy
│   │   ├── dyna_q.py               # Dyna-Q agent (model-based)
│   │   └── sarsa.py                # SARSA agent (model-free)
│   ├── training/
│   │   └── orchestrator.py         # Training loop and snapshots
│   ├── visualization/
│   │   ├── maze_renderer.py        # Vertex-edge maze rendering
│   │   ├── heatmap.py              # State visitation and Q-value heatmaps
│   │   ├── metrics_plotter.py      # Learning curves and comparison charts
│   │   └── training_viz.py         # Live training process visualization
│   └── utils/
│       └── metrics.py              # Metrics collection and statistical tests
├── tests/                          # Unit tests (45 tests)
├── requirements.txt
├── setup.py
└── README.md
```

## Environment

- **Grid**: 16x16 Micromouse maze (256 discrete states)
- **Actions**: 4 (North, East, South, West)
- **Rewards**: Step = -1, Goal = +100, Wall collision = -1
- **Start**: Bottom-left corner (0, 0)
- **Goal**: Center 2x2 cells (7,7), (7,8), (8,7), (8,8)
- **Maze source**: [micromouseonline/mazefiles](https://github.com/micromouseonline/mazefiles) (512 classic mazes bundled)

## Streamlit UI Features

| Tab | Description |
|-----|-------------|
| **Live Training** | Watch Q-table and policy update in real-time during training for both agents |
| **Discovery Timeline** | Episode slider showing exploration progress at key snapshots |
| **Exploration Coverage** | % of maze discovered over episodes (physical visits vs model knowledge) |
| **Model Knowledge** | Dyna-Q's internal transition model visualization |
| **Episode Replay** | Step-through path replay with play/pause controls |
| **Technique Comparison** | Side-by-side algorithm pseudocode, convergence charts, radar comparison |

## Notebooks

| Notebook | Description |
|----------|-------------|
| `01_dyna_q_training.ipynb` | Train Dyna-Q on a random maze: learning curves, optimal path, heatmaps, Q-value map, training process timeline, policy animation |
| `02_sarsa_training.ipynb` | Train SARSA on a random maze: same visualizations as Dyna-Q |
| `03_comparison.ipynb` | Head-to-head comparison: overlaid learning curves, convergence bars, radar chart, statistical tests (t-test, Mann-Whitney U), multi-seed evaluation, side-by-side training process |

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Learning Rate (α) | 0.1 | Step size for Q-value updates |
| Discount Factor (γ) | 0.99 | Weight for future rewards |
| Initial Epsilon | 1.0 | Starting exploration rate |
| Epsilon Decay | 0.995 | Multiplicative decay per episode |
| Planning Steps (Dyna-Q) | 10 | Simulated experiences per real step |
| Episodes | 500 | Training episodes |
| Max Steps | 1000 | Steps per episode timeout |

## Testing

```bash
python -m pytest tests/ -v
```

45 tests covering maze parsing, environment mechanics, agent updates, and training orchestration.

## License

MIT License - see [LICENSE](LICENSE).
