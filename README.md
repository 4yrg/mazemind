# Mazemind: Tabular Reinforcement Learning for Micromouse Maze Pathfinding

A pedagogical system evaluating **Dyna-Q** (model-based) and **SARSA** (model-free) reinforcement learning algorithms in autonomous robot pathfinding through standardized Micromouse mazes.

## Quick Start

```bash
pip install -r requirements.txt
```

### Run the Streamlit UI

```bash
streamlit run app.py
```

### Run Notebooks

```bash
jupyter notebook notebooks/
```

## Project Structure

```
src/mazemind/
  envs/          - Maze parser and RL environment
  agents/        - Dyna-Q and SARSA agent implementations
  training/      - Training orchestrator
  visualization/ - Maze rendering and metrics plotting
  utils/         - Metrics collection
configs/         - Hyperparameter configuration
notebooks/       - Training and comparison notebooks
app.py           - Streamlit side-by-side comparison UI
```

## Algorithms

| Algorithm | Type | Key Feature |
|-----------|------|-------------|
| **Dyna-Q** | Model-based / Off-policy | Plans with simulated experiences from internal model |
| **SARSA** | Model-free / On-policy | Learns only from actual executed actions |

## Environment

- 16x16 Micromouse grid (256 discrete states)
- 4 actions: North, East, South, West
- Rewards: Step=-1, Goal=+100, Wall=-1
- Maze files from [micromouseonline/mazefiles](https://github.com/micromouseonline/mazefiles)
