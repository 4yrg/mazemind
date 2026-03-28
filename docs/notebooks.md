# Running the Notebooks

## Prerequisites

Complete the [Setup Guide](setup.md) first. Ensure the `mazemind` Jupyter kernel is registered.

## Opening Notebooks

```bash
source .venv/Scripts/activate
jupyter notebook notebooks/
```

When prompted to select a kernel, choose **Python (mazemind)**.

## Notebook Descriptions

### 01_dyna_q_training.ipynb

Trains a **Dyna-Q** agent on a randomly selected maze.

**Cells:**
1. **Imports** - Load all required modules
2. **Load Maze** - Random maze from `data/mazes/classic/`, rendered with walls
3. **Train Agent** - 500 episodes with configurable hyperparameters. Reports: success rate, avg reward, episodes to convergence
4. **Learning Curves** - 3-panel figure: reward curve (with std band), success rate, steps per episode
5. **Optimal Path** - Extracted greedy path rendered on the maze
6. **State Visitation Heatmap** - Where the agent explored most
7. **Q-Value Map** - Arrows showing learned Q-values per cell, plus Q-value distribution histogram
8. **Training Process Timeline** - Grid + Q-table heatmap + policy arrows at episodes [1, 10, 50, 200]
9. **Episode Replay Animation** - Interactive FuncAnimation with play/pause/loop controls
10. **Q-Table Evolution Animation** - Animated heatmap showing Q-values spreading backward from goal

### 02_sarsa_training.ipynb

Trains a **SARSA** agent. Same structure as notebook 01 but for the on-policy algorithm.

### 03_comparison.ipynb

Head-to-head comparison of Dyna-Q and SARSA.

**Cells:**
1. **Train Both Agents** - Same maze, same seed
2. **Summary Statistics** - Side-by-side table (success rate, avg reward, convergence speed)
3. **Learning Curves** - Overlaid smoothed curves with std bands
4. **Success Rate Comparison** - Overlaid success rate curves
5. **Convergence Bar Charts** - Episodes to convergence, total steps, success rate
6. **Step Distribution** - Overlaid histograms
7. **Exploration Behavior** - Epsilon decay curves
8. **State Visitation Heatmaps** - Side-by-side
9. **Optimal Paths** - Side-by-side on maze
10. **Q-Value Maps** - Side-by-side with arrows
11. **Q-Value Distributions** - Side-by-side histograms
12. **Radar Chart** - 5-axis comparison (Success Rate, Avg Reward, Speed, Convergence, Sample Efficiency)
13. **Statistical Tests** - t-test and Mann-Whitney U for rewards and steps
14. **Box Plots** - Reward and step distributions
15. **Multi-Seed Evaluation** - 5 seeds with confidence intervals
16. **Side-by-Side Training Process** - Grid + Q-table + policy at same episodes for both agents

## Running All Notebooks via CLI

```bash
source .venv/Scripts/activate

# Dyna-Q training
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=600 \
  --ExecutePreprocessor.kernel_name=mazemind \
  notebooks/01_dyna_q_training.ipynb \
  --output 01_dyna_q_training.ipynb --output-dir notebooks/

# SARSA training
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=600 \
  --ExecutePreprocessor.kernel_name=mazemind \
  notebooks/02_sarsa_training.ipynb \
  --output 02_sarsa_training.ipynb --output-dir notebooks/

# Comparison (takes longer - multi-seed evaluation)
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=900 \
  --ExecutePreprocessor.kernel_name=mazemind \
  notebooks/03_comparison.ipynb \
  --output 03_comparison.ipynb --output-dir notebooks/
```

## Tips

- **Random maze selection**: Each run picks a different maze. Set `np.random.seed(42)` for reproducibility.
- **Animation controls**: Use the JS slider below FuncAnimation cells to scrub through frames.
- **Kernel selection**: If cells fail with `ModuleNotFoundError`, check that the kernel is set to **Python (mazemind)**.
