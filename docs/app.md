# Running the Streamlit UI

## Launch

```bash
source .venv/Scripts/activate
streamlit run app.py
```

Opens at `http://localhost:8501` in your browser.

## UI Layout

### Sidebar: Configuration

| Control | Description | Range |
|---------|-------------|-------|
| Maze Selection | Random or pick specific from 512 classic mazes | Dropdown |
| Learning Rate (α) | Q-value update step size | 0.01 - 0.5 |
| Discount Factor (γ) | Weight for future rewards | 0.9 - 0.999 |
| Initial Epsilon | Starting exploration rate | 0.5 - 1.0 |
| Epsilon Decay | Multiplicative decay per episode | 0.95 - 0.999 |
| Dyna-Q Planning Steps | Simulated experiences per real step | 1 - 50 |
| Training Episodes | Number of episodes to train | 50 - 1000 |
| Max Steps/Episode | Timeout before episode ends | 100 - 2000 |
| Random Seed | For reproducibility | 0 - 9999 |

Click **Run Training** to start. Both agents train simultaneously on the same maze with the same seed.

### Results: 6 Tabs

#### 1. Live Training
Watch the training process in real-time:
- **Grid**: Maze with agent position (red dot), visited cells (colored overlay), path trail
- **Q-Table**: Heatmap showing max Q-value per cell, colored from blue (negative) to red (positive)
- **Policy**: Arrows showing best action per cell (N/E/S/W), sized by Q-value magnitude
- **Metrics**: Episode number, steps, reward, epsilon, model size (Dyna-Q)

Side-by-side for Dyna-Q and SARSA. Updates after each episode.

#### 2. Discovery Timeline
Episode slider showing training snapshots. At each snapshot:
- Maze with cumulative visit heatmap overlay
- Agent's trajectory in the episode
- Episode stats (steps, reward, success/fail)
- Dyna-Q: model size, planning steps

#### 3. Exploration Coverage
- Line chart: % of maze discovered over episodes
- Dyna-Q shows both physical visits and model knowledge
- SARSA shows only physical visits
- Per-agent exploration timeline grids

#### 4. Model Knowledge (Dyna-Q Only)
- Heatmap of Dyna-Q's internal transition model
- Q-value arrow map at selected episode
- Explanation of how Dyna-Q's planning accelerates learning

#### 5. Episode Replay
- Select an episode and step through it frame by frame
- Auto-play with speed control
- Shows agent moving through maze with trail

#### 6. Technique Comparison
- Side-by-side algorithm pseudocode
- Bar charts: convergence speed, avg steps, success rate
- Smoothed learning curves comparison

### After Training
- **Side-by-Side Optimal Paths** - Both agents' best paths rendered on the maze
- **Metrics Summary** - Success rate and avg reward comparison
- **Export** - Download results as CSV

## Stopping the App

Press `Ctrl+C` in the terminal where Streamlit is running.
