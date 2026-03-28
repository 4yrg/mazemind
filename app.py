"""Streamlit app: side-by-side Dyna-Q vs SARSA with training discovery visualization."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from mazemind.envs.maze_parser import parse_maze_file, list_maze_files, load_random_maze
from mazemind.envs.micromouse_env import MicromouseEnv
from mazemind.agents.dyna_q import DynaQAgent
from mazemind.agents.sarsa import SarsaAgent
from mazemind.training.orchestrator import train_with_snapshots, extract_optimal_path
from mazemind.utils.metrics import EpisodeSnapshot
from mazemind.visualization.maze_renderer import (
    render_maze, render_training_snapshot, render_discovery_comparison,
)
from mazemind.visualization.heatmap import (
    render_heatmap, render_q_value_map, render_model_knowledge, render_exploration_timeline,
)
from mazemind.visualization.training_viz import (
    render_side_by_side_training, render_q_table_heatmap, render_policy_grid,
)


st.set_page_config(
    page_title="Mazemind: Tabular RL Maze Pathfinding",
    page_icon="",
    layout="wide",
)

SNAPSHOT_EPISODES = [0, 2, 5, 10, 25, 50, 100, 200, 499]


@st.cache_data
def get_maze_list():
    maze_dir = os.path.join(os.path.dirname(__file__), "data", "mazes", "classic")
    files = list_maze_files(maze_dir)
    return [f.name for f in files]


def main():
    st.title("Mazemind: Tabular RL Maze Pathfinding")
    st.markdown("**Dyna-Q (Model-Based) vs SARSA (Model-Free)** - Environment Discovery Visualization")

    maze_dir = os.path.join(os.path.dirname(__file__), "data", "mazes", "classic")
    maze_names = get_maze_list()

    with st.sidebar:
        st.header("Configuration")
        st.subheader("Maze Selection")
        maze_option = st.radio("Choose maze:", ["Random", "Select specific"], index=0)
        if maze_option == "Select specific":
            selected_maze = st.selectbox("Maze file:", maze_names)
        else:
            selected_maze = None

        st.subheader("Hyperparameters")
        alpha = st.slider("Learning Rate (alpha)", 0.01, 0.5, 0.1, 0.01)
        gamma = st.slider("Discount Factor (gamma)", 0.9, 0.999, 0.99, 0.001)
        epsilon_start = st.slider("Initial Epsilon", 0.5, 1.0, 1.0, 0.05)
        epsilon_decay = st.slider("Epsilon Decay", 0.95, 0.999, 0.995, 0.001)
        n_planning = st.slider("Dyna-Q Planning Steps", 1, 50, 10, 1)
        n_episodes = st.slider("Training Episodes", 50, 1000, 500, 50)
        max_steps = st.slider("Max Steps per Episode", 100, 2000, 1000, 100)
        seed = st.number_input("Random Seed", value=42, min_value=0, max_value=9999)

        snap_indices = [i for i, e in enumerate(SNAPSHOT_EPISODES) if e < n_episodes]
        effective_snaps = [SNAPSHOT_EPISODES[i] for i in snap_indices]
        effective_snaps[-1] = min(effective_snaps[-1], n_episodes - 1)

        run_training = st.button("Run Training", type="primary", use_container_width=True)

    if "results" not in st.session_state:
        st.session_state.results = None

    if run_training:
        if selected_maze:
            maze = parse_maze_file(os.path.join(maze_dir, selected_maze))
        else:
            maze = load_random_maze(maze_dir)

        st.session_state.maze = maze

        progress_bar = st.progress(0, text="Initializing...")
        status_text = st.empty()

        dq_agent = DynaQAgent(n_planning_steps=n_planning, epsilon=epsilon_start, epsilon_decay=epsilon_decay)
        dq_env = MicromouseEnv(maze)

        ss_agent = SarsaAgent(epsilon=epsilon_start, epsilon_decay=epsilon_decay)
        ss_env = MicromouseEnv(maze)

        status_text.text("Training Dyna-Q...")
        dq_metrics, dq_snapshots, dq_traj, dq_exploration = train_with_snapshots(
            dq_agent, dq_env, n_episodes=n_episodes, max_steps=max_steps,
            alpha=alpha, gamma=gamma, seed=seed,
            agent_name="Dyna-Q", maze_name=maze.name,
            snapshot_episodes=effective_snaps,
        )
        progress_bar.progress(0.5, text="Training SARSA...")

        status_text.text("Training SARSA...")
        ss_metrics, ss_snapshots, ss_traj, ss_exploration = train_with_snapshots(
            ss_agent, ss_env, n_episodes=n_episodes, max_steps=max_steps,
            alpha=alpha, gamma=gamma, seed=seed,
            agent_name="SARSA", maze_name=maze.name,
            snapshot_episodes=effective_snaps,
        )
        progress_bar.progress(1.0, text="Complete!")

        dq_path = extract_optimal_path(dq_agent, MicromouseEnv(maze))
        ss_path = extract_optimal_path(ss_agent, MicromouseEnv(maze))

        st.session_state.results = {
            "maze": maze, "dq_agent": dq_agent, "ss_agent": ss_agent,
            "dq_metrics": dq_metrics, "ss_metrics": ss_metrics,
            "dq_snapshots": dq_snapshots, "ss_snapshots": ss_snapshots,
            "dq_traj": dq_traj, "ss_traj": ss_traj,
            "dq_exploration": dq_exploration, "ss_exploration": ss_exploration,
            "dq_path": dq_path, "ss_path": ss_path,
            "dq_env": dq_env, "ss_env": ss_env,
            "config": {"alpha": alpha, "gamma": gamma, "n_planning": n_planning,
                       "n_episodes": n_episodes, "epsilon_start": epsilon_start},
        }
        progress_bar.empty()
        status_text.empty()

    results = st.session_state.results
    if results is None:
        st.info("Configure parameters in the sidebar and click **Run Training** to start.")
        st.markdown("---")
        st.markdown("""
        ### How This Works
        This application trains two tabular RL agents simultaneously on the same maze:
        - **Dyna-Q**: Builds an internal model of the world and uses it to plan with simulated experiences
        - **SARSA**: Learns only from physically executed actions (model-free)

        The **Environment Discovery** section below shows exactly how each agent explores and learns over time.
        """)
        return

    maze = results["maze"]
    dq_agent = results["dq_agent"]
    ss_agent = results["ss_agent"]

    st.markdown("---")
    st.header("Side-by-Side Optimal Paths")
    col1, col2 = st.columns(2)
    with col1:
        fig, _ = render_maze(maze, title="Dyna-Q Optimal Path", path=results["dq_path"])
        st.pyplot(fig)
        plt.close(fig)
        st.markdown(f"**Path length:** {len(results['dq_path'])} steps")
    with col2:
        fig, _ = render_maze(maze, title="SARSA Optimal Path", path=results["ss_path"])
        st.pyplot(fig)
        plt.close(fig)
        st.markdown(f"**Path length:** {len(results['ss_path'])} steps")

    st.markdown("---")
    st.header("Metrics Summary")
    dq_summary = results["dq_metrics"].summary()
    ss_summary = results["ss_metrics"].summary()
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Dyna-Q Success Rate", f"{dq_summary['success_rate']:.1%}")
    with col2:
        st.metric("SARSA Success Rate", f"{ss_summary['success_rate']:.1%}")
    with col3:
        st.metric("Dyna-Q Avg Reward", f"{dq_summary['mean_reward']:.1f}")
    with col4:
        st.metric("SARSA Avg Reward", f"{ss_summary['mean_reward']:.1f}")

    st.markdown("---")
    st.header("Environment Discovery Visualization")

    tab_live, tab_timeline, tab_coverage, tab_model, tab_replay, tab_technique = st.tabs([
        "Live Training", "Discovery Timeline", "Exploration Coverage", "Model Knowledge (Dyna-Q)",
        "Episode Replay", "Technique Comparison",
    ])

    dq_snaps = results["dq_snapshots"]
    ss_snaps = results["ss_snapshots"]

    with tab_live:
        st.subheader("Live Training Process")
        st.markdown("Watch how each agent's **Q-table** and **policy** evolve during training.")

        snap_episodes = [s.episode for s in dq_snaps]
        selected_ep = st.select_slider(
            "Select episode to view training state:",
            options=snap_episodes,
            value=snap_episodes[min(3, len(snap_episodes) - 1)],
            key="live_ep_slider",
        )

        dq_idx = next(i for i, s in enumerate(dq_snaps) if s.episode == selected_ep)
        ss_idx = next(i for i, s in enumerate(ss_snaps) if s.episode == selected_ep)
        dq_s = dq_snaps[dq_idx]
        ss_s = ss_snaps[ss_idx]

        fig = render_side_by_side_training(
            maze,
            dq_q_table=dq_s.q_table_snapshot,
            ss_q_table=ss_s.q_table_snapshot,
            dq_trajectory=dq_s.path,
            ss_trajectory=ss_s.path,
            dq_visits=dq_s.visit_counts,
            ss_visits=ss_s.visit_counts,
            episode=selected_ep,
            dq_steps=dq_s.steps, ss_steps=ss_s.steps,
            dq_reward=dq_s.reward, ss_reward=ss_s.reward,
            dq_epsilon=dq_s.epsilon, ss_epsilon=ss_s.epsilon,
            dq_success=dq_s.success, ss_success=ss_s.success,
            dq_model_size=dq_s.model_size,
        )
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("---")
        st.markdown("""
        **What you're seeing:**
        - **Grid**: Agent's path trail (blue) with visited cells (heatmap) and current position (red dot)
        - **Q-Table**: Color shows max Q-value per cell (blue=negative, red=positive). Goal cell highlighted in gold.
        - **Policy**: Arrows show the best action per cell. Green=positive Q, red=negative Q. Size = magnitude.
        """)

        col_dq, col_ss = st.columns(2)
        with col_dq:
            st.markdown(f"**Dyna-Q** at episode {selected_ep}:")
            st.markdown(f"- Model size: **{dq_s.model_size}** transitions")
            st.markdown(f"- Planning: **{dq_s.planning_steps}** simulated steps per real step")
            st.markdown(f"- Steps: {dq_s.steps} | Reward: {dq_s.reward:.0f}")
        with col_ss:
            st.markdown(f"**SARSA** at episode {selected_ep}:")
            st.markdown(f"- No internal model (model-free)")
            st.markdown(f"- 1 Q-update per real step")
            st.markdown(f"- Steps: {ss_s.steps} | Reward: {ss_s.reward:.0f}")

    with tab_timeline:
        st.subheader("Episode-by-Episode Discovery")
        snap_episodes = [s.episode for s in dq_snaps]
        selected_ep = st.select_slider(
            "Select episode to view:",
            options=snap_episodes,
            value=snap_episodes[min(3, len(snap_episodes) - 1)],
        )

        dq_idx = next(i for i, s in enumerate(dq_snaps) if s.episode == selected_ep)
        ss_idx = next(i for i, s in enumerate(ss_snaps) if s.episode == selected_ep)
        dq_s = dq_snaps[dq_idx]
        ss_s = ss_snaps[ss_idx]

        col1, col2 = st.columns(2)
        with col1:
            fig, _ = render_training_snapshot(
                maze, dq_s.episode, dq_s.path, dq_s.visit_counts,
                agent_name="Dyna-Q", model_size=dq_s.model_size,
                planning_steps=dq_s.planning_steps,
                success=dq_s.success, steps=dq_s.steps, reward=dq_s.reward,
            )
            st.pyplot(fig)
            plt.close(fig)
            dq_explored = int(np.count_nonzero(dq_s.visit_counts))
            st.info(f"**Dyna-Q** has explored **{dq_explored}/256** cells. "
                    f"Internal model has **{dq_s.model_size}** transitions. "
                    f"Each real step triggers **{dq_s.planning_steps}** simulated planning updates.")

        with col2:
            fig, _ = render_training_snapshot(
                maze, ss_s.episode, ss_s.path, ss_s.visit_counts,
                agent_name="SARSA",
                success=ss_s.success, steps=ss_s.steps, reward=ss_s.reward,
            )
            st.pyplot(fig)
            plt.close(fig)
            ss_explored = int(np.count_nonzero(ss_s.visit_counts))
            st.info(f"**SARSA** has explored **{ss_explored}/256** cells. "
                    f"No internal model - learns only from physically executed actions. "
                    f"One Q-update per real step.")

    with tab_coverage:
        st.subheader("Exploration Coverage Over Training")
        total_cells = maze.size * maze.size

        dq_coverage_pct = [int(np.count_nonzero(s.visit_counts)) / total_cells * 100 for s in dq_snaps]
        ss_coverage_pct = [int(np.count_nonzero(s.visit_counts)) / total_cells * 100 for s in ss_snaps]
        dq_model_pct = [s.model_size / (total_cells * 4) * 100 for s in dq_snaps]
        ep_labels = [s.episode for s in dq_snaps]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ep_labels, y=dq_coverage_pct, mode="lines+markers",
            name="Dyna-Q Physical Visits", line=dict(color="#3498db", width=2),
            marker=dict(size=8),
        ))
        fig.add_trace(go.Scatter(
            x=ep_labels, y=dq_model_pct, mode="lines+markers",
            name="Dyna-Q Model Knowledge", line=dict(color="#3498db", width=2, dash="dash"),
            marker=dict(size=8),
        ))
        fig.add_trace(go.Scatter(
            x=ep_labels, y=ss_coverage_pct, mode="lines+markers",
            name="SARSA Physical Visits", line=dict(color="#e74c3c", width=2),
            marker=dict(size=8),
        ))
        fig.update_layout(
            xaxis_title="Episode", yaxis_title="% of Maze",
            template="plotly_white", height=450,
            title="Exploration Coverage: Physical Visits vs Model Knowledge",
            yaxis_range=[-5, 105],
        )
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            fig = render_exploration_timeline(dq_snaps, maze, agent_name="Dyna-Q")
            st.pyplot(fig)
            plt.close(fig)
        with col2:
            fig = render_exploration_timeline(ss_snaps, maze, agent_name="SARSA")
            st.pyplot(fig)
            plt.close(fig)

    with tab_model:
        st.subheader("Dyna-Q Internal Model Knowledge")
        st.markdown("""
        Dyna-Q maintains an **internal transition model** - a dictionary mapping each
        `(state, action)` pair to the observed `(reward, next_state)`. After each real step,
        it runs **N planning steps** by randomly sampling from this model and applying
        Q-learning updates to simulated experiences.
        """)

        model_snap = st.select_slider(
            "Select episode to view model state:",
            options=[s.episode for s in dq_snaps],
            value=dq_snaps[min(3, len(dq_snaps) - 1)].episode,
            key="model_snap",
        )
        dq_s = next(s for s in dq_snaps if s.episode == model_snap)

        col1, col2 = st.columns(2)
        with col1:
            fig, _ = render_model_knowledge(maze, dq_agent.model if dq_s.episode == dq_snaps[-1].episode else {})
            st.pyplot(fig)
            plt.close(fig)
        with col2:
            fig, _ = render_q_value_map(dq_s.q_table_snapshot, maze,
                                         title=f"Dyna-Q Q-Values at Episode {dq_s.episode}")
            st.pyplot(fig)
            plt.close(fig)

        st.markdown(f"""
        At episode **{dq_s.episode}**:
        - Model contains **{dq_s.model_size}** known state-action transitions
        - Each real step generates **{dq_s.planning_steps}** additional Q-value updates from simulated data
        - Total effective updates: **{dq_s.model_size * dq_s.planning_steps}** simulated + real steps
        """)

    with tab_replay:
        st.subheader("Episode Path Replay")
        st.markdown("Watch how the agent navigates the maze in a single episode.")

        replay_agent = st.radio("Select agent:", ["Dyna-Q", "SARSA"], horizontal=True)
        snaps = dq_snaps if replay_agent == "Dyna-Q" else ss_snaps
        snap_labels = [f"Ep {s.episode} ({'OK' if s.success else 'FAIL'}, {s.steps} steps)" for s in snaps]
        selected_snap_idx = st.selectbox("Select episode:", range(len(snap_labels)),
                                          format_func=lambda i: snap_labels[i])
        selected_snap = snaps[selected_snap_idx]

        if len(selected_snap.path) > 1:
            step_idx = st.slider("Step", 0, len(selected_snap.path) - 1, 0)
            fig, _ = render_maze(
                maze,
                title=f"{replay_agent} - Episode {selected_snap.episode} - Step {step_idx}/{len(selected_snap.path)-1}",
                path=selected_snap.path[:step_idx + 1],
                agent_pos=selected_snap.path[step_idx],
            )
            st.pyplot(fig)
            plt.close(fig)

            if st.button("Auto-play", key="autoplay"):
                placeholder = st.empty()
                for i in range(len(selected_snap.path)):
                    fig, _ = render_maze(
                        maze,
                        title=f"{replay_agent} - Step {i}/{len(selected_snap.path)-1}",
                        path=selected_snap.path[:i + 1],
                        agent_pos=selected_snap.path[i],
                    )
                    placeholder.pyplot(fig)
                    plt.close(fig)
                    import time
                    time.sleep(0.15)
                placeholder.empty()
        else:
            st.warning("Path too short for replay.")

    with tab_technique:
        st.subheader("How Each Algorithm Discovers the Environment")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            ### Dyna-Q (Model-Based)
            ```
            1. Take action in environment
            2. Observe (reward, next_state)
            3. Update Q-value (Q-learning)
            4. Store transition in internal model
            5. Repeat N times:
               - Sample random (s,a) from model
               - Simulate experience
               - Update Q-value from simulation
            ```
            **Key insight:** Each real step generates N
            additional "hallucinated" learning steps
            from the internal model. Information about
            the goal propagates backward through the
            maze exponentially faster.
            """)
            dq_model_final = dq_snaps[-1].model_size
            st.metric("Final Model Size", f"{dq_model_final} transitions")

        with col2:
            st.markdown("""
            ### SARSA (Model-Free)
            ```
            1. Take action in environment
            2. Observe (reward, next_state)
            3. Choose next_action (epsilon-greedy)
            4. Update Q-value using actual next_action
            ```
            **Key insight:** SARSA must physically
            visit every state to learn about it. It
            accounts for its own exploration noise,
            making it more conservative but requiring
            many more physical episodes to converge.
            """)
            st.metric("Internal Model", "None (model-free)")

        st.markdown("---")
        st.subheader("Q-Value Propagation Speed")

        dq_conv = dq_summary["episodes_to_convergence"]
        ss_conv = ss_summary["episodes_to_convergence"]

        fig = make_subplots(rows=1, cols=3, subplot_titles=[
            "Episodes to Convergence", "Avg Steps/Episode", "Final Success Rate (%)"
        ])
        fig.add_trace(go.Bar(
            x=["Dyna-Q", "SARSA"],
            y=[dq_conv or n_episodes, ss_conv or n_episodes],
            marker_color=["#3498db", "#e74c3c"], showlegend=False,
        ), row=1, col=1)
        fig.add_trace(go.Bar(
            x=["Dyna-Q", "SARSA"],
            y=[dq_summary["mean_steps"], ss_summary["mean_steps"]],
            marker_color=["#3498db", "#e74c3c"], showlegend=False,
        ), row=1, col=2)
        fig.add_trace(go.Bar(
            x=["Dyna-Q", "SARSA"],
            y=[dq_summary["success_rate"] * 100, ss_summary["success_rate"] * 100],
            marker_color=["#3498db", "#e74c3c"], showlegend=False,
        ), row=1, col=3)
        fig.update_layout(template="plotly_white", height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        window = 50
        dq_rewards = results["dq_metrics"].rewards
        ss_rewards = results["ss_metrics"].rewards
        if len(dq_rewards) >= window:
            dq_smooth = np.convolve(dq_rewards, np.ones(window) / window, mode="valid")
            ss_smooth = np.convolve(ss_rewards, np.ones(window) / window, mode="valid")
            ep_x = list(range(window - 1, n_episodes))
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(range(n_episodes)), y=dq_rewards.tolist(),
                                     mode="lines", line=dict(color="rgba(52,152,219,0.1)"), showlegend=False))
            fig.add_trace(go.Scatter(x=ep_x, y=dq_smooth.tolist(),
                                     mode="lines", name="Dyna-Q", line=dict(color="#3498db", width=2)))
            fig.add_trace(go.Scatter(x=list(range(n_episodes)), y=ss_rewards.tolist(),
                                     mode="lines", line=dict(color="rgba(231,76,60,0.1)"), showlegend=False))
            fig.add_trace(go.Scatter(x=ep_x, y=ss_smooth.tolist(),
                                     mode="lines", name="SARSA", line=dict(color="#e74c3c", width=2)))
            fig.update_layout(xaxis_title="Episode", yaxis_title="Cumulative Reward",
                              template="plotly_white", height=400,
                              title="Learning Curves: Dyna-Q vs SARSA")
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.header("Export Results")
    import pandas as pd
    export_df = pd.DataFrame({
        "episode": list(range(n_episodes)),
        "dyna_q_reward": dq_rewards.tolist(),
        "sarsa_reward": ss_rewards.tolist(),
        "dyna_q_steps": results["dq_metrics"].steps.tolist(),
        "sarsa_steps": results["ss_metrics"].steps.tolist(),
    })
    csv = export_df.to_csv(index=False)
    st.download_button("Download Results CSV", csv, "mazemind_results.csv", "text/csv",
                       use_container_width=True)


if __name__ == "__main__":
    main()
