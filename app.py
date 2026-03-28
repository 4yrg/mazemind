"""Streamlit app: side-by-side Dyna-Q vs SARSA agent comparison."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from mazemind.envs.maze_parser import parse_maze_file, list_maze_files, load_random_maze, download_mazes
from mazemind.envs.micromouse_env import MicromouseEnv
from mazemind.agents.dyna_q import DynaQAgent
from mazemind.agents.sarsa import SarsaAgent
from mazemind.training.orchestrator import train_agent, extract_optimal_path
from mazemind.utils.metrics import ComparisonResult
from mazemind.visualization.maze_renderer import render_maze, render_maze_comparison
from mazemind.visualization.heatmap import render_heatmap, render_heatmap_comparison
from mazemind.visualization.metrics_plotter import (
    plot_comparison_learning_curves,
    plot_comparison_success_rates,
    plot_convergence_bar_chart,
    plot_epsilon_decay,
    plot_q_value_distribution,
    plot_radar_comparison,
    plot_step_distribution,
)


st.set_page_config(
    page_title="Mazemind: Tabular RL Maze Pathfinding",
    page_icon="",
    layout="wide",
)


@st.cache_data
def get_maze_files(maze_dir):
    files = list_maze_files(maze_dir)
    return [f.name for f in files]


@st.cache_data
def ensure_mazes(maze_dir):
    os.makedirs(maze_dir, exist_ok=True)
    files = list_maze_files(maze_dir)
    if len(files) < 3:
        download_mazes(maze_dir)
        files = list_maze_files(maze_dir)
    return [f.name for f in files]


def main():
    st.title("Mazemind: Tabular RL Maze Pathfinding")
    st.markdown("**Dyna-Q (Model-Based) vs SARSA (Model-Free)** side-by-side comparison")

    maze_dir = os.path.join(os.path.dirname(__file__), "data", "mazes", "classic")
    maze_names = ensure_mazes(maze_dir)

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

        run_training = st.button("Run Training", type="primary", use_container_width=True)

    if "results" not in st.session_state:
        st.session_state.results = None

    if run_training:
        if selected_maze:
            maze_path = os.path.join(maze_dir, selected_maze)
            maze = parse_maze_file(maze_path)
        else:
            maze = load_random_maze(maze_dir)

        st.session_state.maze = maze

        progress_bar = st.progress(0, text="Initializing...")
        status_text = st.empty()

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Dyna-Q Agent")
            dq_placeholder = st.empty()
            dq_metrics_placeholder = st.empty()

        with col2:
            st.markdown("### SARSA Agent")
            ss_placeholder = st.empty()
            ss_metrics_placeholder = st.empty()

        dq_agent = DynaQAgent(
            n_planning_steps=n_planning,
            epsilon=epsilon_start,
            epsilon_decay=epsilon_decay,
        )
        dq_env = MicromouseEnv(maze)
        dq_rewards = []
        dq_steps_list = []

        ss_agent = SarsaAgent(
            epsilon=epsilon_start,
            epsilon_decay=epsilon_decay,
        )
        ss_env = MicromouseEnv(maze)
        ss_rewards = []
        ss_steps_list = []

        for ep in range(n_episodes):
            progress = (ep + 1) / n_episodes
            progress_bar.progress(progress, text=f"Episode {ep + 1}/{n_episodes}")
            status_text.text(f"Training both agents... Episode {ep + 1}")

            si = dq_env.state_to_index(dq_env.reset())
            total_r = 0.0
            for step in range(max_steps):
                action = dq_agent.select_action(si)
                result = dq_env.step(action)
                nsi = dq_env.state_to_index(result.state)
                dq_agent.update(si, action, result.reward, nsi, alpha, gamma, result.done)
                si = nsi
                total_r += result.reward
                if result.done:
                    break
            dq_agent.decay_epsilon()
            dq_rewards.append(total_r)
            dq_steps_list.append(step + 1)

            si = ss_env.state_to_index(ss_env.reset())
            action = ss_agent.select_action(si)
            total_r = 0.0
            for step in range(max_steps):
                result = ss_env.step(action)
                nsi = ss_env.state_to_index(result.state)
                next_action = ss_agent.select_action(nsi)
                ss_agent.update(si, action, result.reward, nsi, alpha, gamma, result.done,
                                next_action=next_action)
                si = nsi
                action = next_action
                total_r += result.reward
                if result.done:
                    break
            ss_agent.decay_epsilon()
            ss_rewards.append(total_r)
            ss_steps_list.append(step + 1)

            if (ep + 1) % 25 == 0 or ep == n_episodes - 1:
                window = min(50, ep + 1)
                dq_sr = np.mean([1.0 if dq_rewards[i] > 0 else 0.0
                                 for i in range(max(0, ep - window + 1), ep + 1)])
                ss_sr = np.mean([1.0 if ss_rewards[i] > 0 else 0.0
                                 for i in range(max(0, ep - window + 1), ep + 1)])

                with dq_metrics_placeholder.container():
                    st.metric("Success Rate (last 50)", f"{dq_sr:.1%}")
                    st.metric("Avg Steps", f"{np.mean(dq_steps_list[-window:]):.0f}")
                    st.metric("Epsilon", f"{dq_agent.epsilon:.3f}")

                with ss_metrics_placeholder.container():
                    st.metric("Success Rate (last 50)", f"{ss_sr:.1%}")
                    st.metric("Avg Steps", f"{np.mean(ss_steps_list[-window:]):.0f}")
                    st.metric("Epsilon", f"{ss_agent.epsilon:.3f}")

        progress_bar.progress(1.0, text="Training complete!")
        status_text.text("Extracting optimal paths...")

        dq_path = extract_optimal_path(dq_agent, MicromouseEnv(maze))
        ss_path = extract_optimal_path(ss_agent, MicromouseEnv(maze))

        st.session_state.results = {
            "maze": maze,
            "dq_agent": dq_agent,
            "ss_agent": ss_agent,
            "dq_rewards": dq_rewards,
            "ss_rewards": ss_rewards,
            "dq_steps": dq_steps_list,
            "ss_steps": ss_steps_list,
            "dq_path": dq_path,
            "ss_path": ss_path,
            "dq_env": dq_env,
            "ss_env": ss_env,
            "config": {
                "alpha": alpha, "gamma": gamma,
                "epsilon_start": epsilon_start, "epsilon_decay": epsilon_decay,
                "n_planning": n_planning, "n_episodes": n_episodes,
            },
        }

        progress_bar.empty()
        status_text.empty()

    results = st.session_state.results
    if results is None:
        st.info("Configure parameters in the sidebar and click **Run Training** to start.")
        st.markdown("---")
        st.markdown("""
        ### About
        This application compares two tabular reinforcement learning algorithms:
        - **Dyna-Q**: Model-based agent that plans using simulated experiences
        - **SARSA**: Model-free agent that learns only from actual actions

        Both agents navigate 16x16 Micromouse mazes to reach the goal center.
        """)
        return

    maze = results["maze"]
    dq_agent = results["dq_agent"]
    ss_agent = results["ss_agent"]

    st.markdown("---")
    st.header("Side-by-Side Path Discovery")

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

    col1, col2, col3, col4 = st.columns(4)

    dq_success = sum(1 for r in results["dq_rewards"] if r > 0)
    ss_success = sum(1 for r in results["ss_rewards"] if r > 0)
    n_ep = len(results["dq_rewards"])

    with col1:
        st.metric("Dyna-Q Success Rate", f"{dq_success / n_ep:.1%}",
                  delta=f"{dq_success}/{n_ep} episodes")
    with col2:
        st.metric("SARSA Success Rate", f"{ss_success / n_ep:.1%}",
                  delta=f"{ss_success}/{n_ep} episodes")
    with col3:
        st.metric("Dyna-Q Avg Reward", f"{np.mean(results['dq_rewards']):.1f}")
    with col4:
        st.metric("SARSA Avg Reward", f"{np.mean(results['ss_rewards']):.1f}")

    st.markdown("---")
    st.header("Learning Curves")

    tab1, tab2, tab3 = st.tabs(["Reward Curves", "Success Rate", "Steps per Episode"])

    with tab1:
        fig = go.Figure()
        window = 50
        dq_smooth = np.convolve(results["dq_rewards"], np.ones(window) / window, mode="valid")
        ss_smooth = np.convolve(results["ss_rewards"], np.ones(window) / window, mode="valid")
        episodes = list(range(window - 1, n_ep))

        fig.add_trace(go.Scatter(
            x=list(range(n_ep)), y=results["dq_rewards"],
            mode="lines", line=dict(color="rgba(52,152,219,0.1)"), showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=episodes, y=dq_smooth.tolist(),
            mode="lines", name="Dyna-Q", line=dict(color="#3498db", width=2),
        ))
        fig.add_trace(go.Scatter(
            x=list(range(n_ep)), y=results["ss_rewards"],
            mode="lines", line=dict(color="rgba(231,76,60,0.1)"), showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=episodes, y=ss_smooth.tolist(),
            mode="lines", name="SARSA", line=dict(color="#e74c3c", width=2),
        ))
        fig.update_layout(
            xaxis_title="Episode", yaxis_title="Cumulative Reward",
            template="plotly_white", height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        dq_success_rate = np.convolve(
            [1.0 if r > 0 else 0.0 for r in results["dq_rewards"]],
            np.ones(window) / window, mode="valid",
        )
        ss_success_rate = np.convolve(
            [1.0 if r > 0 else 0.0 for r in results["ss_rewards"]],
            np.ones(window) / window, mode="valid",
        )
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=episodes, y=(dq_success_rate * 100).tolist(),
            mode="lines", name="Dyna-Q", line=dict(color="#3498db", width=2),
        ))
        fig.add_trace(go.Scatter(
            x=episodes, y=(ss_success_rate * 100).tolist(),
            mode="lines", name="SARSA", line=dict(color="#e74c3c", width=2),
        ))
        fig.update_layout(
            xaxis_title="Episode", yaxis_title="Success Rate (%)",
            template="plotly_white", height=400, yaxis_range=[-5, 105],
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        dq_steps_smooth = np.convolve(results["dq_steps"], np.ones(window) / window, mode="valid")
        ss_steps_smooth = np.convolve(results["ss_steps"], np.ones(window) / window, mode="valid")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=episodes, y=dq_steps_smooth.tolist(),
            mode="lines", name="Dyna-Q", line=dict(color="#3498db", width=2),
        ))
        fig.add_trace(go.Scatter(
            x=episodes, y=ss_steps_smooth.tolist(),
            mode="lines", name="SARSA", line=dict(color="#e74c3c", width=2),
        ))
        fig.update_layout(
            xaxis_title="Episode", yaxis_title="Steps",
            template="plotly_white", height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.header("State Visitation Heatmaps")

    col1, col2 = st.columns(2)
    with col1:
        fig, _ = render_heatmap(
            results["dq_env"].get_visit_counts(),
            title="Dyna-Q State Visitation", cmap="Blues", maze=maze,
        )
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        fig, _ = render_heatmap(
            results["ss_env"].get_visit_counts(),
            title="SARSA State Visitation", cmap="Reds", maze=maze,
        )
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("---")
    st.header("Q-Value Maps")

    col1, col2 = st.columns(2)
    with col1:
        from mazemind.visualization.heatmap import render_q_value_map
        fig, _ = render_q_value_map(dq_agent.q_table, maze, title="Dyna-Q Q-Values")
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        fig, _ = render_q_value_map(ss_agent.q_table, maze, title="SARSA Q-Values")
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("---")
    st.header("Comparative Analysis")

    tab_radar, tab_bar, tab_dist, tab_qdist, tab_epsilon = st.tabs([
        "Performance Radar", "Convergence Bars", "Step Distribution",
        "Q-Value Distribution", "Epsilon Decay",
    ])

    dq_steps_arr = np.array(results["dq_steps"])
    ss_steps_arr = np.array(results["ss_steps"])
    dq_rewards_arr = np.array(results["dq_rewards"])
    ss_rewards_arr = np.array(results["ss_rewards"])

    with tab_radar:
        categories = ["Success Rate", "Avg Reward", "Speed", "Path Efficiency", "Consistency"]

        dq_sr = dq_success / n_ep
        ss_sr = ss_success / n_ep
        max_reward = max(abs(dq_rewards_arr.mean()), abs(ss_rewards_arr.mean()), 1)
        max_steps_val = max(dq_steps_arr.mean(), ss_steps_arr.mean(), 1)
        dq_path_eff = 1 - len(results["dq_path"]) / max(len(results["dq_path"]), len(results["ss_path"]), 1)
        ss_path_eff = 1 - len(results["ss_path"]) / max(len(results["dq_path"]), len(results["ss_path"]), 1)
        dq_consistency = 1 - min(dq_steps_arr.std() / max(dq_steps_arr.mean(), 1), 1)
        ss_consistency = 1 - min(ss_steps_arr.std() / max(ss_steps_arr.mean(), 1), 1)

        dq_vals = [dq_sr, (dq_rewards_arr.mean() + 100) / (max_reward + 100),
                    1 - dq_steps_arr.mean() / max_steps_val, dq_path_eff, dq_consistency]
        ss_vals = [ss_sr, (ss_rewards_arr.mean() + 100) / (max_reward + 100),
                    1 - ss_steps_arr.mean() / max_steps_val, ss_path_eff, ss_consistency]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=dq_vals + [dq_vals[0]], theta=categories + [categories[0]],
            fill="toself", name="Dyna-Q",
            fillcolor="rgba(52,152,219,0.2)", line=dict(color="#3498db"),
        ))
        fig.add_trace(go.Scatterpolar(
            r=ss_vals + [ss_vals[0]], theta=categories + [categories[0]],
            fill="toself", name="SARSA",
            fillcolor="rgba(231,76,60,0.2)", line=dict(color="#e74c3c"),
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            template="plotly_white", height=500,
            title="Performance Radar Chart",
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab_bar:
        fig = make_subplots(rows=1, cols=3, subplot_titles=[
            "Success Rate (%)", "Avg Steps/Episode", "Avg Reward",
        ])
        fig.add_trace(go.Bar(
            x=["Dyna-Q", "SARSA"], y=[dq_sr * 100, ss_sr * 100],
            marker_color=["#3498db", "#e74c3c"], showlegend=False,
        ), row=1, col=1)
        fig.add_trace(go.Bar(
            x=["Dyna-Q", "SARSA"],
            y=[float(dq_steps_arr.mean()), float(ss_steps_arr.mean())],
            marker_color=["#3498db", "#e74c3c"], showlegend=False,
        ), row=1, col=2)
        fig.add_trace(go.Bar(
            x=["Dyna-Q", "SARSA"],
            y=[float(dq_rewards_arr.mean()), float(ss_rewards_arr.mean())],
            marker_color=["#3498db", "#e74c3c"], showlegend=False,
        ), row=1, col=3)
        fig.update_layout(template="plotly_white", height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab_dist:
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=results["dq_steps"], nbinsx=30, name="Dyna-Q",
            marker_color="#3498db", opacity=0.6,
        ))
        fig.add_trace(go.Histogram(
            x=results["ss_steps"], nbinsx=30, name="SARSA",
            marker_color="#e74c3c", opacity=0.6,
        ))
        fig.update_layout(
            barmode="overlay", xaxis_title="Steps per Episode",
            yaxis_title="Frequency", template="plotly_white", height=400,
            title="Step Count Distribution",
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab_qdist:
        dq_nonzero = dq_agent.q_table[dq_agent.q_table != 0].flatten()
        ss_nonzero = ss_agent.q_table[ss_agent.q_table != 0].flatten()

        fig = make_subplots(rows=1, cols=2, subplot_titles=["Dyna-Q Q-Values", "SARSA Q-Values"])
        if len(dq_nonzero) > 0:
            fig.add_trace(go.Histogram(
                x=dq_nonzero.tolist(), nbinsx=50,
                marker_color="#3498db", showlegend=False,
            ), row=1, col=1)
        if len(ss_nonzero) > 0:
            fig.add_trace(go.Histogram(
                x=ss_nonzero.tolist(), nbinsx=50,
                marker_color="#e74c3c", showlegend=False,
            ), row=1, col=2)
        fig.update_layout(template="plotly_white", height=350, title="Q-Value Distributions")
        st.plotly_chart(fig, use_container_width=True)

    with tab_epsilon:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(n_ep)), y=[dq_agent.epsilon] * n_ep if False
            else [epsilon_start * (epsilon_decay ** i) for i in range(n_ep)],
            mode="lines", name="Dyna-Q", line=dict(color="#3498db", width=2),
        ))
        fig.add_trace(go.Scatter(
            x=list(range(n_ep)),
            y=[epsilon_start * (epsilon_decay ** i) for i in range(n_ep)],
            mode="lines", name="SARSA", line=dict(color="#e74c3c", width=2),
        ))
        fig.update_layout(
            xaxis_title="Episode", yaxis_title="Epsilon",
            template="plotly_white", height=350, title="Exploration Rate Decay",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.header("Export Results")

    import pandas as pd
    export_df = pd.DataFrame({
        "episode": list(range(n_ep)),
        "dyna_q_reward": results["dq_rewards"],
        "sarsa_reward": results["ss_rewards"],
        "dyna_q_steps": results["dq_steps"],
        "sarsa_steps": results["ss_steps"],
    })

    csv = export_df.to_csv(index=False)
    st.download_button(
        "Download Results CSV",
        csv,
        "mazemind_results.csv",
        "text/csv",
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
