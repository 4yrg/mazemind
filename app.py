"""Streamlit app: live training + discovery playback for Dyna-Q vs SARSA."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import time
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
from mazemind.utils.metrics import EpisodeMetrics, EpisodeSnapshot
from mazemind.visualization.maze_renderer import (
    render_maze, render_training_snapshot, render_discovery_comparison,
)
from mazemind.visualization.heatmap import (
    render_heatmap, render_q_value_map, render_model_knowledge, render_exploration_timeline,
)
from mazemind.visualization.training_viz import (
    render_side_by_side_training, render_playback_frame,
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


def run_live_training(maze, dq_agent, ss_agent, dq_env, ss_env,
                      n_episodes, max_steps, alpha, gamma, seed,
                      live_placeholder, progress_bar, update_every):
    dq_snapshots = []
    ss_snapshots = []
    dq_traj = []
    ss_traj = []
    dq_all_rewards = []
    ss_all_rewards = []
    dq_all_steps = []
    ss_all_steps = []
    dq_cumulative_visits = np.zeros((maze.size, maze.size))
    ss_cumulative_visits = np.zeros((maze.size, maze.size))

    snap_set = set(SNAPSHOT_EPISODES)

    for ep in range(n_episodes):
        si = dq_env.state_to_index(dq_env.reset())
        dq_total_reward = 0.0
        dq_trajectory = [dq_env.state]
        for step in range(max_steps):
            action = dq_agent.select_action(si)
            result = dq_env.step(action)
            nsi = dq_env.state_to_index(result.state)
            dq_total_reward += result.reward
            dq_trajectory.append(result.state)
            dq_agent.update(si, action, result.reward, nsi, alpha, gamma, result.done)
            si = nsi
            if result.done:
                break
        dq_agent.decay_epsilon()
        for pos in dq_trajectory:
            dq_cumulative_visits[pos[0], pos[1]] += 1
        dq_traj.append(dq_trajectory)
        dq_all_rewards.append(dq_total_reward)
        dq_all_steps.append(step + 1)

        si = ss_env.state_to_index(ss_env.reset())
        ss_total_reward = 0.0
        ss_trajectory = [ss_env.state]
        ss_action = ss_agent.select_action(si)
        for step in range(max_steps):
            result = ss_env.step(ss_action)
            nsi = ss_env.state_to_index(result.state)
            ss_total_reward += result.reward
            ss_trajectory.append(result.state)
            ss_next_action = ss_agent.select_action(nsi)
            ss_agent.update(si, ss_action, result.reward, nsi, alpha, gamma, result.done,
                            next_action=ss_next_action)
            si = nsi
            ss_action = ss_next_action
            if result.done:
                break
        ss_agent.decay_epsilon()
        for pos in ss_trajectory:
            ss_cumulative_visits[pos[0], pos[1]] += 1
        ss_traj.append(ss_trajectory)
        ss_all_rewards.append(ss_total_reward)
        ss_all_steps.append(step + 1)

        if ep in snap_set:
            dq_snapshots.append(EpisodeSnapshot(
                episode=ep, path=dq_trajectory, visit_counts=dq_cumulative_visits.copy(),
                model_size=len(dq_agent.model), planning_steps=dq_agent.n_planning_steps,
                q_table_snapshot=dq_agent.q_table.copy(), success=result.done,
                steps=step + 1, reward=dq_total_reward, epsilon=dq_agent.epsilon,
            ))
            ss_snapshots.append(EpisodeSnapshot(
                episode=ep, path=ss_trajectory, visit_counts=ss_cumulative_visits.copy(),
                model_size=0, planning_steps=0,
                q_table_snapshot=ss_agent.q_table.copy(), success=result.done,
                steps=step + 1, reward=ss_total_reward, epsilon=ss_agent.epsilon,
            ))

        if (ep + 1) % update_every == 0 or ep == n_episodes - 1:
            progress_bar.progress((ep + 1) / n_episodes, text=f"Episode {ep + 1}/{n_episodes}")
            fig = render_side_by_side_training(
                maze,
                dq_q_table=dq_agent.q_table.copy(),
                ss_q_table=ss_agent.q_table.copy(),
                dq_trajectory=dq_trajectory,
                ss_trajectory=ss_trajectory,
                dq_visits=dq_cumulative_visits.copy(),
                ss_visits=ss_cumulative_visits.copy(),
                episode=ep,
                dq_steps=step + 1, ss_steps=step + 1,
                dq_reward=dq_total_reward, ss_reward=ss_total_reward,
                dq_epsilon=dq_agent.epsilon, ss_epsilon=ss_agent.epsilon,
                dq_success=result.done, ss_success=result.done,
                dq_model_size=len(dq_agent.model),
            )
            live_placeholder.pyplot(fig)
            plt.close(fig)

    return (dq_snapshots, ss_snapshots, dq_traj, ss_traj,
            dq_all_rewards, ss_all_rewards, dq_all_steps, ss_all_steps,
            dq_cumulative_visits, ss_cumulative_visits)


def main():
    st.title("Mazemind: Tabular RL Maze Pathfinding")
    st.markdown("**Dyna-Q (Model-Based) vs SARSA (Model-Free)** - Live Training & Discovery Playback")

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

        st.subheader("Live Display")
        update_every = st.select_slider("Update every N episodes", options=[1, 2, 5, 10, 25, 50], value=5)

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
        live_placeholder = st.empty()

        dq_agent = DynaQAgent(n_planning_steps=n_planning, epsilon=epsilon_start, epsilon_decay=epsilon_decay)
        dq_env = MicromouseEnv(maze)
        ss_agent = SarsaAgent(epsilon=epsilon_start, epsilon_decay=epsilon_decay)
        ss_env = MicromouseEnv(maze)

        (dq_snaps, ss_snaps, dq_traj, ss_traj,
         dq_rewards, ss_rewards, dq_steps_list, ss_steps_list,
         dq_visits, ss_visits) = run_live_training(
            maze, dq_agent, ss_agent, dq_env, ss_env,
            n_episodes, max_steps, alpha, gamma, seed,
            live_placeholder, progress_bar, update_every,
        )

        progress_bar.progress(1.0, text="Extracting optimal paths...")
        dq_path = extract_optimal_path(dq_agent, MicromouseEnv(maze))
        ss_path = extract_optimal_path(ss_agent, MicromouseEnv(maze))

        dq_metrics, _, _, _ = train_with_snapshots(
            DynaQAgent(n_planning_steps=n_planning, epsilon=epsilon_start, epsilon_decay=epsilon_decay),
            MicromouseEnv(maze), n_episodes=n_episodes, max_steps=max_steps,
            alpha=alpha, gamma=gamma, seed=seed, agent_name="Dyna-Q", maze_name=maze.name,
            snapshot_episodes=[],
        )
        ss_metrics = dq_metrics
        from mazemind.utils.metrics import TrainingMetrics
        dq_metrics = TrainingMetrics(agent_name="Dyna-Q", maze_name=maze.name)
        ss_metrics = TrainingMetrics(agent_name="SARSA", maze_name=maze.name)
        for i in range(len(dq_rewards)):
            dq_metrics.episodes.append(EpisodeMetrics(
                episode=i, total_reward=dq_rewards[i], steps=dq_steps_list[i],
                success=dq_rewards[i] > 0, epsilon=epsilon_start * (epsilon_decay ** i),
            ))
            ss_metrics.episodes.append(EpisodeMetrics(
                episode=i, total_reward=ss_rewards[i], steps=ss_steps_list[i],
                success=ss_rewards[i] > 0, epsilon=epsilon_start * (epsilon_decay ** i),
            ))

        st.session_state.results = {
            "maze": maze, "dq_agent": dq_agent, "ss_agent": ss_agent,
            "dq_metrics": dq_metrics, "ss_metrics": ss_metrics,
            "dq_snapshots": dq_snaps, "ss_snapshots": ss_snaps,
            "dq_traj": dq_traj, "ss_traj": ss_traj,
            "dq_rewards": np.array(dq_rewards), "ss_rewards": np.array(ss_rewards),
            "dq_steps": np.array(dq_steps_list), "ss_steps": np.array(ss_steps_list),
            "dq_path": dq_path, "ss_path": ss_path,
            "dq_env": dq_env, "ss_env": ss_env,
            "n_episodes": n_episodes,
        }
        progress_bar.empty()
        st.rerun()

    results = st.session_state.results
    if results is None:
        st.info("Configure parameters in the sidebar and click **Run Training** to start.")
        return

    maze = results["maze"]
    n_episodes = results["n_episodes"]

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
    st.header("Environment Discovery")

    tab_playback, tab_timeline, tab_coverage, tab_model, tab_technique = st.tabs([
        "Discovery Playback", "Discovery Timeline", "Exploration Coverage",
        "Model Knowledge (Dyna-Q)", "Technique Comparison",
    ])

    dq_snaps = results["dq_snapshots"]
    ss_snaps = results["ss_snapshots"]

    with tab_playback:
        st.subheader("Discovery Playback")
        st.markdown("Select any episode and step through it with Q-table and policy visualization.")

        pb_agent = st.radio("Select agent:", ["Dyna-Q", "SARSA"], horizontal=True, key="pb_agent")

        dq_traj = results["dq_traj"]
        ss_traj = results["ss_traj"]

        traj = dq_traj if pb_agent == "Dyna-Q" else ss_traj
        snaps = dq_snaps if pb_agent == "Dyna-Q" else ss_snaps

        pb_col1, pb_col2, pb_col3 = st.columns([2, 2, 1])
        with pb_col1:
            pb_episode = st.slider("Episode", 0, len(traj) - 1, 0, key="pb_ep")
        with pb_col2:
            ep_traj = traj[pb_episode]
            pb_step = st.slider("Step", 0, max(len(ep_traj) - 1, 0), 0, key="pb_step")
        with pb_col3:
            pb_speed = st.select_slider("Speed", options=[0.5, 0.3, 0.15, 0.05], value=0.15, key="pb_speed",
                                         format_func=lambda x: f"{1/x:.0f} fps")

        snap_idx = 0
        for i, s in enumerate(snaps):
            if s.episode <= pb_episode:
                snap_idx = i
            else:
                break
        nearest_snap = snaps[snap_idx]

        dq_r = results["dq_rewards"][pb_episode] if pb_episode < len(results["dq_rewards"]) else 0
        ss_r = results["ss_rewards"][pb_episode] if pb_episode < len(results["ss_rewards"]) else 0
        dq_s = results["dq_steps"][pb_episode] if pb_episode < len(results["dq_steps"]) else 0
        ss_s = results["ss_steps"][pb_episode] if pb_episode < len(results["ss_steps"]) else 0

        if pb_agent == "Dyna-Q":
            fig = render_playback_frame(
                maze, nearest_snap.q_table_snapshot, ep_traj, pb_step,
                nearest_snap.visit_counts, agent_name="Dyna-Q",
                episode=pb_episode, success=dq_r > 0, steps=dq_s,
                reward=dq_r, epsilon=nearest_snap.epsilon, model_size=nearest_snap.model_size,
            )
        else:
            fig = render_playback_frame(
                maze, nearest_snap.q_table_snapshot, ep_traj, pb_step,
                nearest_snap.visit_counts, agent_name="SARSA",
                episode=pb_episode, success=ss_r > 0, steps=ss_s,
                reward=ss_r, epsilon=nearest_snap.epsilon,
            )

        playback_placeholder = st.empty()
        playback_placeholder.pyplot(fig)
        plt.close(fig)

        play_col1, play_col2 = st.columns(2)
        with play_col1:
            if st.button("Play Episode", key="pb_play", use_container_width=True):
                for i in range(len(ep_traj)):
                    if pb_agent == "Dyna-Q":
                        fig = render_playback_frame(
                            maze, nearest_snap.q_table_snapshot, ep_traj, i,
                            nearest_snap.visit_counts, agent_name="Dyna-Q",
                            episode=pb_episode, success=dq_r > 0, steps=dq_s,
                            reward=dq_r, epsilon=nearest_snap.epsilon, model_size=nearest_snap.model_size,
                        )
                    else:
                        fig = render_playback_frame(
                            maze, nearest_snap.q_table_snapshot, ep_traj, i,
                            nearest_snap.visit_counts, agent_name="SARSA",
                            episode=pb_episode, success=ss_r > 0, steps=ss_s,
                            reward=ss_r, epsilon=nearest_snap.epsilon,
                        )
                    playback_placeholder.pyplot(fig)
                    plt.close(fig)
                    time.sleep(pb_speed)
        with play_col2:
            if st.button("Play All Episodes", key="pb_playall", use_container_width=True):
                for ep_num in range(0, len(traj), max(1, len(traj) // 50)):
                    ep_t = traj[ep_num]
                    s_idx = 0
                    for i, s in enumerate(snaps):
                        if s.episode <= ep_num:
                            s_idx = i
                    ns = snaps[s_idx]
                    for step_i in range(0, len(ep_t), max(1, len(ep_t) // 10)):
                        if pb_agent == "Dyna-Q":
                            fig = render_playback_frame(
                                maze, ns.q_table_snapshot, ep_t, step_i,
                                ns.visit_counts, agent_name="Dyna-Q",
                                episode=ep_num, success=results["dq_rewards"][ep_num] > 0 if ep_num < len(results["dq_rewards"]) else False,
                                steps=results["dq_steps"][ep_num] if ep_num < len(results["dq_steps"]) else 0,
                                reward=results["dq_rewards"][ep_num] if ep_num < len(results["dq_rewards"]) else 0,
                                epsilon=ns.epsilon, model_size=ns.model_size,
                            )
                        else:
                            fig = render_playback_frame(
                                maze, ns.q_table_snapshot, ep_t, step_i,
                                ns.visit_counts, agent_name="SARSA",
                                episode=ep_num, success=results["ss_rewards"][ep_num] > 0 if ep_num < len(results["ss_rewards"]) else False,
                                steps=results["ss_steps"][ep_num] if ep_num < len(results["ss_steps"]) else 0,
                                reward=results["ss_rewards"][ep_num] if ep_num < len(results["ss_rewards"]) else 0,
                                epsilon=ns.epsilon,
                            )
                        playback_placeholder.pyplot(fig)
                        plt.close(fig)
                        time.sleep(0.05)

    with tab_timeline:
        st.subheader("Episode-by-Episode Discovery")
        snap_episodes = [s.episode for s in dq_snaps]
        selected_ep = st.select_slider("Select episode:", options=snap_episodes,
                                        value=snap_episodes[min(3, len(snap_episodes) - 1)])
        dq_s = next(s for s in dq_snaps if s.episode == selected_ep)
        ss_s = next(s for s in ss_snaps if s.episode == selected_ep)

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
            st.info(f"**Dyna-Q**: explored **{dq_explored}/256** cells. Model: **{dq_s.model_size}** transitions.")
        with col2:
            fig, _ = render_training_snapshot(
                maze, ss_s.episode, ss_s.path, ss_s.visit_counts,
                agent_name="SARSA", success=ss_s.success, steps=ss_s.steps, reward=ss_s.reward,
            )
            st.pyplot(fig)
            plt.close(fig)
            ss_explored = int(np.count_nonzero(ss_s.visit_counts))
            st.info(f"**SARSA**: explored **{ss_explored}/256** cells. No internal model.")

    with tab_coverage:
        st.subheader("Exploration Coverage")
        total_cells = maze.size * maze.size
        dq_cov = [int(np.count_nonzero(s.visit_counts)) / total_cells * 100 for s in dq_snaps]
        ss_cov = [int(np.count_nonzero(s.visit_counts)) / total_cells * 100 for s in ss_snaps]
        dq_mod = [s.model_size / (total_cells * 4) * 100 for s in dq_snaps]
        ep_labels = [s.episode for s in dq_snaps]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ep_labels, y=dq_cov, mode="lines+markers", name="Dyna-Q Visits",
                                  line=dict(color="#3498db", width=2), marker=dict(size=8)))
        fig.add_trace(go.Scatter(x=ep_labels, y=dq_mod, mode="lines+markers", name="Dyna-Q Model",
                                  line=dict(color="#3498db", width=2, dash="dash"), marker=dict(size=8)))
        fig.add_trace(go.Scatter(x=ep_labels, y=ss_cov, mode="lines+markers", name="SARSA Visits",
                                  line=dict(color="#e74c3c", width=2), marker=dict(size=8)))
        fig.update_layout(xaxis_title="Episode", yaxis_title="% of Maze", template="plotly_white",
                          height=450, title="Exploration Coverage", yaxis_range=[-5, 105])
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
        model_snap = st.select_slider("Episode:", options=[s.episode for s in dq_snaps],
                                       value=dq_snaps[min(3, len(dq_snaps) - 1)].episode, key="model_snap")
        dq_s = next(s for s in dq_snaps if s.episode == model_snap)
        col1, col2 = st.columns(2)
        with col1:
            fig, _ = render_model_knowledge(maze, results["dq_agent"].model)
            st.pyplot(fig)
            plt.close(fig)
        with col2:
            fig, _ = render_q_value_map(dq_s.q_table_snapshot, maze, title=f"Q-Values at Ep {dq_s.episode}")
            st.pyplot(fig)
            plt.close(fig)
        st.markdown(f"Model: **{dq_s.model_size}** transitions | Planning: **{dq_s.planning_steps}** steps/real step")

    with tab_technique:
        st.subheader("Algorithm Comparison")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Dyna-Q (Model-Based)\n```\n1. Take action\n2. Observe (reward, next_state)\n3. Update Q (Q-learning)\n4. Store in model\n5. Plan N times from model\n```\nEach real step generates N simulated updates.")
            st.metric("Final Model Size", f"{dq_snaps[-1].model_size} transitions")
        with col2:
            st.markdown("### SARSA (Model-Free)\n```\n1. Take action\n2. Observe (reward, next_state)\n3. Choose next_action\n4. Update Q with actual next_action\n```\nNo internal model. 1 update per step.")
            st.metric("Internal Model", "None")

        st.markdown("---")
        dq_conv = dq_summary["episodes_to_convergence"]
        ss_conv = ss_summary["episodes_to_convergence"]
        fig = make_subplots(rows=1, cols=3, subplot_titles=["Convergence", "Avg Steps", "Success Rate"])
        fig.add_trace(go.Bar(x=["Dyna-Q", "SARSA"], y=[dq_conv or n_episodes, ss_conv or n_episodes],
                              marker_color=["#3498db", "#e74c3c"], showlegend=False), row=1, col=1)
        fig.add_trace(go.Bar(x=["Dyna-Q", "SARSA"], y=[dq_summary["mean_steps"], ss_summary["mean_steps"]],
                              marker_color=["#3498db", "#e74c3c"], showlegend=False), row=1, col=2)
        fig.add_trace(go.Bar(x=["Dyna-Q", "SARSA"], y=[dq_summary["success_rate"]*100, ss_summary["success_rate"]*100],
                              marker_color=["#3498db", "#e74c3c"], showlegend=False), row=1, col=3)
        fig.update_layout(template="plotly_white", height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        window = 50
        dq_r = results["dq_rewards"]
        ss_r = results["ss_rewards"]
        if len(dq_r) >= window:
            dq_sm = np.convolve(dq_r, np.ones(window)/window, mode="valid")
            ss_sm = np.convolve(ss_r, np.ones(window)/window, mode="valid")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(range(window-1, n_episodes)), y=dq_sm.tolist(),
                                      mode="lines", name="Dyna-Q", line=dict(color="#3498db", width=2)))
            fig.add_trace(go.Scatter(x=list(range(window-1, n_episodes)), y=ss_sm.tolist(),
                                      mode="lines", name="SARSA", line=dict(color="#e74c3c", width=2)))
            fig.update_layout(xaxis_title="Episode", yaxis_title="Reward", template="plotly_white",
                              height=400, title="Learning Curves")
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    import pandas as pd
    export_df = pd.DataFrame({
        "episode": list(range(n_episodes)),
        "dyna_q_reward": dq_r.tolist(), "sarsa_reward": ss_r.tolist(),
        "dyna_q_steps": results["dq_steps"].tolist(), "sarsa_steps": results["ss_steps"].tolist(),
    })
    csv = export_df.to_csv(index=False)
    st.download_button("Download CSV", csv, "mazemind_results.csv", "text/csv", use_container_width=True)


if __name__ == "__main__":
    main()
