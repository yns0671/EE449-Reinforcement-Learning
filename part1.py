# -*- coding: utf-8 -*-
"""
Created on Sun May  4 00:39:48 2025

@author: Yunus Tosun
"""
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from utils import plot_value_function, plot_policy, plot_learning_curves



class MazeEnvironment:
    def __init__(self):
        # Define the maze layout, rewards, action space (up, down, left, right)
        self.maze = np.array([
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
            [1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1],
            [1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1],
            [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1],
            [1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0],
            [1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0],
            [1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 2, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0],
            [1, 1, 1, 0, 2, 0, 1, 0, 0, 3, 0]
        ])
        self.start_pos = (0,0) # Start position of the agent
        self.current_pos = self.start_pos
        self.state_penalty = -1
        self.trap_penalty = -100
        self.goal_reward = 100
        self.actions = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

    def reset(self):
        self.current_pos = self.start_pos
        return self.current_pos

    def step(self, action):
        move = self.actions[action]
        next_pos = (self.current_pos[0] + move[0], self.current_pos[1] + move[1])
        # Check if next_pos is out of bounds or hits wall (1)
        if (0 <= next_pos[0] < self.maze.shape[0] and
            0 <= next_pos[1] < self.maze.shape[1] and
            self.maze[next_pos] != 1):
            self.current_pos = next_pos
        # Determine reward
        cell_value = self.maze[self.current_pos]
        if cell_value == 0:
            reward = self.state_penalty
        elif cell_value == 2:
            reward = self.trap_penalty
        elif cell_value == 3:
            reward = self.goal_reward
        else:
            reward = self.state_penalty
        return self.current_pos, reward


class MazeTD0(MazeEnvironment): # Inherited from MazeEnvironment
    def __init__(self, maze, alpha=0.1, gamma=0.95, epsilon=0.2, episodes=10000):
        super().__init__()
        self.maze = maze
        self.alpha = alpha #Learning Rate
        self.gamma = gamma #Discount factor
        self.epsilon = epsilon #Exploration Rate
        self.episodes = episodes
        self.utility = np.random.uniform(low=-1, high=1, size=self.maze.shape)  # FILL HERE, Encourage exploration

    def choose_action(self, state):
        #Explore and Exploit: Choose the best action based on current utility values
        #Discourage invalid moves
        if np.random.rand() < self.epsilon:
            # Exploration: choose random valid action
            valid_actions = []
            for action, move in self.actions.items():
                new_x = state[0] + move[0]
                new_y = state[1] + move[1]
                if 0 <= new_x < self.maze.shape[0] and 0 <= new_y < self.maze.shape[1]:
                    if self.maze[new_x, new_y] != 1:  # not obstacle
                        valid_actions.append(action)
            return np.random.choice(valid_actions)
        else:
            # Exploitation: choose best action
            best_action = None
            best_utility = -np.inf
            for action, move in self.actions.items():
                new_x = state[0] + move[0]
                new_y = state[1] + move[1]
                if 0 <= new_x < self.maze.shape[0] and 0 <= new_y < self.maze.shape[1]:
                    if self.maze[new_x, new_y] != 1:
                        util = self.utility[new_x, new_y]
                        if util > best_utility:
                            best_utility = util
                            best_action = action
            return best_action

    def update_utility_value(self, current_state, reward, new_state):
        current_value = self.utility[current_state]  # FILL HERE
        new_value = self.utility[new_state]          # FILL HERE
        # TD(0) update formula
        self.utility[current_state] = current_value + self.alpha * (reward + self.gamma * new_value - current_value)  # FILL HERE

    def run_episodes(self):
        for ep in range(self.episodes):
            state = self.reset()
            done = False
            while not done:
                action = self.choose_action(state)
                next_state, reward = self.step(action)
                self.update_utility_value(state, reward, next_state)
                state = next_state
                if self.maze[state] in [2, 3]:  # Trap or Goal = terminal states
                    done = True
        return self.utility




# Assuming the Maze and MazeTD0 classes are already defined

# Set up the parameter combinations from Table 1
alpha_values = [0.001, 0.01, 0.1, 0.5, 1.0]
gamma_values = [0.10, 0.25, 0.50, 0.75, 0.95]
epsilon_values = [0, 0.2, 0.5, 0.8, 1.0]



def run_experiments():
    os.makedirs("results", exist_ok=True)
    json_paths = []
    snapshot_eps = [1, 50, 100, 1000, 5000, 10000]

    # Basit moving average
    def moving_average(data, window=100):
        c = np.cumsum(np.insert(data, 0, 0))
        ma = (c[window:] - c[:-window]) / window
        return np.concatenate([np.full(window-1, ma[0]), ma])

    DEFAULT_ALPHA, DEFAULT_GAMMA, DEFAULT_EPSILON = 0.1, 0.95, 0.2
    sweeps = [
        ("alpha", alpha_values, DEFAULT_GAMMA, DEFAULT_EPSILON),
        ("gamma", DEFAULT_ALPHA, gamma_values, DEFAULT_EPSILON),
        ("epsilon", DEFAULT_ALPHA, DEFAULT_GAMMA, epsilon_values),
    ]

    for name, a_vals, g_vals, e_vals in sweeps:
        alphas   = a_vals   if isinstance(a_vals, list) else [a_vals]
        gammas   = g_vals   if isinstance(g_vals, list) else [g_vals]
        epsilons = e_vals   if isinstance(e_vals, list) else [e_vals]

        for alpha in alphas:
            for gamma in gammas:
                for epsilon in epsilons:
                    label = f"{name}_{alpha if name=='alpha' else gamma if name=='gamma' else epsilon}"
                    print(f"\nRunning {label}: α={alpha}, γ={gamma}, ε={epsilon}")

                    env   = MazeEnvironment()
                    maze  = env.maze
                    agent = MazeTD0(maze, alpha=alpha, gamma=gamma, epsilon=epsilon, episodes=10000)

                    deltas = []
                    episode_rewards = []

                    for ep in range(1, agent.episodes + 1):
                        prev_util = agent.utility.copy()

                        s = agent.reset()
                        done = False
                        total_r = 0
                        step_count = 0
                        MAX_STEPS = 500  # episode başına adım limiti

                        while not done and step_count < MAX_STEPS:
                            a = agent.choose_action(s)
                            ns, r = agent.step(a)
                            agent.update_utility_value(s, r, ns)
                            s = ns
                            total_r += r
                            step_count += 1

                            if maze[s] in [2, 3]:
                                done = True

                        episode_rewards.append(total_r)
                        deltas.append(np.nansum(np.abs(agent.utility - prev_util)))

                        if ep in snapshot_eps:
                            U = agent.utility.copy()
                            plot_value_function(U, maze)
                            plot_policy(U, maze)

                    # Convergence plot
                    plt.figure()
                    plt.plot(deltas)
                    plt.title(f"Convergence ({label})")
                    plt.xlabel("Episode")
                    plt.ylabel("Sum(|ΔU|)")
                    plt.grid()
                    plt.savefig(f"results/convergence_{label}.png")
                    plt.show()

                    # JSON kaydet
                    avg_scores = moving_average(episode_rewards, 100).tolist()
                    data = {
                        "episode_rewards": episode_rewards,
                        "average_scores": avg_scores
                    }
                    fout = f"results/{label}.json"
                    with open(fout, "w") as f:
                        json.dump(data, f)
                    json_paths.append(fout)

    # Tüm learning-curve’ları tek grafikte ve results klasörüne kaydet
    plot_learning_curves(json_paths, output_file="results/learning_curves.png")








run_experiments()
