import matplotlib.pyplot as plt
import numpy as np

# Parsing functions for each agent type
def parse_dqn_results(file_path):
    episodes, rewards = [], []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith("dqn, Episode:"):
                try:
                    episode = int(line.split("Episode:")[1].split(",")[0].strip())
                    reward = float(line.split("Average Reward:")[1].split(",")[0].strip())
                    episodes.append(episode)
                    rewards.append(reward)
                except (IndexError, ValueError):
                    print(f"Skipping malformed line: {line.strip()}")
    return np.array(episodes), np.array(rewards)

def parse_dqn_boltzmann_results(file_path):
    episodes, rewards, temperatures = [], [], []
    with open(file_path, 'r') as file:
        for line in file:
            if "dqn_boltzmann Episode" in line:
                try:
                    episode = int(line.split("Episode ")[1].split(":")[0].strip())
                    reward = float(line.split("Average Reward =")[1].split(",")[0].strip())
                    temp = None
                    if "Temperature =" in line:
                        temp = float(line.split("Temperature =")[1].split(",")[0].strip())
                    episodes.append(episode)
                    rewards.append(reward)
                    temperatures.append(temp)
                except (IndexError, ValueError):
                    print(f"Skipping malformed line: {line.strip()}")
    return np.array(episodes), np.array(rewards), np.array(temperatures)


def parse_ppo_results(file_path):
    episodes, rewards = [], []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith("PPO, Episode:"):
                try:
                    episode = int(line.split("Episode:")[1].split(",")[0].strip())
                    reward = float(line.split("Average Reward:")[1].split(",")[0].strip())
                    episodes.append(episode)
                    rewards.append(reward)
                except (IndexError, ValueError):
                    print(f"Skipping malformed line: {line.strip()}")
    return np.array(episodes), np.array(rewards)

# File paths
dqn_file = "experiments/dqn_results.txt"
dqn_boltzmann_file = "experiments/boltzmann_results.txt"
ppo_file = "experiments/ppo_results.txt"

# Parse results for each agent
dqn_episodes, dqn_rewards = parse_dqn_results(dqn_file)
dqn_boltzmann_episodes, dqn_boltzmann_rewards, _ = parse_dqn_boltzmann_results(dqn_boltzmann_file)
ppo_episodes, ppo_rewards = parse_ppo_results(ppo_file)

# Plotting
plt.figure(figsize=(12, 8))

# Plot DQN
plt.plot(dqn_episodes, dqn_rewards, label="DQN", linestyle="-", marker="o")

# Plot DQN Boltzmann
plt.plot(dqn_boltzmann_episodes, dqn_boltzmann_rewards, label="DQN_Boltzmann_T2", linestyle="--", marker="s")

# Plot PPO
plt.plot(ppo_episodes, ppo_rewards, label="PPO", linestyle="-.", marker="^")

# Add labels, legend, and title
plt.xlabel("Episodes")
plt.ylabel("Average Reward")
plt.title("Comparison of Average Rewards for DQN, DQN_Boltzmann_T2, and PPO")
plt.legend()
plt.grid(True)

# Show plot
plt.tight_layout()
plt.show()