import numpy as np
import matplotlib.pyplot as plt
import sys

def load_dqn_results(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            # Ensure the line contains the expected keywords
            if "Episode" in line and "Average Reward" in line:
                try:
                    # Extract Episode
                    episode_str = line.split("Episode:")[1].split(",")[0].strip()
                    episode = int(episode_str)

                    # Extract Average Reward
                    reward_str = line.split("Average Reward:")[1].split(",")[0].strip()
                    reward = float(reward_str)

                    # Append parsed data
                    data.append((episode, reward))
                except (IndexError, ValueError) as e:
                    print(f"Skipping malformed line: {line.strip()}")
                    print(f"Error: {e}")
    if not data:
        print("No valid data found in the file.")
    return np.array(data)

# File path to your data
file_path = "experiments/dqn_results.txt"
data = load_dqn_results(file_path)

# Handle empty data
if data.size == 0:
    print("No valid data to plot. Exiting.")
    sys.exit(1)

# Ensure correct dimensions
if data.ndim == 1:
    data = np.expand_dims(data, axis=0)

# Extract data for plotting
episodes = data[:, 0]
rewards = data[:, 1]

# Generate the plot
plt.figure(figsize=(10, 6))
plt.plot(episodes, rewards, label='DQN')

plt.xlabel('Episodes')
plt.ylabel('Average Reward')
plt.title('Average Reward over Episodes')
plt.legend()
plt.grid()
plt.show()