import numpy as np
import matplotlib.pyplot as plt

def load_results(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            if "Episode" in line and "Average Reward" in line:
                # Extract Episode
                episode = int(line.split("Episode")[1].split(":")[0].strip())
                # Extract Average Reward
                reward = float(line.split("Average Reward =")[1].split(",")[0].strip())
                # Extract Temperature if it exists
                temperature_part = line.split("Temperature =")
                temperature = float(temperature_part[1].split(",")[0].strip()) if len(temperature_part) > 1 else None
                # Append parsed data
                data.append((episode, reward, temperature))
    return np.array(data)

# File path to your data
file_path = "experiments/boltzmann_results.txt"  # Update to the correct path
data = load_results(file_path)

# Extract data for plotting
episodes = data[:, 0]
rewards = data[:, 1]
temperature = data[0, 2] if len(data[0]) > 2 else None  # Assume temperature is constant

# Generate the plot
plt.figure(figsize=(10, 6))
label = f'DQN_Boltzmann_T{temperature:.1f}' if temperature is not None else 'DQN_Boltzmann'
plt.plot(episodes, rewards, label=label)

plt.xlabel('Episodes')
plt.ylabel('Average Reward')
plt.title('Average Reward over Episodes')
plt.legend()
plt.grid()
plt.show()