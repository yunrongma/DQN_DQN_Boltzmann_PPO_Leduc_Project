import pandas as pd
import matplotlib.pyplot as plt

# Paths to your result files
dqn_results_path = 'experiments/dqn_results.txt'
ppo_results_path = 'experiments/ppo_results.txt'

# Load the DQN results (only the first 10 lines)
dqn_data = pd.read_csv(dqn_results_path, header=None, names=["Agent", "Episode", "AvgReward", "Time"], nrows=10)

# Load the PPO results
ppo_data = pd.read_csv(ppo_results_path, header=None, names=["Agent", "Episode", "AvgReward", "Time"])
plt.figure(figsize=(10, 6), dpi=150)  # 10x6 inches and 150 DPI
# Plot average rewards
plt.plot(dqn_data['Episode'], dqn_data['AvgReward'], label='DQN')
plt.plot(ppo_data['Episode'], ppo_data['AvgReward'], label='PPO')
plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.legend()
plt.title('DQN vs. PPO Average Rewards in Leduc Hold\'em')
plt.savefig('experiments/ppo_vs_dqn.png')
plt.show()
