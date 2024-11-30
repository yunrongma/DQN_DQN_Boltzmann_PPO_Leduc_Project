import sys
import os

parent_dir = os.path.dirname(".")
sys.path.append(parent_dir)

import rlcard
import random
import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt

# Add the root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from agents.dqn_agent import DQNAgentBase
from agents.dqn_boltzmann_agent import DQNAgentWithBoltzmann
from rlcard.agents.random_agent import RandomAgent
from rlcard.utils import tournament

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Initialize environment
env = rlcard.make('leduc-holdem')
eval_env = rlcard.make('leduc-holdem')
action_num = env.game.get_num_actions()
state_shape = env.state_shape[0]

# Initialize DQN Boltzmann agent
boltzmann_agent = DQNAgentWithBoltzmann(
    scope='dqn_boltzmann',
    action_num=action_num,
    state_shape=state_shape,
    mlp_layers=[32],
    replay_memory_size=5000,
    replay_memory_init_size=300,
    update_target_estimator_every=200,
    discount_factor=0.99,
    batch_size=32,
    learning_rate=0.0005,
    mode = 'boltzmann',
    extra_params=5  #temperature
)

# dqn_agent = DQNAgentBase(
#     scope='dqn',
#     action_num=action_num,
#     state_shape=state_shape,
#     replay_memory_size=5000,
#     replay_memory_init_size=300,
#     update_target_estimator_every=200,
#     epsilon_start=1,
#     epsilon_end = 0.05,
#     epsilon_decay_steps=3000,
#     discount_factor=0.99,
#     batch_size=32,
#     learning_rate=0.001,
#     mode = 'dqn'
# )

dqn_agent = DQNAgentBase(
    scope='dqn',
    action_num=action_num,
    state_shape=state_shape,
    replay_memory_size=20000,
    replay_memory_init_size=300,
    update_target_estimator_every=200,
    epsilon_start=1,
    epsilon_end = 0.01,
    epsilon_decay_steps=3000,
    discount_factor=0.99,
    batch_size=32,
    learning_rate=0.001,
    mode = 'dqn'
)

# Random agent as opponent
random_agent = RandomAgent(num_actions=action_num)

# Training parameters
total_episodes = 20000
eval_interval = 500
eval_episodes = 200
results_file = os.path.join('experiments', 'boltzmann_results2.txt')
# results_file = os.path.join('experiments', 'dqn_results.txt')
save_dir = 'models'
os.makedirs(save_dir, exist_ok=True)
os.makedirs(os.path.dirname(results_file), exist_ok=True)

def train_agent(agent):
    print(f"Prepopulating replay memory for {agent.scope}...")
    env.set_agents([agent, random_agent])
    eval_env.set_agents([agent, random_agent])
    env.reset()
    eval_env.reset()

    # Properly set temperature based on mode
    for _ in range(agent.replay_memory_init_size):
        trajectories, payoffs = env.run(is_training=True)
        total_reward = payoffs[0]
        for i in range(0, len(trajectories[0]) - 1, 2):
            state = trajectories[0][i]['obs']
            action = trajectories[0][i + 1]
            next_state = trajectories[0][i + 2]['obs'] if i + 2 < len(trajectories[0]) else np.zeros_like(state)
            done = (i + 2 >= len(trajectories[0]))
            agent.feed((state, action, total_reward, next_state, done))
    print(f"Replay memory initialized for {agent.scope}.")
    print(f"Replay memory size: {len(agent.replay_memory)} / {agent.replay_memory_init_size}")
    
    episode_rewards = []
    start_time = time.time()

    for episode in range(1, total_episodes + 1):
        trajectories, payoffs = env.run(is_training=True)
        total_reward = payoffs[0]
        episode_loss = []

        for i in range(0, len(trajectories[0]) - 1, 2):
            state = trajectories[0][i]['obs']
            action = trajectories[0][i + 1]
            next_state = trajectories[0][i + 2]['obs'] if i + 2 < len(trajectories[0]) else np.zeros_like(state)
            done = (i + 2 >= len(trajectories[0]))
            agent.feed((state, action, total_reward, next_state, done))

        if len(agent.replay_memory) >= agent.batch_size:
            batch = random.sample(agent.replay_memory, agent.batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            loss = agent.update(states, actions, rewards, next_states, dones)
            episode_loss.append(loss)

        if episode % eval_interval == 0:
            avg_reward = tournament(eval_env, eval_episodes)[0]
            avg_loss = np.mean(episode_loss) if episode_loss else 0
            episode_rewards.append((episode, avg_reward))
            print(f"[{agent.scope}] Episode {episode}: Average Reward = {avg_reward:.4f}, Average Loss = {avg_loss:.6f}")

            with open(results_file, 'a') as f:
                if agent.mode == 'boltzmann':
                    f.write(f"{agent.scope} Episode {episode}: Temperature = {agent.temperature:.4f}, Average Reward = {avg_reward:.4f}, "
                            f"Average Loss: {avg_loss:.6f}, Time Elapsed: {time.time() - start_time:.2f}s\n")             
                else:  # Argmax mode
                    f.write(f"{agent.scope}, Episode: {episode}, Average Reward: {avg_reward:.4f}, "
                            f"Average Loss: {avg_loss:.6f}, Time Elapsed: {time.time() - start_time:.2f}s\n")
            
            if agent.mode == 'boltzmann':
                agent.decay_temperature(episode)

    # Save the model with the desired naming format
    if  agent.mode == 'boltzmann':  # Boltzmann agent with specific temperature
        agent.save(os.path.join(save_dir, f"new4_boltzmann_T{agent.extra_params}_model.keras"))
    elif agent.mode == 'dqn':        # DQN agent
        agent.save(os.path.join(save_dir, "dqn_model4.keras"))

    print(f"[{agent.scope}] Training complete. Model saved.")

    temp_episodes, temp_rewards = zip(*episode_rewards)

    plt.figure(figsize=(10, 6))
    if agent.mode == 'boltzmann':
        plt.plot(temp_episodes, temp_rewards, label=f'DQN_Boltzmann (T={agent.extra_params})')
        tfile = 'temperature_' + str(agent.extra_params) + '.npz'
        np.savez(tfile,array1 = temp_episodes, array2 = temp_rewards)
    elif agent.mode == 'dqn':
        plt.plot(temp_episodes, temp_rewards, label=f'DQN')
        np.savez('dqn.npz',array1 = temp_episodes, array2 = temp_rewards)

    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.title(f'Reward Comparison: {agent.scope}')
    plt.legend()
    plt.grid()
    if agent.mode == 'boltzmann':
        plt.savefig(f'reward_comparison_T{boltzmann_agent.extra_params}.png')
    elif agent.mode == 'dqn':
        plt.savefig(f'reward_comparison_dqn.png')

    return episode_rewards

# ret = train_agent(dqn_agent)
boltzmann_agent.temperature_test_action(2)
ret = train_agent(boltzmann_agent)

'''
def create_greedy_temperature_plt_from_file():
    with np.load('greedy.npz') as data:
        episodes_greedy = data['array1']
        rewards_greedy = data['array2']

    episodes_boltzmann = []
    rewards_boltzmann = []
    for i in range(len(temperatureList)):
        tfile = 'temperature_' + str(temperatureList[i]) + '.npz'
        with np.load(tfile) as data:
            episodes = data['array1']
            rewards = data['array2']
        episodes_boltzmann.append(episodes)
        rewards_boltzmann.append(rewards)

    for i in range(len(temperatureList)):
        plt.figure(figsize=(10, 6))

        plt.plot(episodes_greedy, rewards_greedy, label='DQN_Greedy')
        # Plot Boltzmann agent results for the specific temperature
        plt.plot(episodes_boltzmann[i], rewards_boltzmann[i], label=f'Boltzmann (T={temperatureList[i]})')
        plt.xlabel('Episodes')
        plt.ylabel('Average Reward')
        plt.title('Reward Comparison: DQN_Boltzmann vs Greedy')
        plt.legend()
        plt.grid()
        # Save each graph for the respective temperature
        plt.savefig(f'reward_comparison_T_{temperatureList[i]}.png')

create_greedy_temperature_plt_from_file()


dqn_rewards = train_agent(dqn_agent, 'DQN', -1)
temp_episodes_greedy, temp_rewards_greedy = zip(*dqn_rewards)
np.savez('dqn.npz',array1 = temp_episodes_greedy, array2 = temp_rewards_greedy)
with np.load('dqn.npz') as data:
    episodes_greedy = data['array1']
    rewards_greedy = data['array2']

plt.figure(figsize=(10, 6))
plt.plot(episodes_greedy, rewards_greedy, label='DQN')
plt.xlabel('Episodes')
plt.ylabel('Average Reward')
plt.title('Reward DQN')
plt.legend()
plt.grid()
plt.savefig(f'dqn.png')

# Train Argmax baseline (using Boltzmann agent in Argmax mode)
print("Training DQN_Boltzmann in Greedy mode...")
# boltzmann_agent.temperature = 0  # Set Boltzmann agent to Argmax mode

greedy_rewards = train_agent(greedy_agent, 'DQN_Greedy', -1)
temp_episodes_greedy, temp_rewards_greedy = zip(*greedy_rewards)
np.savez('greedy.npz',array1 = temp_episodes_greedy, array2 = temp_rewards_greedy)
with np.load('greedy.npz') as data:
    episodes_greedy = data['array1']
    rewards_greedy = data['array2']

plt.figure(figsize=(10, 6))
plt.plot(episodes_greedy, rewards_greedy, label='DQN_Greedy')
plt.xlabel('Episodes')
plt.ylabel('Average Reward')
plt.title('Reward Greedy')
plt.legend()
plt.grid()
plt.savefig(f'greedy.png')

# Train Boltzmann agents with different temperatures
episodes_boltzmann = []
rewards_boltzmann = []

for temp in temperatureList:
    temp_rewards = train_agent(boltzmann_agent, f'DQN_Boltzmann_Temperature_{temp}', temp)
    temp_episodes, temp_rewards = zip(*temp_rewards)
    tfile = 'temperature_' + str(temp) + '.npz'
    np.savez(tfile,array1 = temp_episodes, array2 = temp_rewards)
    with np.load(tfile) as data:
        episodes = data['array1']
        rewards = data['array2']
    episodes_boltzmann.append(episodes)
    rewards_boltzmann.append(rewards)

# Plot reward comparison
for i in range(len(temperatureList)):
    plt.figure(figsize=(10, 6))
    # Add the Argmax baseline for comparison
    #plt.plot(episodes_greedy, rewards_greedy, label='DQN_Greedy')
    # Plot Boltzmann agent results for the specific temperature
    plt.plot(episodes_boltzmann[i], rewards_boltzmann[i], label=f'Boltzmann (T={temperatureList[i]})')
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.title('Reward Comparison: DQN_Boltzmann vs Greedy')
    plt.legend()
    plt.grid()
    # Save each graph for the respective temperature
    plt.savefig(f'reward_comparison_T{temperatureList[i]}.png')

print("Reward comparison plots saved.")
'''