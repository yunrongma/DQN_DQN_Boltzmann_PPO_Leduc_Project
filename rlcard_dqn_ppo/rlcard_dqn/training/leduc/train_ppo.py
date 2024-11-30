import rlcard
import sys
import os
import random
import numpy as np
import tensorflow as tf
import time

# Suppress TensorFlow warnings (optional)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO messages

# Add the root directory of your project to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from agents.ppo_agent import PPOAgentBase
from rlcard.agents.random_agent import RandomAgent
from rlcard.utils import tournament

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Load the game environment
env = rlcard.make('leduc-holdem')
eval_env = rlcard.make('leduc-holdem')

# Get the number of actions and state shape
action_num = env.game.get_num_actions()
state_shape = env.state_shape[0]

# Initialize the PPO agent
agent = PPOAgentBase(
    scope='ppo',
    action_num=action_num,
    state_shape=state_shape,
    mlp_layers=[64, 64],
    clip_ratio=0.2,
    learning_rate=0.0003,
    value_coef=0.5,
    entropy_coef=0.02,
    update_target_every=512,  # Number of timesteps per update
    gamma=0.99,
    lam=0.95,
    epochs=30,
    minibatch_size=64
)

# Initialize a random agent for the second player
random_agent = RandomAgent(num_actions=action_num)

# Set agents in the environment
env.set_agents([agent, random_agent])
eval_env.set_agents([agent, random_agent])

# Training parameters
num_episodes = 20000
eval_interval = 1000
eval_episodes = 1000

# Initialize variables for data collection
batch_states = []
batch_actions = []
batch_log_probs = []
batch_values = []
batch_rewards = []
batch_dones = []
total_timesteps = 0

# CHANGE: Initialize lists to store losses for plotting
policy_losses = []
value_losses = []
entropies = []

# CHANGE: Initialize variables for early stopping
best_avg_reward = -np.inf
patience = 10
patience_counter = 0

# Start timing
start_time = time.time()
results_file = open("ppo_results.txt", "w")

# Training loop
for episode in range(1, num_episodes + 1):
    trajectories, payoffs = env.run(is_training=True)
    agent_trajectory = trajectories[0]
    agent_payoff = payoffs[0]

    states = []
    actions = []
    log_probs = []
    values = []
    rewards = []
    dones = []

    # Process the trajectory
    for i in range(0, len(agent_trajectory) - 1, 2):
        ts = agent_trajectory[i]
        action = agent_trajectory[i + 1]

        state = ts['obs']
        obs = np.expand_dims(state, axis=0).astype(np.float32)
        logits = agent.policy_network(obs)
        action_probs = tf.nn.softmax(logits)
        log_probs_tensor = tf.nn.log_softmax(logits)
        value = agent.value_network(obs)
        action_mask = tf.one_hot(action, action_num)
        log_prob = tf.reduce_sum(log_probs_tensor * action_mask, axis=1).numpy()[0]
        value = value.numpy()[0][0]

        states.append(state.astype(np.float32))
        actions.append(action)
        log_probs.append(log_prob)
        values.append(value)
        rewards.append(0.0)  # Intermediate rewards are zero
        dones.append(False)  # Not done yet

    # Append the final step with the payoff
    if len(agent_trajectory) % 2 == 1:
        ts = agent_trajectory[-1]
        state = ts['obs']
        obs = np.expand_dims(state, axis=0).astype(np.float32)
        logits = agent.policy_network(obs)
        value = agent.value_network(obs)
        value = value.numpy()[0][0]

        states.append(state.astype(np.float32))
        actions.append(0)  # Placeholder action
        log_probs.append(0.0)  # Placeholder log_prob
        values.append(value)
        rewards.append(agent_payoff)  # Final reward is the payoff
        dones.append(True)  # Episode done

    # Accumulate data
    batch_states.extend(states)
    batch_actions.extend(actions)
    batch_log_probs.extend(log_probs)
    batch_values.extend(values)
    batch_rewards.extend(rewards)
    batch_dones.extend(dones)
    total_timesteps += len(states)

    # Update policy if enough timesteps have been collected
    if total_timesteps >= agent.update_target_every:
        # Convert batches to numpy arrays
        batch_states = np.array(batch_states)
        batch_actions = np.array(batch_actions)
        batch_log_probs = np.array(batch_log_probs)
        batch_values = np.array(batch_values)
        batch_rewards = np.array(batch_rewards)
        batch_dones = np.array(batch_dones, dtype=np.float32)

        # Compute advantages and returns
        advantages, returns = agent.compute_advantages(batch_rewards, batch_values, batch_dones)

        # Update the policy and get losses
        avg_policy_loss, avg_value_loss, avg_entropy = agent.update_policy(
            batch_states, batch_actions, batch_log_probs, returns, advantages)

        # CHANGE: Store the losses for potential plotting
        policy_losses.append(avg_policy_loss)
        value_losses.append(avg_value_loss)
        entropies.append(avg_entropy)

        # CHANGE: Optionally, print the losses
        print(f'Update Completed - Policy Loss: {avg_policy_loss:.4f}, Value Loss: {avg_value_loss:.4f}, Entropy: {avg_entropy:.4f}')

        # Reset batch data
        batch_states = []
        batch_actions = []
        batch_log_probs = []
        batch_values = []
        batch_rewards = []
        batch_dones = []
        total_timesteps = 0

    # Evaluate the agent's performance at regular intervals
    if episode % eval_interval == 0:
        avg_reward = tournament(eval_env, eval_episodes)[0]
        time_elapsed = time.time() - start_time
        result_line = f'PPO, Episode: {episode}, Average Reward: {avg_reward:.4f}, Time Elapsed: {time_elapsed:.2f}s\n'
        print(result_line)
        results_file.write(result_line)

        # CHANGE: Early stopping check
        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            patience_counter = 0
            # Optionally, save the best model
            agent.save('ppo_leduc_best_model')
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print('Early stopping triggered. Training has converged.')
            break  # Exit the training loop

# Save the trained model
agent.save('ppo_leduc_model')
results_file.close()
