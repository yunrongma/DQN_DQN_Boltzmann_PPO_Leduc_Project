# import rlcard
# import sys
# import os
# import numpy as np
# import tensorflow as tf
# import random
# import logging

# # Ensure the directory for evaluation results exists
# results_dir = 'evaluation_results'
# os.makedirs(results_dir, exist_ok=True)

# # Set up logging
# logging.basicConfig(
#     filename=os.path.join(results_dir, 'evaluation_log.log'),
#     filemode='w',
#     format='%(asctime)s - %(message)s',
#     level=logging.INFO
# )

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

# from rlcard.utils import tournament
# from agents.dqn_agent import DQNAgentBase
# from agents.dqn_boltzmann_agent import DQNAgentWithBoltzmann
# from agents.ppo_agent import PPOAgentBase
# from rlcard.agents.random_agent import RandomAgent
# from rlcard.agents.cfr_agent import CFRAgent

# def set_global_seed(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     tf.random.set_seed(seed)

# # Set random seed for reproducibility
# set_global_seed(42)

# # Directory to save results
# results_file = os.path.join(results_dir, 'results_leduc_tournament.txt')

# # Initialize Leduc environment with step_back enabled
# env = rlcard.make('leduc-holdem', config={'allow_step_back': True})

# # Load pre-trained models for each agent type
# agents = {
#     'DQN': DQNAgentBase(
#         scope='dqn',
#         action_num=env.game.get_num_actions(),
#         state_shape=env.state_shape[0],
#         mlp_layers=[32]
#     ),
#     'DQN_Boltzmann': DQNAgentWithBoltzmann(
#         scope='dqn_boltzmann',
#         action_num=env.game.get_num_actions(),
#         state_shape=env.state_shape[0],
#         mlp_layers=[32],
#         temperature=1.0
#     ),
#     'PPO': PPOAgentBase(
#         scope='ppo',
#         action_num=env.game.get_num_actions(),
#         state_shape=env.state_shape[0],
#         mlp_layers=[64, 64]
#     )
# }

# # Load models
# agents['DQN'].load('models/dqn_model_leduc.keras')
# agents['DQN_Boltzmann'].load('models/dqn_boltzmann_leduc_model.keras')
# agents['PPO'].load(
#     policy_path='models/ppo_leduc_model_policy.weights.h5',
#     value_path='models/ppo_leduc_model_value.weights.h5'
# )

# logging.info("Agents loaded successfully.")

# # Add CFR-based agent for exploitability evaluation
# cfr_agent = CFRAgent(env)
# cfr_training_iterations = 10000  # Number of CFR iterations
# print(f"Training CFRAgent for {cfr_training_iterations} iterations...")
# for _ in range(cfr_training_iterations):
#     cfr_agent.train()
# logging.info("CFRAgent trained successfully.")

# # Add RandomAgent as a baseline
# random_agent = RandomAgent(num_actions=env.game.get_num_actions())
# agents['Random'] = random_agent

# # Evaluation settings
# agent_pairs = [
#     ('DQN', 'DQN_Boltzmann'),
#     ('DQN', 'PPO'),
#     ('DQN_Boltzmann', 'PPO'),
#     ('DQN', 'Random'),
#     ('DQN_Boltzmann', 'Random'),
#     ('PPO', 'Random')
# ]

# # Evaluation parameters
# eval_episodes = 1000

# # Evaluate exploitability
# print("Evaluating exploitability...")
# with open(results_file, 'w') as f:
#     f.write("Agent1 vs Agent2 - Avg Reward (Agent1), Avg Reward (Agent2), Win Rate (Agent1), Win Rate (Agent2), Draw Rate\n")
#     f.write("\nExploitability Results:\n")

# for agent_name, agent in agents.items():
#     if agent_name == 'Random':  # Skip exploitability for RandomAgent
#         continue

#     env.set_agents([agent, cfr_agent])
#     exploitability = 0
#     for _ in range(eval_episodes):
#         _, payoffs = env.run(is_training=False)
#         exploitability += -payoffs[0]  # Negative payoff indicates exploitable behavior

#     avg_exploitability = exploitability / eval_episodes
#     logging.info(f"Exploitability of {agent_name}: {avg_exploitability:.4f}")
#     with open(results_file, 'a') as f:
#         f.write(f"Exploitability of {agent_name}: {avg_exploitability:.4f}\n")
#     print(f"Exploitability of {agent_name}: {avg_exploitability:.4f}")

# # Evaluate agent pairs
# print("Evaluating agent pairs...")
# for agent1_name, agent2_name in agent_pairs:
#     logging.info(f"Starting evaluation of {agent1_name} vs {agent2_name}...")
#     print(f"Evaluating {agent1_name} vs {agent2_name}...")
    
#     # Set agents for the environment
#     env.set_agents([agents[agent1_name], agents[agent2_name]])

#     # Initialize win and draw counts
#     agent1_wins, agent2_wins, draws = 0, 0, 0

#     # Run episodes and tally wins
#     for _ in range(eval_episodes):
#         _, payoffs = env.run(is_training=False)
#         if payoffs[0] > payoffs[1]:
#             agent1_wins += 1
#         elif payoffs[1] > payoffs[0]:
#             agent2_wins += 1
#         else:
#             draws += 1

#     # Calculate win rates and draw rate
#     win_rate_agent1 = agent1_wins / eval_episodes
#     win_rate_agent2 = agent2_wins / eval_episodes
#     draw_rate = draws / eval_episodes

#     # Run a tournament to get average rewards
#     avg_reward_agent1, avg_reward_agent2 = tournament(env, eval_episodes)

#     # Log and save results
#     logging.info(f"{agent1_name} vs {agent2_name} - Avg Reward (Agent1): {avg_reward_agent1:.4f}, "
#                  f"Avg Reward (Agent2): {avg_reward_agent2:.4f}, "
#                  f"Win Rate (Agent1): {win_rate_agent1:.2%}, Win Rate (Agent2): {win_rate_agent2:.2%}, "
#                  f"Draw Rate: {draw_rate:.2%}")

#     with open(results_file, 'a') as f:
#         f.write(f"{agent1_name} vs {agent2_name} - Avg Reward ({agent1_name}): {avg_reward_agent1:.4f}, "
#                 f"Avg Reward ({agent2_name}): {avg_reward_agent2:.4f}, "
#                 f"Win Rate ({agent1_name}): {win_rate_agent1:.2%}, Win Rate ({agent2_name}): {win_rate_agent2:.2%}, "
#                 f"Draw Rate: {draw_rate:.2%}\n")

#     print(f"{agent1_name} vs {agent2_name}: Avg Reward = {avg_reward_agent1:.4f} (Agent1), "
#           f"{avg_reward_agent2:.4f} (Agent2), Win Rate (Agent1) = {win_rate_agent1:.2%}, "
#           f"Win Rate (Agent2) = {win_rate_agent2:.2%}, Draw Rate = {draw_rate:.2%}")

# print("Evaluation complete. Results saved to:", results_file)
# logging.info("Evaluation complete. Results saved.")


import rlcard
import sys
import os
import numpy as np
import tensorflow as tf
import random
import logging
import seaborn as sns
import matplotlib.pyplot as plt

# Ensure the directory for evaluation results exists
results_dir = 'evaluation_results'
os.makedirs(results_dir, exist_ok=True)

# Set up logging
logging.basicConfig(
    filename=os.path.join(results_dir, 'tournament_log.log'),
    filemode='w',
    format='%(asctime)s - %(message)s',
    level=logging.INFO
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from rlcard.utils import tournament
from agents.dqn_agent import DQNAgentBase
from agents.dqn_boltzmann_agent import DQNAgentWithBoltzmann
from agents.ppo_agent import PPOAgentBase
from rlcard.agents.random_agent import RandomAgent
from rlcard.agents.cfr_agent import CFRAgent

def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

# Set random seed for reproducibility
set_global_seed(42)

# Directory to save results
results_file = os.path.join(results_dir, 'tournament1.txt')

# Initialize Leduc environment with step_back enabled
env = rlcard.make('leduc-holdem', config={'allow_step_back': True})

# Load pre-trained models for each agent type
agents = {
    'DQN': DQNAgentBase(
        scope='dqn',
        action_num=env.game.get_num_actions(),
        state_shape=env.state_shape[0],
        mlp_layers=[32]
    ),
    'DQN_Boltzmann': DQNAgentWithBoltzmann(
        scope='dqn_boltzmann',
        action_num=env.game.get_num_actions(),
        state_shape=env.state_shape[0],
        mlp_layers=[32]
    ),
    'PPO': PPOAgentBase(
        scope='ppo',
        action_num=env.game.get_num_actions(),
        state_shape=env.state_shape[0],
        mlp_layers=[64, 64]
    )
}

# Load models
agents['DQN'].load('models/dqn_model4.keras')
agents['DQN_Boltzmann'].load('models/new4_boltzmann_T2_model.keras')
agents['PPO'].load(
    policy_path='models/ppo_leduc_model_policy.weights.h5',
    value_path='models/ppo_leduc_model_value.weights.h5'
)

logging.info("Agents loaded successfully.")

# # Add CFR-based agent for exploitability evaluation
# cfr_agent = CFRAgent(env)
# cfr_training_iterations = 10000  # Number of CFR iterations
# print(f"Training CFRAgent for {cfr_training_iterations} iterations...")

# # Sequential CFR training
# for _ in range(cfr_training_iterations):
#     cfr_agent.train()

# logging.info("CFRAgent trained successfully.")

# Add RandomAgent as a baseline
random_agent = RandomAgent(num_actions=env.game.get_num_actions())
agents['Random'] = random_agent

# Evaluation settings
agent_pairs = [
    ('DQN', 'DQN_Boltzmann'),
    ('DQN', 'PPO'),
    ('DQN_Boltzmann', 'PPO'),
    ('DQN', 'Random'),
    ('DQN_Boltzmann', 'Random'),
    ('PPO', 'Random')
]

# # Evaluate exploitability
# print("Evaluating exploitability...")
# exploitability_results = {}
# eval_exploitability_episodes = 10000
# for agent_name, agent in agents.items():
#     if agent_name == 'Random':  # Skip exploitability for RandomAgent
#         continue

#     env.set_agents([agent, cfr_agent])
#     exploitability = 0
#     for i in range(eval_exploitability_episodes):
#         _, payoffs = env.run(is_training=False)
#         exploitability += -payoffs[0]  # Negative payoff indicates exploitable behavior
#         if i % 1000 == 0:
#             print(f"Exploitability progress: {i}/{eval_exploitability_episodes}")

#     avg_exploitability = exploitability / eval_exploitability_episodes
#     exploitability_results[agent_name] = avg_exploitability
#     logging.info(f"Exploitability of {agent_name}: {avg_exploitability:.4f}")
#     print(f"Exploitability of {agent_name}: {avg_exploitability:.4f}")

# # Generate exploitability comparison graph
# plt.figure(figsize=(10, 6))
# agent_names = list(exploitability_results.keys())
# exploitability_values = list(exploitability_results.values())
# sns.barplot(x=agent_names, y=exploitability_values, palette="viridis")
# plt.title("Exploitability Comparison")
# plt.ylabel("Exploitability")
# plt.xlabel("Agents")
# plt.savefig(os.path.join(results_dir, 'exploitability_comparison.png'))
# plt.close()

# Evaluate agent pairs

# Evaluate agent pairs
print("Evaluating agent pairs...")
pair_results = []
eval_episodes = 1000  # Number of episodes for pair evaluation
with open(results_file, 'w') as f:
    f.write("Agent Pair Evaluation Results:\n")
    f.write("=" * 60 + "\n")

for agent1_name, agent2_name in agent_pairs:
    logging.info(f"Starting evaluation of {agent1_name} vs {agent2_name}...")
    print(f"Evaluating {agent1_name} vs {agent2_name}...")
    
    # Set agents for the environment
    env.set_agents([agents[agent1_name], agents[agent2_name]])

    # Initialize win and draw counts
    agent1_wins, agent2_wins, draws = 0, 0, 0
    reward_agent1, reward_agent2 = [], []

    # Run episodes and tally wins
    for _ in range(eval_episodes):
        _, payoffs = env.run(is_training=False)
        reward_agent1.append(payoffs[0])
        reward_agent2.append(payoffs[1])
        if payoffs[0] > payoffs[1]:
            agent1_wins += 1
        elif payoffs[1] > payoffs[0]:
            agent2_wins += 1
        else:
            draws += 1

    # Calculate win rates and rewards
    win_rate_agent1 = agent1_wins / eval_episodes
    win_rate_agent2 = agent2_wins / eval_episodes
    draw_rate = draws / eval_episodes
    avg_reward_agent1 = np.mean(reward_agent1)
    avg_reward_agent2 = np.mean(reward_agent2)

    # Log and save results
    logging.info(f"{agent1_name} vs {agent2_name} - Avg Reward (Agent1): {avg_reward_agent1:.4f}, "
                 f"Avg Reward (Agent2): {avg_reward_agent2:.4f}, "
                 f"Win Rate (Agent1): {win_rate_agent1:.2%}, Win Rate (Agent2): {win_rate_agent2:.2%}, "
                 f"Draw Rate: {draw_rate:.2%}")

    with open(results_file, 'a') as f:
        f.write(f"{agent1_name} vs {agent2_name} - Avg Reward ({agent1_name}): {avg_reward_agent1:.4f}, "
                f"Avg Reward ({agent2_name}): {avg_reward_agent2:.4f}, "
                f"Win Rate ({agent1_name}): {win_rate_agent1:.2%}, Win Rate ({agent2_name}): {win_rate_agent2:.2%}, "
                f"Draw Rate: {draw_rate:.2%}\n")

    print(f"{agent1_name} vs {agent2_name}: Avg Reward = {avg_reward_agent1:.4f} (Agent1), "
          f"{avg_reward_agent2:.4f} (Agent2), Win Rate (Agent1) = {win_rate_agent1:.2%}, "
          f"Win Rate (Agent2) = {win_rate_agent2:.2%}, Draw Rate = {draw_rate:.2%}")

    pair_results.append({
        'Agent1': agent1_name,
        'Agent2': agent2_name,
        'Win Rate Agent1': win_rate_agent1,
        'Win Rate Agent2': win_rate_agent2,
        'Avg Reward Agent1': avg_reward_agent1,
        'Avg Reward Agent2': avg_reward_agent2,
        'Draw Rate': draw_rate
    })

print("Evaluation complete. Results saved to:", results_file)
logging.info("Evaluation complete. Results saved.")