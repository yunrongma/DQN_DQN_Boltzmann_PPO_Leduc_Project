import numpy as np
import tensorflow as tf
from agents.base_agent import BaseAgent  # Import the custom BaseAgent class
from collections import deque
import random

class PPOAgentBase(BaseAgent):
    def __init__(self, scope, action_num, state_shape, mlp_layers,
                 clip_ratio=0.2,
                 replay_memory_size=10000, batch_size=64,
                 learning_rate=0.0003,
                 value_coef=0.5, entropy_coef=0.01,
                 update_target_every=2048,
                 gamma=0.99, lam=0.95, epochs=10, minibatch_size=64):
        '''
        Initialize the PPO agent with parameters and neural network setup.
        '''
        self.scope = scope
        self.action_num = action_num
        self.state_shape = state_shape
        self.mlp_layers = mlp_layers
        self.clip_ratio = clip_ratio
        self.replay_memory = deque(maxlen=replay_memory_size)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.update_target_every = update_target_every
        self.gamma = gamma
        self.lam = lam
        self.epochs = epochs
        self.minibatch_size = minibatch_size

        self.use_raw = False

        self.step_counter = 0

        # Build policy and value networks
        self.build_networks()

        # Initialize optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    def build_networks(self):
        '''
        Build the policy and value networks.
        '''
        # Input layer
        inputs = tf.keras.layers.Input(shape=self.state_shape)

        # Hidden layers
        x = inputs
        for units in self.mlp_layers:
            x = tf.keras.layers.Dense(units, activation='relu')(x)

        # Policy output
        logits = tf.keras.layers.Dense(self.action_num, activation=None)(x)
        self.policy_network = tf.keras.Model(inputs=inputs, outputs=logits)

        # Value output
        value = tf.keras.layers.Dense(1, activation=None)(x)
        self.value_network = tf.keras.Model(inputs=inputs, outputs=value)

    def step(self, state):
        '''
        Select an action using the current policy (for training).
        :param state: The full state dictionary provided by RLCard environment.
        :return: Chosen action
        '''
        # Extract the actual observation data
        obs = state['obs']
        obs = np.expand_dims(obs, axis=0).astype(np.float32)
        logits = self.policy_network(obs)
        action_probs = tf.nn.softmax(logits).numpy()[0]
        action = np.random.choice(self.action_num, p=action_probs)
        return action

    def eval_step(self, state):
        '''
        Select an action for evaluation (no exploration).
        :param state: The full state dictionary provided by RLCard environment.
        :return: Tuple (Chosen action, Empty dictionary for RLCard)
        '''
        # Extract the actual observation data
        obs = state['obs']
        obs = np.expand_dims(obs, axis=0).astype(np.float32)
        logits = self.policy_network(obs)
        action_probs = tf.nn.softmax(logits).numpy()[0]
        action = np.argmax(action_probs)
        return action, {}  # RLCard expects an action and additional info (empty dict)

    def compute_advantages(self, rewards, values, dones):
        '''
        Compute Generalized Advantage Estimation (GAE)
        '''
        advantages = np.zeros_like(rewards)
        lastgaelam = 0
        for t in reversed(range(len(rewards))):
            if dones[t]:
                nextnonterminal = 0.0
                nextvalues = 0.0
            else:
                nextnonterminal = 1.0
                nextvalues = values[t + 1] if t + 1 < len(values) else 0.0
            delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        returns = advantages + values
        return advantages, returns

    def update_policy(self, states, actions, old_log_probs, returns, advantages):
        # Convert data to tensors with consistent data types
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        old_log_probs = tf.convert_to_tensor(old_log_probs, dtype=tf.float32)
        returns = tf.convert_to_tensor(returns, dtype=tf.float32)
        advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)

        dataset = tf.data.Dataset.from_tensor_slices((states, actions, old_log_probs, returns, advantages))
        dataset = dataset.shuffle(buffer_size=1024).batch(self.minibatch_size)

        # CHANGE: Initialize variables to accumulate losses
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        count = 0

        for _ in range(self.epochs):
            for batch_states, batch_actions, batch_old_log_probs, batch_returns, batch_advantages in dataset:
                policy_loss, value_loss, entropy = self.train_step(
                    batch_states, batch_actions, batch_old_log_probs, batch_returns, batch_advantages)
                total_policy_loss += policy_loss.numpy()
                total_value_loss += value_loss.numpy()
                total_entropy += entropy.numpy()
                count += 1

        # CHANGE: Calculate average losses
        avg_policy_loss = total_policy_loss / count
        avg_value_loss = total_value_loss / count
        avg_entropy = total_entropy / count

        # CHANGE: Return average losses
        return avg_policy_loss, avg_value_loss, avg_entropy

    def train_step(self, states, actions, old_log_probs, returns, advantages):
        with tf.GradientTape() as tape:
            # Forward pass
            logits = self.policy_network(states)
            values = self.value_network(states)
            values = tf.squeeze(values, axis=1)

            # Calculate log probabilities
            log_probs = tf.nn.log_softmax(logits)
            action_masks = tf.one_hot(actions, self.action_num)
            selected_log_probs = tf.reduce_sum(log_probs * action_masks, axis=1)

            # Calculate ratio
            ratios = tf.exp(selected_log_probs - old_log_probs)

            # Calculate surrogate losses
            surr1 = ratios * advantages
            surr2 = tf.clip_by_value(ratios, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
            policy_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))

            # Calculate value loss
            value_loss = tf.reduce_mean((returns - values) ** 2)

            # Calculate entropy bonus
            entropy = -tf.reduce_mean(tf.reduce_sum(tf.exp(log_probs) * log_probs, axis=1))
            entropy_loss = -self.entropy_coef * entropy

            # Total loss
            total_loss = policy_loss + self.value_coef * value_loss + entropy_loss

        # Compute gradients and update parameters
        grads = tape.gradient(total_loss, self.policy_network.trainable_variables + self.value_network.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.policy_network.trainable_variables + self.value_network.trainable_variables))

        # CHANGE: Return individual losses
        return policy_loss, value_loss, entropy

    def save(self, filepath='ppo_model'):
        '''
        Save the PPO model.
        :param filepath: Path to save the model
        '''
        self.policy_network.save_weights(filepath + '_policy.weights.h5')
        self.value_network.save_weights(filepath + '_value.weights.h5')

    # def load(self, filepath='ppo_model'):
    #     '''
    #     Load the PPO model.
    #     :param filepath: Path to load the model
    #     '''
    #     self.policy_network.load_weights(filepath + '_policy.weights.h5')
    #     self.value_network.load_weights(filepath + '_value.weights.h5')

    def load(self, policy_path=None, value_path=None):
        """
        Load the PPO model weights.
        :param policy_path: Path to the policy network weights file.
        :param value_path: Path to the value network weights file.
        """
        if policy_path:
            self.policy_network.load_weights(policy_path)
        if value_path:
            self.value_network.load_weights(value_path)
