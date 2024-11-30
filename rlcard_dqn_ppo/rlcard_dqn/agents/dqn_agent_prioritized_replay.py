import numpy as np
import tensorflow as tf
from collections import deque
import random
from agents.base_agent import BaseAgent

class DQNAgentWithPER(BaseAgent):
    def __init__(self, scope, action_num, state_shape, mlp_layers=[16],
                 replay_memory_size=1000, replay_memory_init_size=200,
                 update_target_estimator_every=300, skip_update_interval=3,
                 epsilon_start=1.0, epsilon_end=0.1, epsilon_decay_steps=3000,
                 discount_factor=0.99, batch_size=16, learning_rate=0.0005,
                 alpha=0.6, beta=0.4, beta_increment_per_sampling=0.001):
        self.scope = scope
        self.action_num = action_num
        self.state_shape = state_shape
        self.mlp_layers = mlp_layers

        # Exploration parameters
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.epsilon = epsilon_start

        # Replay buffer
        self.replay_memory_size = replay_memory_size
        self.replay_memory_init_size = replay_memory_init_size  # Ensure this attribute is included
        self.replay_memory = deque(maxlen=replay_memory_size)
        self.priorities = deque(maxlen=replay_memory_size)

        # Training parameters
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.update_target_estimator_every = update_target_estimator_every
        self.learning_rate = learning_rate
        self.skip_update_interval = skip_update_interval

        # PER parameters
        self.alpha = alpha  # Controls prioritization level
        self.beta = beta  # Importance-sampling weight
        self.beta_increment_per_sampling = beta_increment_per_sampling

        # Q-network and target network
        self.q_estimator = self.build_network()
        self.target_estimator = self.build_network()

        # Loss tracking
        self.total_loss = 0
        self.loss_count = 0
        self.step_counter = 0

        # Attribute for RLCard compatibility
        self.use_raw = False

    def build_network(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.InputLayer(shape=self.state_shape))
        for units in self.mlp_layers:
            model.add(tf.keras.layers.Dense(units, activation='relu'))
        model.add(tf.keras.layers.Dense(self.action_num, activation=None))
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mse')
        return model

    def feed(self, transition):
        max_priority = max(self.priorities) if self.priorities else 1.0
        self.replay_memory.append(transition)
        self.priorities.append(max_priority)

    def sample_batch(self):
        if len(self.replay_memory) < self.batch_size:
            return None

        priorities = np.array(self.priorities, dtype=np.float32)
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(len(self.replay_memory), self.batch_size, p=probabilities)
        samples = [self.replay_memory[idx] for idx in indices]
        importance_sampling_weights = (len(self.replay_memory) * probabilities[indices]) ** -self.beta
        importance_sampling_weights /= importance_sampling_weights.max()

        self.beta = min(1.0, self.beta + self.beta_increment_per_sampling)

        return samples, importance_sampling_weights, indices

    def update_priorities(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = abs(td_error) + 1e-5

    def step(self, state):
        obs = state['obs']
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_num)
        q_values = self.q_estimator.predict(np.expand_dims(obs, axis=0), verbose=0)[0]
        return np.argmax(q_values)

    def eval_step(self, state):
        obs = state['obs']
        q_values = self.q_estimator.predict(np.expand_dims(obs, axis=0), verbose=0)[0]
        action = np.argmax(q_values)
        return action, {}

    def update(self):
        batch = self.sample_batch()
        if batch is None:
            return

        samples, importance_sampling_weights, indices = batch
        states, actions, rewards, next_states, dones = map(np.array, zip(*samples))

        q_values_next = self.target_estimator.predict(next_states, verbose=0)
        q_values = self.q_estimator.predict(states, verbose=0)

        td_errors = []
        for i in range(self.batch_size):
            td_target = rewards[i] + (1 - dones[i]) * self.discount_factor * np.max(q_values_next[i])
            td_errors.append(td_target - q_values[i][actions[i]])
            q_values[i][actions[i]] = td_target

        self.update_priorities(indices, td_errors)

        history = self.q_estimator.fit(states, q_values, sample_weight=importance_sampling_weights, epochs=1, verbose=0)
        self.total_loss += history.history['loss'][0]
        self.loss_count += 1

        if self.step_counter % self.update_target_estimator_every == 0:
            self.target_estimator.set_weights(self.q_estimator.get_weights())

        self.step_counter += 1

    def get_average_loss(self):
        if self.loss_count == 0:
            return 0
        avg_loss = self.total_loss / self.loss_count
        self.total_loss = 0
        self.loss_count = 0
        return avg_loss

    def save(self, filepath='dqn_model.keras'):
        self.q_estimator.save(filepath)

    def load(self, filepath='dqn_model.keras'):
        self.q_estimator = tf.keras.models.load_model(filepath)