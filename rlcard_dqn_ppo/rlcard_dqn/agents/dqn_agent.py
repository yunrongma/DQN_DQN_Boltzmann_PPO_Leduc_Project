import numpy as np
import tensorflow as tf
from collections import deque
import random
from agents.base_agent import BaseAgent

class DQNAgentBase(BaseAgent):
    def __init__(self, scope, action_num, state_shape, mlp_layers=[16],
                 replay_memory_size=1000, replay_memory_init_size=200,
                 update_target_estimator_every=300, skip_update_interval=3,
                 epsilon_start=1.0, epsilon_end=0.1, epsilon_decay_steps=3000,
                 discount_factor=0.99, batch_size=16, learning_rate=0.0005,
                 mode='normal'):
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
        self.replay_memory_init_size = replay_memory_init_size
        self.replay_memory = deque(maxlen=replay_memory_size)

        # Training parameters
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.update_target_estimator_every = update_target_estimator_every
        self.learning_rate = learning_rate
        self.skip_update_interval = skip_update_interval

        self.mode = mode

        # Q-network and target network
        self.q_estimator = self.build_network()
        self.target_estimator = self.build_network()

        # Loss tracking for convergence
        self.total_loss = 0
        self.loss_count = 0
        self.step_counter = 0
        self.moving_avg_loss = None
        self.moving_avg_window = 100  # Window size for moving average

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
        self.replay_memory.append(transition)

    def epsilon_greedy_action(self, q_values):
        # Always return the action with the highest Q-value   
        if np.random.rand() < self.epsilon:
            if(self.epsilon > self.epsilon_end):
                self.epsilon = self.epsilon - (self.epsilon_start  - self.epsilon_end)/self.epsilon_decay_steps
            return np.random.choice(self.action_num)
        return np.argmax(q_values)
    
    def step(self, state):
        obs = state['obs']
        if np.random.rand() < self.epsilon:
            if(self.epsilon > self.epsilon_end):
                self.epsilon = self.epsilon - (self.epsilon_start  - self.epsilon_end)/self.epsilon_decay_steps
            return np.random.choice(self.action_num)
        
        q_values = self.q_estimator.predict(np.expand_dims(obs, axis=0), verbose=0)[0]
        return np.argmax(q_values)

    def eval_step(self, state):
        """Evaluate step: Select action without exploration."""
        obs = state['obs']
        q_values = self.q_estimator.predict(np.expand_dims(obs, axis=0), verbose=0)[0]
        action = np.argmax(q_values)
        return action, {}

    def update(self, states, actions, rewards, next_states, dones):
        # Copy the update method from DQNAgentWithBoltzmann
        states = np.array(states)
        next_states = np.array(next_states)

        actions = np.array(actions, dtype=np.int32)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)

        q_values_next = self.target_estimator.predict(next_states, verbose=0)
        q_values = self.q_estimator.predict(states, verbose=0)

        targets = q_values.copy()
        for i in range(len(actions)):
            q_update = rewards[i] + (1 - dones[i]) * self.discount_factor * np.max(q_values_next[i])
            targets[i, actions[i]] = np.clip(q_update, -10, 10)

        history = self.q_estimator.fit(states, targets, epochs=1, verbose=0)
        loss = history.history['loss'][0]

        if self.step_counter % self.update_target_estimator_every == 0:
            self.target_estimator.set_weights(self.q_estimator.get_weights())

        self.step_counter += 1
        return loss

    def get_average_loss(self):
        if self.loss_count == 0:
            return 0
        avg_loss = self.total_loss / self.loss_count
        self.total_loss = 0
        self.loss_count = 0

        # Update moving average
        if self.moving_avg_loss is None:
            self.moving_avg_loss = avg_loss
        else:
            alpha = 2 / (self.moving_avg_window + 1)
            self.moving_avg_loss = alpha * avg_loss + (1 - alpha) * self.moving_avg_loss

        return avg_loss

    def save(self, filepath='dqn_model.keras'):
        self.q_estimator.save(filepath)

    def load(self, filepath='dqn_model.keras'):
        self.q_estimator = tf.keras.models.load_model(filepath)