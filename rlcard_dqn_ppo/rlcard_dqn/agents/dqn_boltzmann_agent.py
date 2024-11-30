import rlcard
import sys
import os
import random
import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt
from collections import deque

# Add the root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from rlcard.agents.random_agent import RandomAgent
from rlcard.utils import tournament

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

import numpy as np
import tensorflow as tf
from collections import deque
import random


class DQNAgentWithBoltzmann:
    def __init__(self, scope, action_num, state_shape, mlp_layers,
                 replay_memory_size=10000,
                 replay_memory_init_size=1000,
                 update_target_estimator_every=500,
                 epsilon_start=1.0, epsilon_end=0.05,
                 epsilon_decay_steps=15000,
                 discount_factor=0.99, batch_size=64,
                 learning_rate=0.0001, tau=0.005, 
                 mode="boltzmann", extra_params=5):
        self.scope = scope
        self.action_num = action_num
        self.state_shape = state_shape
        self.mlp_layers = mlp_layers

        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.epsilon = epsilon_start

        self.temperature = extra_params
        self.initial_temperature = extra_params

        self.replay_memory = deque(maxlen=replay_memory_size)
        self.replay_memory_init_size = replay_memory_init_size

        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.update_target_estimator_every = update_target_estimator_every
        self.tau = tau

        self.learning_rate = learning_rate

        self.mode = mode
        self.extra_params = extra_params
        # Learning rate scheduler
        self.learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.learning_rate,
            decay_steps=5000,
            decay_rate=0.96,
            staircase=True
        )

        self.q_estimator = self.build_network()
        self.target_estimator = self.build_network()
        self.step_counter = 0
        self.use_raw = False

    def temperature_test_action(self,temperature):
        self.extra_params = temperature
        self.initial_temperature = temperature
        self.temperature = temperature
        
    def build_network(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=self.state_shape))
        for units in self.mlp_layers:
            model.add(tf.keras.layers.Dense(units, activation='relu'))
        model.add(tf.keras.layers.Dense(self.action_num, activation=None))

        #optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate_schedule)
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mse')
        return model

    def boltzmann_action(self, q_values):
        if self.temperature < 0.5:
            return np.argmax(q_values)

        max_clip_value = 100
        q_values = np.clip(q_values, -max_clip_value, max_clip_value)

        exp_q = np.exp(q_values / self.temperature)
        probs = exp_q / np.sum(exp_q) if np.sum(exp_q) > 0 else np.ones(self.action_num) / self.action_num
        return np.random.choice(self.action_num, p=probs)


    def eval_step(self, state):
        """Evaluate step: Select action without exploration."""
        obs = state['obs']
        q_values = self.q_estimator.predict(np.expand_dims(obs, axis=0), verbose=0)[0]
        action = np.argmax(q_values)
        return action, {}

    def decay_temperature(self, episode):
        # Dynamic decay for exploitability
        decay_rate = 0.001
        min_temp = 0.5
        self.temperature = max(min_temp, self.initial_temperature * np.exp(-decay_rate * episode))

    def feed(self, transition):
        self.replay_memory.append(transition)

    def step(self, state):
        obs = state['obs']
        q_values = self.q_estimator.predict(np.expand_dims(obs, axis=0), verbose=0)[0]
        action = self.boltzmann_action(q_values)
        
        return action

    def update(self, states, actions, rewards, next_states, dones):
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

    def save(self, filepath='dqn_boltzmann_model.keras'):
        self.q_estimator.save(filepath)

    def load(self, filepath='dqn_boltzmann_model.keras'):
        self.q_estimator = tf.keras.models.load_model(filepath)