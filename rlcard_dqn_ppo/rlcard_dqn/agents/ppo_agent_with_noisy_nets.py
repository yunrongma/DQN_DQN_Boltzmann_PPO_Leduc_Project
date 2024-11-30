import numpy as np
import tensorflow as tf
from agents.base_agent import BaseAgent  # Import the custom BaseAgent class
from collections import deque
import random

class NoisyDense(tf.keras.layers.Layer):
    def __init__(self, units, activation=None, sigma_init=0.5, **kwargs):
        super(NoisyDense, self).__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.sigma_init = sigma_init

    def build(self, input_shape):
        # Define the shapes
        input_dim = int(input_shape[-1])
        sqrt_input_dim = np.sqrt(input_dim)

        # Initialize weights and biases
        # Weight parameters
        self.w_mu = self.add_weight(
            name='w_mu',
            shape=(input_dim, self.units),
            initializer=tf.random_uniform_initializer(-1 / sqrt_input_dim, 1 / sqrt_input_dim),
            trainable=True
        )
        self.w_sigma = self.add_weight(
            name='w_sigma',
            shape=(input_dim, self.units),
            initializer=tf.constant_initializer(self.sigma_init / sqrt_input_dim),
            trainable=True
        )

        # Bias parameters
        self.b_mu = self.add_weight(
            name='b_mu',
            shape=(self.units,),
            initializer=tf.random_uniform_initializer(-1 / sqrt_input_dim, 1 / sqrt_input_dim),
            trainable=True
        )
        self.b_sigma = self.add_weight(
            name='b_sigma',
            shape=(self.units,),
            initializer=tf.constant_initializer(self.sigma_init / sqrt_input_dim),
            trainable=True
        )

    def call(self, inputs, training=None):
        if training:
            # Sample noise
            input_dim = inputs.shape[-1]
            epsilon_input = self._scale_noise(input_dim)
            epsilon_output = self._scale_noise(self.units)
            epsilon_w = tf.matmul(tf.expand_dims(epsilon_input, -1), tf.expand_dims(epsilon_output, 0))
            epsilon_b = epsilon_output
        else:
            # During inference, set noise to zero
            epsilon_w = tf.zeros((inputs.shape[-1], self.units))
            epsilon_b = tf.zeros((self.units,))

        w = self.w_mu + self.w_sigma * epsilon_w
        b = self.b_mu + self.b_sigma * epsilon_b

        output = tf.matmul(inputs, w) + b
        if self.activation is not None:
            output = self.activation(output)
        return output

    def _scale_noise(self, size):
        # Sample noise from a standard normal distribution
        x = tf.random.normal([size])
        return tf.sign(x) * tf.sqrt(tf.abs(x))
    
class PPOAgentWithNoisyNets(BaseAgent):
    def __init__(self, scope, action_num, state_shape, mlp_layers, clip_ratio=0.2,
                 replay_memory_size=10000, batch_size=64, learning_rate=0.0003,
                 value_coef=0.5, entropy_coef=0.01, update_target_every=2048,
                 gamma=0.99, lam=0.95, epochs=10, minibatch_size=64, sigma_init=0.5):
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
        self.sigma_init = sigma_init

        self.use_raw = False

        self.step_counter = 0

        # Build policy and value networks
        self.build_networks()

        # Initialize optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    def build_networks(self):
        '''
        Build the policy and value networks with noisy layers.
        '''
        # Input layer
        inputs = tf.keras.layers.Input(shape=self.state_shape)

        # Hidden layers with noisy layers
        x = inputs
        for units in self.mlp_layers:
            x = NoisyDense(units, activation='relu', sigma_init=self.sigma_init)(x)

        # Policy output
        logits = NoisyDense(self.action_num, activation=None, sigma_init=self.sigma_init)(x)
        self.policy_network = tf.keras.Model(inputs=inputs, outputs=logits)

        # Value output
        value = NoisyDense(1, activation=None, sigma_init=self.sigma_init)(x)
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
        logits = self.policy_network(obs, training=True)  # Sample noise during training
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
        logits = self.policy_network(obs, training=False)  # No noise during evaluation
        action_probs = tf.nn.softmax(logits).numpy()[0]
        action = np.argmax(action_probs)
        return action, {}  # RLCard expects an action and additional info (empty dict)

    def feed(self, transition):
        '''
        Store a transition in the replay memory.
        :param transition: Tuple (state, action, reward, next_state, done, log_prob, value)
        '''
        # Ensure the transition is a tuple with (state, action, reward, next_state, done, log_prob, value)
        if isinstance(transition, tuple) and len(transition) == 7:
            self.replay_memory.append(transition)
        else:
            raise ValueError("Transition must be a tuple of (state, action, reward, next_state, done, log_prob, value)")

    def update(self):
        '''
        Train the PPO agent with experiences from the replay memory.
        '''
        if len(self.replay_memory) < self.batch_size:
            return

        # Unpack transitions
        transitions = list(self.replay_memory)
        states, actions, rewards, next_states, dones, log_probs, values = zip(*transitions)

        # Convert to arrays with appropriate dtypes
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int32)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)
        log_probs = np.array(log_probs, dtype=np.float32)
        values = np.array(values, dtype=np.float32)

        # Compute advantages and returns
        advantages, returns = self.compute_gae(rewards, values, dones)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Optimize policy and value networks
        dataset = tf.data.Dataset.from_tensor_slices((states, actions, advantages, returns, log_probs))
        dataset = dataset.shuffle(buffer_size=1024).batch(self.minibatch_size)

        for _ in range(self.epochs):
            for batch_states, batch_actions, batch_advantages, batch_returns, batch_old_log_probs in dataset:
                self.train_step(batch_states, batch_actions, batch_advantages, batch_returns, batch_old_log_probs)

        # Clear replay memory after update
        self.replay_memory.clear()


    def compute_gae(self, rewards, values, dones):
        '''
        Compute Generalized Advantage Estimation (GAE)
        '''
        advantages = np.zeros_like(rewards)
        lastgaelam = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) -1:
                nextnonterminal = 1.0 - dones[-1]
                nextvalues = values[-1]
            else:
                nextnonterminal = 1.0 - dones[t+1]
                nextvalues = values[t+1]
            delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        returns = advantages + values
        return advantages, returns

    @tf.function
    def train_step(self, states, actions, advantages, returns, old_log_probs):
        # Ensure inputs are float32
        states = tf.cast(states, tf.float32)
        actions = tf.cast(actions, tf.int32)
        advantages = tf.cast(advantages, tf.float32)
        returns = tf.cast(returns, tf.float32)
        old_log_probs = tf.cast(old_log_probs, tf.float32)

        with tf.GradientTape() as tape:
            # Get current policy
            logits = self.policy_network(states, training=True)  # Sample noise during training
            action_masks = tf.one_hot(actions, self.action_num)
            new_log_probs = tf.nn.log_softmax(logits)
            new_log_probs = tf.reduce_sum(new_log_probs * action_masks, axis=1)

            # Policy ratio
            ratio = tf.exp(new_log_probs - old_log_probs)

            # Clipped surrogate objective
            clipped_ratio = tf.clip_by_value(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
            surrogate = -tf.minimum(ratio * advantages, clipped_ratio * advantages)
            policy_loss = tf.reduce_mean(surrogate)

            # Value loss
            values = tf.squeeze(self.value_network(states, training=True), axis=1)  # Sample noise during training
            value_loss = tf.reduce_mean(tf.square(returns - values))

            # Entropy bonus
            entropy = -tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=action_masks, logits=logits))
            entropy_loss = -self.entropy_coef * entropy

            # Total loss
            total_loss = policy_loss + self.value_coef * value_loss + entropy_loss

        # Compute gradients and apply updates
        gradients = tape.gradient(total_loss, self.policy_network.trainable_variables + self.value_network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.policy_network.trainable_variables + self.value_network.trainable_variables))

    def save(self, filepath='ppo_noisy_model'):
        '''
        Save the PPO model.
        :param filepath: Path to save the model
        '''
        self.policy_network.save_weights(filepath + '_policy.weights.h5')
        self.value_network.save_weights(filepath + '_value.weights.h5')
    
    def load_policy(self, policy_path='ppo_noisy_model_policy.weights.h5'):
        """Load weights for the policy network."""
        self.policy_network.load_weights(policy_path)

    def load_value(self, value_path='ppo_noisy_model_value.weights.h5'):
        """Load weights for the value network."""
        self.value_network.load_weights(value_path)

    def load(self, filepath='ppo_noisy_model'):
        '''
        Load the PPO model.
        :param filepath: Path to load the model
        '''
        self.policy_network.load_weights(filepath + '_policy.weights.h5')
        self.value_network.load_weights(filepath + '_value.weights.h5')
