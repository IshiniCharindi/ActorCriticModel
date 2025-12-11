import pickle
from collections import deque
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl
import tensorflow.keras.losses as kls
import tensorflow.keras.optimizers as ko
import tensorflow.keras.backend as K
from tensorflow.keras.activations import relu
from tlcs.logger import get_logger
logger = get_logger(__name__)
lrelu = lambda x : relu(x, alpha =0.01)
class ProbabilityDistribution(tf.keras.Model):
    def call(self, logits):
        return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)
class Model2(tf.keras.Model):
    def __init__(self, num_actions):
        super().__init__(name='mlp_policy') # Fix: Use keyword 'name'
        self.num_actions = num_actions
        self.value1 = kl.Dense(48, activation='relu', name='value1')
        self.value2 = kl.Dense(24, activation='relu', name='value2')
        self.value = kl.Dense(1, name='value')
        self.logits1 = kl.Dense(48, activation='relu', name='policy_logits1')
        self.logits2 = kl.Dense(24, activation='relu', name='policy_logits2')
        self.logits = kl.Dense(num_actions, name='policy_logits')
        self.dist = ProbabilityDistribution()
    def build(self, input_shape):
        super().build(input_shape)
        self.logits1.build(input_shape)
        self.logits2.build((None, 48))
        self.logits.build((None, 24))
        self.value1.build(input_shape)
        self.value2.build((None, 48))
        self.value.build((None, 24))
        self.dist.build((None, self.num_actions))
    def call(self, inputs):
        x = tf.convert_to_tensor(inputs, dtype=tf.float32)
        hidden_logs = self.logits1(x)
        hidden_logs = self.logits2(hidden_logs)
        hidden_vals = self.value1(x)
        hidden_vals = self.value2(hidden_vals)
        return self.logits(hidden_logs), self.value(hidden_vals)
    def action_value(self, obs):
        logits, value = self.call(obs)
        action = self.dist(logits)
        return action.numpy().squeeze(), value.numpy().squeeze()
class ACAgent:
    def __init__(self, state_size, action_size, ID, n_step_size, gamma, alpha, entropy, value):
        self.type = 'AC'
        self.params = {'value': value, 'entropy': entropy, 'gamma': gamma, 'learning_rate' : alpha}
        self.model = Model2(action_size)
        self.model.compile(
            optimizer=ko.RMSprop(learning_rate = self.params['learning_rate']),
            loss=[self._logits_loss, self._value_loss]
        )
        self.model.build((None, state_size))
        self.trainstep = 0
        self.signal_id = ID
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=n_step_size)
        self.episode_memory = []
        self.episode_reward = []
        self.n_step_size = n_step_size
        self.loss = []
        self.losses = 0
        self._test_model()
    def _test_model(self):
        self.model.action_value(np.zeros((1, self.state_size)))
    def remember(self, state, action, reward, next_state, done):
        self.memory.append([state, action, reward, next_state, done])
        self.episode_memory.append([state, action, reward, next_state, done])
        self.episode_reward.append(reward)
    def reset(self):
        self.episode_memory = []
        self.episode_reward = []
    def choose_action(self, state):
        action, _ = self.model.action_value(state)
        return action
    def learn(self):
        if len(self.memory) < self.n_step_size:
            return
        Sample = np.array(self.memory, dtype=object)
        states = np.vstack(Sample[:,0])
        actions = np.array(Sample[:,1], dtype='int32')
        rewards = np.array(Sample[:,2], dtype=float)
        next_state = Sample[-1,3]
        _, values = self.model.action_value(states)
        values = np.squeeze(values)
        _, next_value = self.model.action_value(next_state)
        next_value = np.squeeze(next_value)
        returns, advs = self._returns_advantages(rewards, values, next_value)
        acts_and_advs = np.concatenate([actions[:, np.newaxis], advs[:, np.newaxis]], axis=-1)
        self.losses = self.model.train_on_batch(states, [acts_and_advs, returns])
        self.memory.clear()
    def _returns_advantages(self, rewards, values, next_value):
        returns = np.append(np.zeros_like(rewards), next_value)
        for t in reversed(range(len(rewards))):
            returns[t] = rewards[t] + self.params['gamma'] * returns[t+1]
        returns = returns[:-1]
        advantages = returns - values
        return returns, advantages
    def _value_loss(self, returns, value):
        return self.params['value'] * kls.MeanSquaredError()(returns, value)
    def _logits_loss(self, acts_and_advs, logits):
        actions, advantages = tf.split(acts_and_advs, 2, axis=-1)
        weighted_sparse_ce = kls.SparseCategoricalCrossentropy(from_logits=True)
        actions = tf.cast(actions, tf.int32)
        policy_loss = weighted_sparse_ce(actions, logits, sample_weight=advantages)
        entropy_loss = kls.categorical_crossentropy(tf.nn.softmax(logits), logits, from_logits=True)
        return policy_loss - self.params['entropy']*entropy_loss
    def save_agent(self, out_path, model_name, agent_type, Session_ID, episode):
        Weights_Filename = out_path / f"Agent{self.signal_id}.weights.h5"
        Optimizer_Filename = out_path / f"Agent{self.signal_id}_Optimizer.h5"
        logger.info(f'Saving weights for agent-{self.signal_id}')
        self.model.save_weights(Weights_Filename)
        symbolic_weights = self.model.optimizer.variables
        weight_values = K.batch_get_value(symbolic_weights)
        with open(Optimizer_Filename, 'wb') as f:
            pickle.dump(weight_values, f)
    def load_agent(self, model_path, model_name, agent_type, Session_ID, episode, best=False):
        Weights_Filename = model_path / f"Agent{self.signal_id}.weights.h5"
        Optimizer_Filename = model_path / f"Agent{self.signal_id}_Optimizer.h5"
        self._test_model()
        self.model.load_weights(Weights_Filename)
        with open(Optimizer_Filename, 'rb') as f:
            weight_values = pickle.load(f)
        self.model.optimizer.set_weights(weight_values)