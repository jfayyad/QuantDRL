#!/usr/bin/python

"""
Title: Deep Deterministic Policy Gradient (DDPG)
Author: [amifunny](https://github.com/amifunny)
"""
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import math
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
from KLD import *


################## Parameters ##################
# client = airsim.MultirotorClient()
# client.confirmConnection()

num_states = 18
num_actions = 9

upper_bound = 1
lower_bound = -1

actor_nodes_1 = 100
actor_nodes_2 = 50

critic_nodes_1 = 100
critic_nodes_2 = 50
critic_nodes_input_1 = 20
critic_nodes_input_2 = 40
critic_nodes_output = 40

# actor_nodes_1 = 64
# actor_nodes_2 = 64
#
# critic_nodes_1 = 64
# critic_nodes_2 = 64
# critic_nodes_input_1 = 16
# critic_nodes_input_2 = 32
# critic_nodes_output = 32

print("Max Value of Action ->  {}".format(upper_bound))
print("Min Value of Action ->  {}".format(lower_bound))

"""
To implement better exploration by the Actor network, we use noisy perturbations,
specifically an **Ornstein-Uhlenbeck process** for generating noise, as described in the paper.
It samples noise from a correlated normal distribution.
"""


class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.08, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
                self.x_prev
                + self.theta * (self.mean - self.x_prev) * self.dt
                + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)

"""
The `Buffer` class implements Experience Replay.
---
![Algorithm](https://i.imgur.com/mS6iGyJ.jpg)
---
**Critic loss** - Mean Squared Error of `y - Q(s, a)`
where `y` is the expected return as seen by the Target network,
and `Q(s, a)` is action value predicted by the Critic network. `y` is a moving target
that the critic model tries to achieve; we make this target
stable by updating the Target model slowly.
**Actor loss** - This is computed using the mean of the value given by the Critic network
for the actions taken by the Actor network. We seek to maximize this quantity.
Hence we update the Actor network so that it produces actions that get
the maximum predicted value as seen by the Critic, for a given state.
"""

class Buffer:
    def __init__(self, buffer_capacity=100000, batch_size=64):
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = np.squeeze(np.array(obs_tuple[1]))
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]
        # print 'record'
        # print self.state_buffer
        # print self.action_buffer
        # print np.squeeze(np.array(obs_tuple[1]))
        # print obs_tuple[2]
        # print obs_tuple[3]

        self.buffer_counter += 1

    # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
    @tf.function
    def update(
            self,
            state_batch,
            action_batch,
            reward_batch,
            next_state_batch,
    ):
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        with tf.GradientTape() as tape:
            target_actions = target_actor(next_state_batch, training=True)
            y = reward_batch + gamma * target_critic([next_state_batch, target_actions], training=True)
            critic_value = critic_model([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
        critic_optimizer.apply_gradients(
            zip(critic_grad, critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = actor_model(state_batch, training=True)
            critic_value = critic_model([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
        actor_optimizer.apply_gradients(
            zip(actor_grad, actor_model.trainable_variables)
        )

    # We compute the loss and update parameters
    def learn(self):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        self.update(state_batch, action_batch, reward_batch, next_state_batch)


# This update target parameters slowly
# Based on rate `tau`, which is much less than one.
@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))


"""
Here we define the Actor and Critic networks. These are basic Dense models
with `ReLU` activation.
Note: We need the initialization for last layer of the Actor to be between
`-0.003` and `0.003` as this prevents us from getting `1` or `-1` output values in
the initial stages, which would squash our gradients to zero,
as we use the `tanh` activation.
"""


def get_actor():
    # Initialize weights between -3e-3 and 3-e3
    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

    inputs = layers.Input(shape=(num_states))
    out = layers.Dense(actor_nodes_1, activation="relu")(inputs)
    out = layers.Dense(actor_nodes_2, activation="relu")(out)
    outputs = layers.Dense(num_actions, activation="tanh", kernel_initializer=last_init)(out)

    # Our upper bound is 2.0 for Pendulum.
    outputs = outputs * upper_bound
    model = tf.keras.Model(inputs, outputs)
    return model


def get_critic():
    # State as input
    state_input = layers.Input(shape=(num_states))
    state_out = layers.Dense(critic_nodes_input_1, activation="relu")(state_input)
    state_out = layers.Dense(critic_nodes_input_2, activation="relu")(state_out)

    # Action as input
    action_input = layers.Input(shape=(num_actions))
    action_out = layers.Dense(critic_nodes_output, activation="relu")(action_input)

    # Both are passed through seperate layer before concatenating
    concat = layers.Concatenate()([state_out, action_out])

    out = layers.Dense(critic_nodes_1, activation="relu")(concat)
    out = layers.Dense(critic_nodes_2, activation="relu")(out)
    outputs = layers.Dense(1)(out)

    # Outputs single value for give state-action
    model = tf.keras.Model([state_input, action_input], outputs)

    return model


"""
`policy()` returns an action sampled from our Actor network plus some noise for
exploration.
"""


def policy(state, noise_object):
    sampled_actions = tf.squeeze(actor_model(state))
    noise = noise_object()  # np.array([np.squeeze(np.array(noise_object())) , np.squeeze(np.array(noise_object()))])
    # print(np.array(sampled_actions))
    # print(noise)

    # Adding noise to action
    sampled_actions = sampled_actions.numpy() + noise
    # print(sampled_actions)

    # We make sure action is within bounds
    legal_action = np.clip(sampled_actions, lower_bound, upper_bound)

    return [np.squeeze(legal_action)]



############################# Training hyperparameters #############################

std_dev = 0.1  # 0.2
ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

actor_model = get_actor()
critic_model = get_critic()

target_actor = get_actor()
target_critic = get_critic()

# Making the weights equal initially
target_actor.set_weights(actor_model.get_weights())
target_critic.set_weights(critic_model.get_weights())

# Learning rate for actor-critic models
critic_lr = 0.002 # 0.001
actor_lr = 0.001  # 0.0002

critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

total_episodes = 500
# Discount factor for future rewards
gamma = 0.99
# Used to update target networks
tau = 0.001  # 0.005
buffer = Buffer(50000, 64)
num_itr = 4  # number of iterations for each episode
avg_N = 10  # number of values that used for averaging

# To store reward history of each episode
ep_reward_list = []
# To store average reward history of last few episodes
avg_reward_list = []
# To store [prev_state, action, reward, state] of each iteration
itr_data_prev_state = []
itr_data_action = []
itr_data_action_bit = []
itr_data_reward = []
itr_data_state = []


# def model_stats(model):
#     n_L = []
#     for name, module in model.named_modules():
#         if isinstance(module, nn.Conv2d or nn.Linear):
#             n_L.append(name)
#
#     return n_L


def getState(num_bits):
    # s_i = [ L_i, c_in/c_out, b_cf, a_i-1 ]   ;   b_cf: conv or fully connected
    s1 = [1,    1/20, 1, num_bits[1][0], num_bits[1][1], num_bits[1][2]]
    s2 = [2,   20/50, 1, num_bits[2][0], num_bits[2][1], num_bits[2][2]]
    s3 = [3, 500/800, 0, num_bits[3][0], num_bits[3][1], num_bits[3][2]]
    # s4 = [4, 10/500, 0, num_bits[1][0]]
    # s_norm = [0, ]
    state = np.concatenate([s1, s2, s3])
    return state


def getReward(bitwidth_):
    ####################### Accuracy reward #######################
    # acc_quantize = acc_quant(q_model, bitwidth_)
    # error = acc_original - acc_quantize
    acc, R = acc_kl(model, q_model, bitwidth)
    error = acc[0] - acc[1]

    if error <= 0.0002:
        reward_acc = 9
    elif error >= 0.015:  # error greater than 1%
        reward_acc = -1 + ((-20+1)/(1-0.015))*(error-0.015)
    else:
        reward_acc = - math.log(error)

    ###################### bitwidth rewards #######################
    # Weights & Biases
    c1 = model.conv1.out_channels
    c2 = model.conv2.out_channels
    ct = c1 + c2
    f1 = model.fc1.out_features
    f2 = model.fc2.out_features
    ft = f1 + f2
    reward_bit_bias = 8 - ((c1 * bitwidth_[1][1] + c2 * bitwidth_[2][1] + f1 * bitwidth_[3][1]) / (c1 + c2 + f1))
    reward_bit_weight = (c1 / ct) * (8.0 - bitwidth_[1][0]) + (c2 / ct) * (8.0 - bitwidth_[2][0]) + (f1 / ft) * (8.0 - bitwidth_[3][0])

    # Activations
    k = 5
    act_conv1 = (24 * 24) * 20
    act_conv2 = (8 * 8) * 50
    act_fc1 = 500
    reward_bit_act = 8 - ((act_conv1 * bitwidth_[1][2] + act_conv2 * bitwidth_[2][2] + act_fc1 * bitwidth_[3][2]) / (act_conv1 + act_conv2 + act_fc1))

    # Average Weight of the Network
    parW_conv1 = (k * k * 1) * 20
    parW_conv2 = (k * k * 20) * 50
    parW_fc1 = (4 * 4) * 50 * 500
    avg_bit_para = (parW_conv1 * bitwidth_[1][0] + parW_conv2 * bitwidth_[2][0] + parW_fc1 * bitwidth_[3][0] +
                      c1 * bitwidth_[1][1] + c2 * bitwidth_[2][1] + f1 * bitwidth_[3][1]) / (c1 + c2 + f1 + parW_conv1 + parW_conv2 + parW_fc1)
    print('  parameters average bitwidth = ', avg_bit_para)
    reward_bit_weight_total = (8 - avg_bit_para)

    ####################### KL-D reward #######################
    reward_kl = - np.log10(np.sum(R))

    a_acc = 1.2
    a_bias = 0.8
    a_act = 1.0
    a_weight = 0.5
    a_avg_para = 1.5
    a_kl = 1.0

    return (a_acc * reward_acc), (a_weight * reward_bit_weight), (a_avg_para * reward_bit_weight_total), (a_bias * reward_bit_bias), (a_act * reward_bit_act), (a_kl * reward_kl)


################################################### Training ###################################################

bitwidth = [8, [2, 2, 2], [2, 2, 2], [2, 2, 2]]
# acc_original = testQuant(q_model, test_loader, bitwidth, quant=False)
# acc_original = acc_org(model)

for ep in range(1, total_episodes + 1):
    # Reset the setpoint for the new episode
    prev_state = getState(bitwidth)
    episodic_reward = 0

    for i in range(1, num_itr + 1):
        # t = time.time()     # tic
        print('<<<<<<<<<<<<<<<<<<<<<<<<<<<< Episode # {}-{} >>>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(ep, i))
        tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
        action = policy(tf_prev_state, ou_noise)
        action_bits = np.floor((7 * np.array(action) + 9)/2)
        # print('> previous_state  = {}'.format(prev_state))
        # print('  selected_action = {}'.format(action))
        # print("  action_value    = {}".format(action_bits))
        # print("---- apply action ----")

        # Recieve the new state and reward from environment:    state, reward, done, info = env.step(action)
        action_value = np.squeeze(action_bits)
        action_v = action_value.astype(int)
        print("  action_value = [{}, {}, {}], [{}, {}, {}], [{}, {}, {}]".format(
                        action_v[0], action_v[1], action_v[2],
                        action_v[3], action_v[4], action_v[5],
                        action_v[6], action_v[7], action_v[8]))

        # action_value = getattr(action_v, "tolist", lambda: action_v)()
        bitwidth = [8, [action_value[0], action_value[1], action_value[2]],
                       [action_value[3], action_value[4], action_value[5]],
                       [action_value[6], action_value[7], action_value[8]]]
        r_acc, r_w, r_bp, r_b, r_a, r_kl = getReward(bitwidth)
        reward = r_acc + r_w + r_bp + r_b + r_a
        state = getState(bitwidth)
        # print('  new_state  = {}'.format(state))
        print('  rewards: [accuracy, weight, avg_bit_para, bias, activation, kl-d] =\n           [{}, {}, {}, {}, {}, {}] \n  total_reward = {}'.format(r_acc, r_w, r_bp, r_b, r_a, r_kl, reward))

        buffer.record((prev_state, action, reward, state))

        itr_data_prev_state.append(prev_state)
        itr_data_action.append(np.squeeze(np.array(action)))
        itr_data_action_bit.append(action_value)
        itr_data_reward.append(reward)
        itr_data_state.append(state)
        episodic_reward += reward

        buffer.learn()
        update_target(target_actor.variables, actor_model.variables, tau)
        update_target(target_critic.variables, critic_model.variables, tau)
        prev_state = state

        # elapsed = time.time() - t   # tac
        # print('  elapsed time per iteration = ', elapsed)

    # print('  eps_counter     = {}'.format(counter))
    ep_reward_list.append(episodic_reward)
    print('########################################################################')
    print('           << Episode # {} is over! Updtae Actor-Critic ... >>           '.format(ep))
    print("> episode_reward = {}".format(episodic_reward))
    # Mean of last 40 episodes
    avg_reward = np.mean(ep_reward_list[-avg_N:])
    print("  avg reward is ==> {}".format(avg_reward))
    avg_reward_list.append(avg_reward)

# Save the weights
actor_model.save_weights("./Results/pendulum_actor.h5")
critic_model.save_weights("./Results/pendulum_critic.h5")
target_actor.save_weights("./Results/pendulum_target_actor.h5")
target_critic.save_weights("./Results/pendulum_target_critic.h5")

np.savetxt('./Results/avg_reward_list.csv', avg_reward_list)
np.savetxt('./Results/ep_reward_list.csv', ep_reward_list)
np.savetxt('./Results/itr_data_prev_state.csv', itr_data_prev_state)
np.savetxt('./Results/itr_data_action.csv', itr_data_action)
np.savetxt('./Results/itr_data_action_bit.csv', itr_data_action_bit)
np.savetxt('./Results/itr_data_reward.csv', itr_data_reward)
np.savetxt('./Results/itr_data_state.csv', itr_data_state)

# Plotting graph
# Episodes versus Avg. Rewards
plt.plot(avg_reward_list)
plt.xlabel("Episode")
plt.ylabel("Avg. Epsiodic Reward")
plt.show()