#!/usr/bin/python

"""
Title: Deep Deterministic Policy Gradient (DDPG)
Author: [amifunny](https://github.com/amifunny)
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# import setup_path
# import airsim
import time
import random
import math
import sys
import numpy as np
# import rospy
# from visualization_msgs.msg import Marker
# from nav_msgs.msg import Odometry
import tensorflow as tf
from tensorflow import keras
# from tensorflow.keras import layers
from keras import layers
# from std_msgs.msg import Float32
# from std_msgs.msg import Float32MultiArray
# from geometry_msgs.msg import PoseStamped

import matplotlib.pyplot as plt

from main import *


################## Parameters ##################
# client = airsim.MultirotorClient()
# client.confirmConnection()

num_states = 10
num_actions = 2

upper_bound = 1
lower_bound = -1

actor_nodes_1 = 100
actor_nodes_2 = 50

critic_nodes_1 = 100
critic_nodes_2 = 50
critic_nodes_input_1 = 20
critic_nodes_input_2 = 40
critic_nodes_output = 40

# actor_nodes_1 = 128
# actor_nodes_2 = 128

# critic_nodes_1 = 128
# critic_nodes_2 = 128
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
    print(np.array(sampled_actions))
    print(noise)

    # Adding noise to action
    sampled_actions = sampled_actions.numpy() + noise
    print(sampled_actions)

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
critic_lr = 0.001  # 0.002
actor_lr = 0.0002  # 0.001

critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

total_episodes = 100

# Discount factor for future rewards
gamma = 0.99
# Used to update target networks
tau = 0.001  # 0.005

buffer = Buffer(50000, 64)

num_itr = 5  # number of iterations for each episode

avg_N = 30  # number of values that used for averaging


# # sweeping (pre-exploration):
# action_sweep = False  # if set True, then adjust the parameters (below) accordingly
# sweeping_itr = 4
# action_values = []
# action_min = 30
# action_max = 170
#
# for i in range(sweeping_itr):
#     value_i = action_min + (i * (action_max - action_min) / (sweeping_itr - 1))
#     for j in range(sweeping_itr):
#         value_j = action_min + (j * (action_max - action_min) / (sweeping_itr - 1))
#         action_values.append([value_i, value_j])
#
# sweeping_epi = len(action_values)

"""
Now we implement our main training loop, and iterate over episodes.
We sample actions using `policy()` and train with `learn()` at each time step,
along with updating the Target networks at a rate `tau`.
"""

# To store reward history of each episode
ep_reward_list = []
# To store average reward history of last few episodes
avg_reward_list = []
# To store [prev_state, action, reward, state] of each iteration
itr_data_prev_state = []
itr_data_action = []
itr_data_reward = []
itr_data_state = []

# # Load the previous model and continue training ...
# load_model = False
#
# if load_model == True:
#     actor_model.load_weights("pendulum_actor.h5")
#     critic_model.load_weights("pendulum_critic.h5")
#     target_actor.load_weights("pendulum_target_actor.h5")
#     target_critic.load_weights("pendulum_target_critic.h5")
#     prevSet_itr_data_prev_state = np.loadtxt("itr_data_prev_state.csv", dtype=float)
#     prevSet_itr_data_action = np.loadtxt("itr_data_action.csv", dtype=float)
#     prevSet_itr_data_reward = np.loadtxt("itr_data_reward.csv", dtype=float)
#     prevSet_itr_data_state = np.loadtxt("itr_data_state.csv", dtype=float)
#     prevSet_avg_reward_list = np.loadtxt('avg_reward_list.csv')
#     prevSet_ep_reward_list = np.loadtxt('ep_reward_list.csv')
#     for i in range(len(prevSet_itr_data_reward)):
#         buffer.record((prevSet_itr_data_prev_state[i], prevSet_itr_data_action[i], prevSet_itr_data_reward[i],
#                        prevSet_itr_data_state[i]))
#         itr_data_prev_state.append(prevSet_itr_data_prev_state[i])
#         itr_data_action.append(prevSet_itr_data_action[i])
#         itr_data_reward.append(prevSet_itr_data_reward[i])
#         itr_data_state.append(prevSet_itr_data_state[i])
#     for i in range(len(prevSet_ep_reward_list)):
#         ep_reward_list.append(prevSet_ep_reward_list[i])
#         avg_reward_list.append(prevSet_avg_reward_list[i])
#     print('previous nn weights are loaded .... ')

# #############################################  MAIN CODE #############################################
# ## Start ROS -------------------------------
# rospy.init_node('a2c_node', anonymous=False)
# goal_publisher = rospy.Publisher("/firefly/command/pose", PoseStamped, queue_size=1)
# goal = PoseStamped()
# goal.header.seq = 1
# goal.header.stamp = rospy.Time.now()
# goal.header.frame_id = "world"
# goal.pose.orientation.x = 0.0
# goal.pose.orientation.y = 0.0
# goal.pose.orientation.z = 0.0
# goal.pose.orientation.w = 1.0
# rospy.sleep(1)
#
# # Sweeping
# if (load_model == False) and (action_sweep == True):
#     for ep in range(1, sweeping_epi + 1):
#         # Reset the setpoint for the new episode
#         goal.pose.position.x = 5.0
#         goal.pose.position.y = 5.0
#         goal.pose.position.z = 6.0
#         resetDrone(altitude=6, velMinMax=2.0, duration=1.0)
#         goal_publisher.publish(goal)
#         updateSetpoint(goal)
#
#         prev_state = getState()
#
#         episodic_reward = 0
#
#         for i in range(1, num_itr + 1):
#             print('<<<<<<<<<<<<<<<<<<<<<<<<<<<< Sweeping # {}-{} >>>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(ep, i))
#             goal_publisher.publish(goal)
#             updateSetpoint(goal)
#
#             action = np.array(action_values[ep - 1], dtype=float)
#
#             print('> previous_state  = {}'.format(prev_state))
#             print("  action_value    = {}".format(action))
#             print("---- apply action ----")
#
#             action = action / 200
#
#             W_ddpg = Float32MultiArray()
#             W_ddpg.data = [action[0], action[1]]
#
#             # Publish the action to nmph:
#             pub = rospy.Publisher('/a2c_action', Float32MultiArray, queue_size=1)
#             pub.publish(W_ddpg)
#             rospy.sleep(0.05)
#
#             # Recieve the new state and reward from environment:    state, reward, done, info = env.step(action)
#             # nmph will predict the trajectory .... now wait for the next prediction (below):
#             path = rospy.wait_for_message("/firefly/predicted_state", Marker)
#             sub = rospy.Subscriber("/airsim_node_planner/drone_1/odom_local_ned_drop", Odometry, odometryCb)
#             reward_rms, reward_sp, reward_des = pathFollower(path)
#             reward = reward_rms + reward_sp + reward_des
#             state = getState()
#             print('  new_state = {}'.format(state))
#             print('  reward_rms = {} ,  reward_sp = {} ,  reward_des = {}'.format(reward_rms, reward_sp, reward_des))
#             print('  reward    = {}'.format(reward))
#
#             buffer.record((prev_state, action, reward, state))
#             episodic_reward += reward
#
#             buffer.learn()
#             update_target(target_actor.variables, actor_model.variables, tau)
#             update_target(target_critic.variables, critic_model.variables, tau)
#
#             prev_state = state
#
#             # Update the setpoint
#             goal.pose.position.x += 4.0
#             if (i % 2) == 0:
#                 goal.pose.position.y += 4.0
#             else:
#                 goal.pose.position.y -= 4.0
#             rospy.sleep(0.05)
#             sub.unregister()
#
#         print('########################################################################')
#         print('                   << Sweeping # {} is over! ...  >>                    '.format(ep))
#         print("> episode_reward = {}".format(episodic_reward))
#


# def model_stats(model):
#     n_L = []
#     for name, module in model.named_modules():
#         if isinstance(module, nn.Conv2d or nn.Linear):
#             n_L.append(name)
#
#     return n_L


def getState(num_bits):
    # s_i = [ L_i, c_in, c_out, b_cf, a_i-1 ]   ;   b_cf: conv or fully connected
    s1 = [1, 1/50, 20/50, 1, num_bits[1][0]]
    s2 = [2, 20/50, 50/50, 1, num_bits[2][0]]
    # s3 = [3, 800, 500, 0, num_bits[3][0]]
    # s4 = [4, 500, 10, 0, num_bits[1][0]]
    # s_norm = [0, ]
    state = np.concatenate([s1, s2])
    return state


def getReward(bitwidth_):
    acc = testQuant(q_model, test_loader, bitwidth_, quant=True, stats=stats)
    error = acc_original - acc

    if error <= 0.0001:
        reward_acc = 10
    elif error >= 0.05:
        reward_acc = -5
    else:
        reward_acc = - math.log(error)

    reward_bits = (8.0 - bitwidth_[1][0]) + (8.0 - bitwidth_[2][0])

    return reward_acc, reward_bits


# # n_L, c_in, c_out, b_cf = model_stats(q_model)
# state = getState(bitwidth)
# tf_prev_state = tf.expand_dims(tf.convert_to_tensor(state), 0)
# print(state)

acc_original = testQuant(q_model, test_loader, bitwidth, quant=False)

# Training
for ep in range(1, total_episodes + 1):
    # Reset the setpoint for the new episode
    bitwidth = [8, [1, 8, 8], [1, 8, 8], [8, 8, 8]]
    prev_state = getState(bitwidth)

    episodic_reward = 0

    for i in range(1, num_itr + 1):
        print('<<<<<<<<<<<<<<<<<<<<<<<<<<<< Episode # {}-{} >>>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(ep, i))
        tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
        action = policy(tf_prev_state, ou_noise)
        action_bits = np.floor((7 * np.array(action) + 9)/2)
        print('> previous_state  = {}'.format(prev_state))
        print('  selected_action = {}'.format(action))
        print("  action_value    = {}".format(action_bits))
        print("---- apply action ----")

        # Recieve the new state and reward from environment:    state, reward, done, info = env.step(action)
        action_value = np.squeeze(action_bits)
        # action_value = getattr(action_v, "tolist", lambda: action_v)()
        bitwidth = [8, [action_value[0], 8, 8], [action_value[1], 8, 8], [8, 8, 8]]
        reward_acc, reward_bits = getReward(bitwidth)
        reward = reward_acc + reward_bits
        state = getState(bitwidth)
        print('  new_state  = {}'.format(state))
        print('  reward_acc = {} ,  reward_bits = {} ,  total_reward = {}'.format(reward_acc, reward_bits, reward))
        # print('  reward    = {}'.format(reward))

        buffer.record((prev_state, action, reward, state))

        itr_data_prev_state.append(prev_state)
        itr_data_action.append(np.squeeze(np.array(action)))
        itr_data_reward.append(reward)
        itr_data_state.append(state)
        episodic_reward += reward

        buffer.learn()
        update_target(target_actor.variables, actor_model.variables, tau)
        update_target(target_critic.variables, critic_model.variables, tau)

        prev_state = state


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
actor_model.save_weights("pendulum_actor.h5")
critic_model.save_weights("pendulum_critic.h5")
target_actor.save_weights("pendulum_target_actor.h5")
target_critic.save_weights("pendulum_target_critic.h5")

np.savetxt('avg_reward_list.csv', avg_reward_list)
np.savetxt('ep_reward_list.csv', ep_reward_list)
np.savetxt('itr_data_prev_state.csv', itr_data_prev_state)
np.savetxt('itr_data_action.csv', itr_data_action)
np.savetxt('itr_data_reward.csv', itr_data_reward)
np.savetxt('itr_data_state.csv', itr_data_state)

# Plotting graph
# Episodes versus Avg. Rewards
plt.plot(avg_reward_list)
plt.xlabel("Episode")
plt.ylabel("Avg. Epsiodic Reward")
plt.show()