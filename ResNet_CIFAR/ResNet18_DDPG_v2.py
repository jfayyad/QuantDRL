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
from ResNet18_CIFAR import *
from collections.abc import Iterable
import wandb

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

w_and_b = True

if w_and_b:
    wandb.init(project = "ResNet18_DDPG_V2.1",
               config={ "actor_nodes_1": 400,
                        "actor_nodes_2": 200,
                        "critic_nodes_1": 400,
                        "critic_nodes_2 ": 200,
                        "critic_nodes_input_1": 150,
                        "critic_nodes_input_2": 200,
                        "critic_nodes_output": 200,
                        "critic_lr": 0.001,
                        "actor_lr": 0.0005,
                        "total_episodes": 4000,
                        "num_itr": 1,
                        "buffer_batch_size": 64
               }
               )



################## Parameters ##################
# client = airsim.MultirotorClient()
# client.confirmConnection()

num_states = 135
num_actions = 62

upper_bound = 1
lower_bound = -1

actor_nodes_1 = 400
actor_nodes_2 = 200

critic_nodes_1 = 400
critic_nodes_2 = 200
critic_nodes_input_1 = 150
critic_nodes_input_2 = 200
critic_nodes_output = 200

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
critic_lr = 0.001 # 0.001
actor_lr = 0.0005  # 0.0002

critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

total_episodes = 4000
# Discount factor for future rewards
gamma = 0.99
# Used to update target networks
tau = 0.001  # 0.005
buffer = Buffer(50000, 64)
num_itr = 1  # number of iterations for each episode
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

def flatten(lis):
    for item in lis:
        if isinstance(item, Iterable) and not isinstance(item, str):
            for x in flatten(item):
                yield x
        else:
            yield item


def getState(num_bits):
    # s_i = [ L_i, c_in/c_out, b_cf, a_i-1 ]   ;   b_cf: conv or fully connected
    s1 = list(flatten([1, 1, num_bits[1]]))
    s2 = list(flatten([2, 1, num_bits[2]]))
    s3 = list(flatten([3, 1, num_bits[3]]))
    s4 = list(flatten([4, 1, num_bits[4]]))
    s5 = list(flatten([5, 1, num_bits[5]]))
    s6 = list(flatten([6, 0, num_bits[6]]))
    state = np.concatenate([s1, s2, s3, s4, s5, s6])
    return state


def getReward(bitwidth_):
    acc, R = acc_kl(bitwidth_)

    a_acc = 1.3
    a_act = 1.0
    a_weight = 1.2
    a_kl = 1.0
    # a_avg_para = 1.5
    # a_bias = 0.8
    a_lowbit = 0.0

    # temp = bitwidth_.copy()
    # temp.remove(temp[0])
    a = list(flatten(bitwidth_))

    ####################### low bitwidth reward #######################
    reward_lowBit = 0
    # for i in range(len(a)):                     # low bitwidth reward
    #     if a[i] > 1: reward_lowBit += (8 - a[i]) / 60
    ####################### Accuracy reward #######################
    error = acc[0] - acc[1]
    if w_and_b: wandb.log({"acc_quant": acc[1]})

    if error <= 0.03:
        reward_acc = 8
        # for i in range(len(a)):                     # low bitwidth reward
        #     reward_lowBit += (8 - a[i]) / 20
    elif error >= 0.1:  # error greater than 1%
        reward_acc = ((-20)/(1-0.1))*(error-0.1)
    else:
        reward_acc = - math.log(error)

    ####################### Average Weight and act of the Network #######################
    b0_par = (a[1]*9472) #9600
    b1_par = (a[7] * 37056 + a[13] * 37056 + a[19] * 37056 + a[25] * 37056) # 74112*2
    b2_par = (a[31] * 74112 + a[49] * 147840 + a[37] * 147840 + a[55] * 147840 + a[43] * 8576) #526208
    b3_par = (a[61] * 295680 + a[79] * 590592 + a[67] * 590592 + a[85] * 590592 + a[73] * 33536) #2100992
    b4_par = (a[91] * 1181184 + a[109] * 2360832 + a[97] * 2360832 + a[115] * 2360832 + a[103] * 132608) #8396288
    fc_par = a[121]*5130 #5130

    avg_par = (b0_par + b1_par + b2_par + b3_par + b4_par + fc_par) / 11186442
    if w_and_b: wandb.log({"avg_par": avg_par})
    print('  avg_par = {}'.format(avg_par))
    reward_bit_weight = (8 - avg_par)

    b0_act = (a[3]*(64*112*112) + a[6]*(64*112*112))        # 2*(64*112*112)
    b1_act = (a[9] + a[12] + a[15] + a[18])*(64*56*56) + (a[21] + a[24] + a[27] + a[30])*(64*56*56)  # 8*(64*56*56)
    b2_act = (a[33] + a[36] + a[39] + a[42])*(128*28*28) + (a[51] + a[54] + a[57] + a[60])*(128*28*28) + (a[45] + a[48])*(128*28*28)  # 10*(128*28*28)
    b3_act = (a[63] + a[66] + a[69] + a[72])*(256*14*14) + (a[81] + a[84] + a[87] + a[90])*(256*14*14) + (a[75] + a[78])*(256*14*14)  # 10*(256*14*14)
    b4_act = (a[93] + a[96] + a[99] + a[102])*(512*7*7) + (a[111] + a[114] + a[117] + a[120])*(512*7*7) + (a[105] + a[108])*(512*7*7)      # 10*(512*7*7)
    fc_act = a[123]*512*10  # 5120

    avg_act = (b0_act + b1_act + b2_act + b3_act + b4_act + fc_act) / 4972544
    if w_and_b: wandb.log({"avg_act": avg_act})
    print('  avg_act = {}'.format(avg_act))
    reward_bit_act = (8 - avg_act)

    # ####################### KL-D reward #######################
    reward_kl = - np.log10(np.sum(R))

    return (a_acc * reward_acc), (a_weight * reward_bit_weight), (a_act * reward_bit_act), (a_kl * reward_kl), (a_lowbit * reward_lowBit)


################################################### Training ###################################################

bitwidth = [8,                                                                                                                                   # bitwidth[0] = input (act)
            [[8, 1], 8, [8, 8], 8],                                                                                                               # bitwidth[1] = conv1_bn1: [ conv1[w, b], act, bn[w, b], act]
            [[[[8, 1], 8, [8, 8], 8] , [[8, 1], 8, [8, 8], 8]],                            [[[8, 1], 8, [8, 8], 8] , [[8, 1], 8, [8, 8], 8] ]],   # bitwidth[2] = block0{conv1_conv2_downsample} x2
            [[[[8, 1], 8, [8, 8], 8] , [[8, 1], 8, [8, 8], 8] , [[8, 1], 8, [8, 8], 8]],   [[[8, 1], 8, [8, 8], 8] , [[8, 1], 8, [8, 8], 8] ]],   # bitwidth[3] = block1{conv1_conv2_downsample} x2
            [[[[8, 1], 8, [8, 8], 8] , [[8, 1], 8, [8, 8], 8] , [[8, 1], 8, [8, 8], 8]],   [[[8, 1], 8, [8, 8], 8] , [[8, 1], 8, [8, 8], 8] ]],   # bitwidth[4] = block2{conv1_conv2_downsample} x2
            [[[[8, 1], 8, [8, 8], 8] , [[8, 1], 8, [8, 8], 8] , [[8, 1], 8, [8, 8], 8]],   [[[8, 1], 8, [8, 8], 8] , [[8, 1], 8, [8, 8], 8] ]],   # bitwidth[5] = block3{conv1_conv2_downsample} x2
            [[8, 1], 8]                                                                                                                           # bitwidth[6] = fc:[[w, b], act]
            ]

# acc, R = acc_kl(bitwidth)

for ep in range(1, total_episodes + 1):
    # Reset the setpoint for the new episode
    prev_state = getState(bitwidth)
    episodic_reward = 0

    for i in range(1, num_itr + 1):
        # t = time.time()     # tic
        print('<<<<<<<<<<<<<<<<<<<<<<<<<<<< Episode # {}-{} >>>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(ep, i))
        tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
        action = policy(tf_prev_state, ou_noise)
        action_bits = np.floor((3 * np.array(action) + 5))

        a = np.squeeze(action_bits)

        bitwidth  = [8,
                    [[a[0], 1], a[1], [8, 8], a[2]],
                    [[[[a[3], 1], a[4], [8, 8], a[5]] , [[a[6], 1], a[7], [8, 8], a[8]]],                                               [[[a[9], 1], a[10], [8, 8], a[11]] , [[a[12], 1], a[13], [8, 8], a[14]] ]],
                    [[[[a[15], 1], a[16], [8, 8], a[17]] , [[a[18], 1], a[19], [8, 8], a[20]] , [[a[21], 1], a[22], [8, 8], a[23]]],    [[[a[24], 1], a[25], [8, 8], a[26]] , [[a[27], 1], a[28], [8, 8], a[29]] ]],
                    [[[[a[30], 1], a[31], [8, 8], a[32]] , [[a[33], 1], a[34], [8, 8], a[35]] , [[a[36], 1], a[37], [8, 8], a[38]]],    [[[a[39], 1], a[40], [8, 8], a[41]] , [[a[42], 1], a[43], [8, 8], a[44]] ]],
                    [[[[a[45], 1], a[46], [8, 8], a[47]] , [[a[48], 1], a[49], [8, 8], a[50]] , [[a[51], 1], a[52], [8, 8], a[53]]],    [[[a[54], 1], a[55], [8, 8], a[56]] , [[a[57], 1], a[58], [8, 8], a[59]] ]],
                    [[a[60], 1], a[61]]
                    ]

        ap = np.array(list(flatten(bitwidth)))
        action_v = ap.astype(int)
        print('  action  = {}'.format(action_v))

        r_acc, r_w, r_a, r_kl, r_lb = getReward(bitwidth)
        reward = r_acc + r_w + r_a + r_kl + r_lb

        state = getState(bitwidth)
        # print('  reward  = {}'.format(reward))
        # print('  rewards: [accuracy, weight, activation, kl-d] =\n           [{}, {}, {}, {}] \n  total_reward = {}'.format(r_acc, r_w, r_a, r_kl, reward))
        print('  rewards: [accuracy, weight, activation, kl-d, low_bit] =\n           [{}, {}, {}, {}, {}] \n  total_reward = {}'.format(r_acc, r_w, r_a, r_kl, r_lb, reward))

        buffer.record((prev_state, action, reward, state))

        itr_data_prev_state.append(prev_state)
        itr_data_action.append(np.squeeze(np.array(action)))
        itr_data_action_bit.append(a)
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
    if w_and_b: wandb.log({"Reward": episodic_reward, "episod": ep})
    print('########################################################################')
    print('           << Episode # {} is over! Updtae Actor-Critic ... >>           '.format(ep))
    print("> episode_reward = {}".format(episodic_reward))
    # Mean of last 40 episodes
    avg_reward = np.mean(ep_reward_list[-avg_N:])
    print("  avg reward is ==> {}".format(avg_reward))
    avg_reward_list.append(avg_reward)

# # Save the weights
# actor_model.save_weights("./Results/pendulum_actor.h5")
# critic_model.save_weights("./Results/pendulum_critic.h5")
# target_actor.save_weights("./Results/pendulum_target_actor.h5")
# target_critic.save_weights("./Results/pendulum_target_critic.h5")

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

if w_and_b: wandb.finish()
