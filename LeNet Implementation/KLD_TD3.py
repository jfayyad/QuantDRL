#!/usr/bin/python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import random
import time
import argparse
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tensorlayer as tl
from tensorlayer.layers import Dense
from tensorlayer.models import Model
import numpy as np
from KLD import *

Normal = tfp.distributions.Normal
tl.logging.set_verbosity(tl.logging.DEBUG)

# add arguments in command  --train/test
parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=True)
parser.add_argument('--test', dest='test', action='store_true', default=False)
args = parser.parse_args()

#####################  hyper parameters  ####################
# choose env
ENV_ID = 'TD3_Quant'  # environment id
RANDOM_SEED = 2  # random seed
RENDER = False  # render while training

# RL training
ALG_NAME = 'TD3'
TRAIN_EPISODES = 300    # total number of episodes for training
TEST_EPISODES = 10      # total number of episodes for training
MAX_STEPS = 200         # maximum number of steps for one episode
BATCH_SIZE = 64         # update batch size
EXPLORE_STEPS = 500     # 500 for random action sampling in the beginning of training

state_dim = 18          # number of states
action_dim = 9          # number of actions
action_range = 1        # scale action, [-action_range, action_range]
num_itr = 10            # number of iterations on each episode
avg_N = 20              # number of values that used for averaging

HIDDEN_DIM = 100        # size of hidden layers for networks
UPDATE_ITR = 3          # repeated updates for single step
Q_LR = 3e-4             # q_net learning rate
POLICY_LR = 3e-4        # policy_net learning rate
POLICY_TARGET_UPDATE_INTERVAL = 3   # delayed steps for updating the policy network and target networks
EXPLORE_NOISE_SCALE = 1.0           # range of action noise for exploration
EVAL_NOISE_SCALE = 0.5              # range of action noise for evaluation of action value
REWARD_SCALE = 1.                   # value range of reward
REPLAY_BUFFER_SIZE = 5e5            # size of replay buffer

###############################  TD3  ####################################

class ReplayBuffer:
    """
    a ring buffer for storing transitions and sampling for training
    :state: (state_dim,)
    :action: (action_dim,)
    :reward: (,), scalar
    :next_state: (state_dim,)
    :done: (,), scalar (0 and 1) or bool (True and False)
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = int((self.position + 1) % self.capacity)  # as a ring buffer

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))  # stack for each element
        """ 
        the * serves as unpack: sum(a,b) <=> batch=(a,b), sum(*batch) ;
        zip: a=[1,2], b=[2,3], zip(a,b) => [(1, 2), (2, 3)] ;
        the map serves as mapping the function on each list element: map(square, [2,3]) => [4,9] ;
        np.stack((1,2)) => array([1, 2])
        """
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class QNetwork(Model):
    """ the network for evaluate values of state-action pairs: Q(s,a) """

    def __init__(self, num_inputs, num_actions, hidden_dim, init_w=3e-3):
        super(QNetwork, self).__init__()
        input_dim = num_inputs + num_actions
        # w_init = tf.keras.initializers.glorot_normal(seed=None)
        w_init = tf.random_uniform_initializer(-init_w, init_w)

        self.linear1 = Dense(n_units=hidden_dim, act=tf.nn.relu, W_init=w_init, in_channels=input_dim, name='q1')
        self.linear2 = Dense(n_units=hidden_dim, act=tf.nn.relu, W_init=w_init, in_channels=hidden_dim, name='q2')
        self.linear3 = Dense(n_units=1, W_init=w_init, in_channels=hidden_dim, name='q3')

    def forward(self, input):
        x = self.linear1(input)
        x = self.linear2(x)
        x = self.linear3(x)
        return x


class PolicyNetwork(Model):
    """ the network for generating non-deterministic (Gaussian distributed) action from the state input """

    def __init__(self, num_inputs, num_actions, hidden_dim, action_range=1., init_w=3e-3):
        super(PolicyNetwork, self).__init__()
        w_init = tf.random_uniform_initializer(-init_w, init_w)

        self.linear1 = Dense(n_units=hidden_dim, act=tf.nn.relu, W_init=w_init, in_channels=num_inputs, name='policy1')
        self.linear2 = Dense(n_units=hidden_dim, act=tf.nn.relu, W_init=w_init, in_channels=hidden_dim, name='policy2')
        self.linear3 = Dense(n_units=hidden_dim, act=tf.nn.relu, W_init=w_init, in_channels=hidden_dim, name='policy3')
        self.output_linear = Dense(
            n_units=num_actions, W_init=w_init, b_init=tf.random_uniform_initializer(-init_w, init_w),
            in_channels=hidden_dim, name='policy_output'
        )
        self.action_range = action_range
        self.num_actions = num_actions

    def forward(self, state):
        x = self.linear1(state)
        x = self.linear2(x)
        x = self.linear3(x)
        output = tf.nn.tanh(self.output_linear(x))  # unit range output [-1, 1]
        return output

    def evaluate(self, state, eval_noise_scale):
        """
        generate action with state for calculating gradients;
        eval_noise_scale: as the trick of target policy smoothing, for generating noisy actions.
        """
        state = state.astype(np.float32)
        action = self.forward(state)

        action = self.action_range * action

        # add noise
        normal = Normal(0, 1)
        eval_noise_clip = 2 * eval_noise_scale
        noise = normal.sample(action.shape) * eval_noise_scale
        noise = tf.clip_by_value(noise, -eval_noise_clip, eval_noise_clip)
        action = action + noise
        return action

    def get_action(self, state, explore_noise_scale, greedy=False):
        """ generate action with state for interaction with envronment """
        action = self.forward([state])
        action = self.action_range * action.numpy()[0]
        if greedy:
            return action
        # add noise
        normal = Normal(0, 1)
        noise = normal.sample(action.shape) * explore_noise_scale
        action += noise
        return action.numpy()

    def sample_action(self):
        """ generate random actions for exploration """
        a = tf.random.uniform([self.num_actions], -1, 1)
        return self.action_range * a.numpy()


class TD3:

    def __init__(
            self, state_dim, action_dim, action_range, hidden_dim, replay_buffer, policy_target_update_interval=1,
            q_lr=3e-4, policy_lr=3e-4
    ):
        self.replay_buffer = replay_buffer

        # initialize all networks
        self.q_net1 = QNetwork(state_dim, action_dim, hidden_dim)
        self.q_net2 = QNetwork(state_dim, action_dim, hidden_dim)
        self.target_q_net1 = QNetwork(state_dim, action_dim, hidden_dim)
        self.target_q_net2 = QNetwork(state_dim, action_dim, hidden_dim)
        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim, action_range)
        self.target_policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim, action_range)
        print('Q Network (1,2): ', self.q_net1)
        print('Policy Network: ', self.policy_net)

        # initialize weights of target networks
        self.target_q_net1 = self.target_ini(self.q_net1, self.target_q_net1)
        self.target_q_net2 = self.target_ini(self.q_net2, self.target_q_net2)
        self.target_policy_net = self.target_ini(self.policy_net, self.target_policy_net)

        # set train mode
        self.q_net1.train()
        self.q_net2.train()
        self.target_q_net1.eval()
        self.target_q_net2.eval()
        self.policy_net.train()
        self.target_policy_net.eval()

        self.update_cnt = 0
        self.policy_target_update_interval = policy_target_update_interval

        self.q_optimizer1 = tf.optimizers.Adam(q_lr)
        self.q_optimizer2 = tf.optimizers.Adam(q_lr)
        self.policy_optimizer = tf.optimizers.Adam(policy_lr)

    def target_ini(self, net, target_net):
        """ hard-copy update for initializing target networks """
        for target_param, param in zip(target_net.trainable_weights, net.trainable_weights):
            target_param.assign(param)
        return target_net

    def target_soft_update(self, net, target_net, soft_tau):
        """ soft update the target net with Polyak averaging """
        for target_param, param in zip(target_net.trainable_weights, net.trainable_weights):
            target_param.assign(  # copy weight value into target parameters
                target_param * (1.0 - soft_tau) + param * soft_tau
            )
        return target_net

    def update(self, batch_size, eval_noise_scale, reward_scale=10., gamma=0.9, soft_tau=1e-2):
        """ update all networks in TD3 """
        self.update_cnt += 1
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)

        reward = reward[:, np.newaxis]  # expand dim
        done = done[:, np.newaxis]

        new_next_action = self.target_policy_net.evaluate(
            next_state, eval_noise_scale=eval_noise_scale
        )  # clipped normal noise
        reward = reward_scale * (reward - np.mean(reward, axis=0)) / (
            np.std(reward, axis=0) + 1e-6
        )  # normalize with batch mean and std; plus a small number to prevent numerical problem

        # Training Q Function
        target_q_input = tf.concat([next_state, new_next_action], 1)  # the dim 0 is number of samples
        target_q_min = tf.minimum(self.target_q_net1(target_q_input), self.target_q_net2(target_q_input))

        target_q_value = reward + (1 - done) * gamma * target_q_min  # if done==1, only reward
        q_input = tf.concat([state, action], 1)  # input of q_net

        with tf.GradientTape() as q1_tape:
            predicted_q_value1 = self.q_net1(q_input)
            q_value_loss1 = tf.reduce_mean(tf.square(predicted_q_value1 - target_q_value))
        q1_grad = q1_tape.gradient(q_value_loss1, self.q_net1.trainable_weights)
        self.q_optimizer1.apply_gradients(zip(q1_grad, self.q_net1.trainable_weights))

        with tf.GradientTape() as q2_tape:
            predicted_q_value2 = self.q_net2(q_input)
            q_value_loss2 = tf.reduce_mean(tf.square(predicted_q_value2 - target_q_value))
        q2_grad = q2_tape.gradient(q_value_loss2, self.q_net2.trainable_weights)
        self.q_optimizer2.apply_gradients(zip(q2_grad, self.q_net2.trainable_weights))

        # Training Policy Function
        if self.update_cnt % self.policy_target_update_interval == 0:
            with tf.GradientTape() as p_tape:
                new_action = self.policy_net.evaluate(
                    state, eval_noise_scale=0.0
                )  # no noise, deterministic policy gradients
                new_q_input = tf.concat([state, new_action], 1)
                # """ implementation 1 """
                # predicted_new_q_value = tf.minimum(self.q_net1(new_q_input),self.q_net2(new_q_input))
                """ implementation 2 """
                predicted_new_q_value = self.q_net1(new_q_input)
                policy_loss = -tf.reduce_mean(predicted_new_q_value)
            p_grad = p_tape.gradient(policy_loss, self.policy_net.trainable_weights)
            self.policy_optimizer.apply_gradients(zip(p_grad, self.policy_net.trainable_weights))

            # Soft update the target nets
            self.target_q_net1 = self.target_soft_update(self.q_net1, self.target_q_net1, soft_tau)
            self.target_q_net2 = self.target_soft_update(self.q_net2, self.target_q_net2, soft_tau)
            self.target_policy_net = self.target_soft_update(self.policy_net, self.target_policy_net, soft_tau)

    def save(self):  # save trained weights
        path = os.path.join('Results', '_'.join([ALG_NAME, ENV_ID]))
        if not os.path.exists(path):
            os.makedirs(path)
        extend_path = lambda s: os.path.join(path, s)
        tl.files.save_npz(self.q_net1.trainable_weights, extend_path('model_q_net1.npz'))
        tl.files.save_npz(self.q_net2.trainable_weights, extend_path('model_q_net2.npz'))
        tl.files.save_npz(self.target_q_net1.trainable_weights, extend_path('model_target_q_net1.npz'))
        tl.files.save_npz(self.target_q_net2.trainable_weights, extend_path('model_target_q_net2.npz'))
        tl.files.save_npz(self.policy_net.trainable_weights, extend_path('model_policy_net.npz'))
        tl.files.save_npz(self.target_policy_net.trainable_weights, extend_path('model_target_policy_net.npz'))

    def load(self):  # load trained weights
        path = os.path.join('Results', '_'.join([ALG_NAME, ENV_ID]))
        extend_path = lambda s: os.path.join(path, s)
        tl.files.load_and_assign_npz(extend_path('model_q_net1.npz'), self.q_net1)
        tl.files.load_and_assign_npz(extend_path('model_q_net2.npz'), self.q_net2)
        tl.files.load_and_assign_npz(extend_path('model_target_q_net1.npz'), self.target_q_net1)
        tl.files.load_and_assign_npz(extend_path('model_target_q_net2.npz'), self.target_q_net2)
        tl.files.load_and_assign_npz(extend_path('model_policy_net.npz'), self.policy_net)
        tl.files.load_and_assign_npz(extend_path('model_target_policy_net.npz'), self.target_policy_net)


########################################### Env Functions ###########################################

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

#############################################  MAIN CODE #############################################
if __name__ == '__main__':
    # initialization of bitwidth values:
    bitwidth = [8, [2, 2, 2], [2, 2, 2], [2, 2, 2]]

    # To store [rewards, prev_state, action, reward, next_state]
    ep_reward_list = []
    avg_reward_list = []
    itr_data_state = []
    itr_data_action = []
    itr_data_action_bit = []
    itr_data_reward = []
    itr_data_next_state = []

    # reproducible
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    # initialization of buffer
    replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
    # initialization of trainer
    agent = TD3(
        state_dim, action_dim, action_range, HIDDEN_DIM, replay_buffer, POLICY_TARGET_UPDATE_INTERVAL, Q_LR, POLICY_LR
    )

    # training loop
    if args.train:
        state = getState(bitwidth).astype(np.float32)   # need an extra call here to make inside
        agent.policy_net([state])
        agent.target_policy_net([state])

        for ep in range(1, TRAIN_EPISODES+1):
            state = getState(bitwidth).astype(np.float32)
            episodic_reward = 0

            for i in range(1, num_itr+1):
                print('<<<<<<<<<<<<<<<<<<<<<<<<<<<< Episode # {}-{} >>>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(ep,i))
                # get actions
                action = agent.policy_net.get_action(state)
                action_bits = np.floor((7 * np.array(action) + 9) / 2)
                action_value = np.squeeze(action_bits)
                action_v = action_value.astype(int)
                print("  action_value = [{}, {}, {}], [{}, {}, {}], [{}, {}, {}]".format(
                    action_v[0], action_v[1], action_v[2],
                    action_v[3], action_v[4], action_v[5],
                    action_v[6], action_v[7], action_v[8]))

                # apply actions
                bitwidth = [8, [action_value[0], action_value[1], action_value[2]],
                            [action_value[3], action_value[4], action_value[5]],
                            [action_value[6], action_value[7], action_value[8]]]

                # get rewards and next states
                r_acc, r_w, r_bp, r_b, r_a, r_kl = getReward(bitwidth)
                reward = r_acc + r_w + r_bp + r_b + r_a
                next_state = getState(bitwidth).astype(np.float32)

                # add to the memory buffer
                if i == num_itr:
                    done = 1
                else:
                    done = 0
                replay_buffer.push(state, action, reward, next_state, done)

                # append/store values
                itr_data_state.append(state)
                itr_data_action.append(np.squeeze(np.array(action)))
                itr_data_action_bit.append(action_value)
                itr_data_reward.append(reward)
                itr_data_next_state.append(next_state)
                episodic_reward += reward

                if len(replay_buffer) > BATCH_SIZE:
                    for i in range(UPDATE_ITR):
                        agent.update(BATCH_SIZE, EVAL_NOISE_SCALE, REWARD_SCALE)

                state = next_state

            ep_reward_list.append(episodic_reward)
            print('########################################################################')
            print('           << Episode # {} is over! Updtae Actor-Critic ... >>           '.format(ep))
            print("> episode_reward = {}".format(episodic_reward))
            # Mean of last 40 episodes
            avg_reward = np.mean(ep_reward_list[-avg_N:])
            print("  avg reward is ==> {}".format(avg_reward))
            avg_reward_list.append(avg_reward)
            if (ep % 50) == 0:
                np.savetxt('ep_reward_list_{}.csv'.format(ep), ep_reward_list)

        # Save the model:
        agent.save()
        np.savetxt('./Results/avg_reward_list.csv', avg_reward_list)
        np.savetxt('./Results/ep_reward_list.csv', ep_reward_list)
        np.savetxt('./Results/itr_data_prev_state.csv', itr_data_state)
        np.savetxt('./Results/itr_data_action.csv', itr_data_action)
        np.savetxt('./Results/itr_data_action_bit.csv', itr_data_action_bit)
        np.savetxt('./Results/itr_data_reward.csv', itr_data_reward)
        np.savetxt('./Results/itr_data_next_state.csv', itr_data_next_state)

        # Plotting graph
        # Episodes versus Avg. Rewards
        plt.plot(avg_reward_list)
        plt.xlabel("Episode")
        plt.ylabel("Avg. Epsiodic Reward")
        plt.show()

    # if args.test:
    #     agent.load()
    #
    #     # need an extra call here to make inside functions be able to use model.forward
    #     state = env.reset().astype(np.float32)
    #     agent.policy_net([state])
    #
    #     for episode in range(TEST_EPISODES):
    #         state = env.reset().astype(np.float32)
    #         episode_reward = 0
    #         for step in range(MAX_STEPS):
    #             env.render()
    #             action = agent.policy_net.get_action(state, EXPLORE_NOISE_SCALE, greedy=True)
    #             state, reward, done, info = env.step(action)
    #             state = state.astype(np.float32)
    #             episode_reward += reward
    #             if done:
    #                 break
    #         print(
    #             'Testing  | Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
    #                 episode + 1, TEST_EPISODES, episode_reward,
    #                 time.time() - t0
    #             )
    #         )