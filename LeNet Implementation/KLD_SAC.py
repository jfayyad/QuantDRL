#!/usr/bin/python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import argparse
import random
import matplotlib.pyplot as plt
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
ENV_ID = 'SAC_Quant'    # environment id
RANDOM_SEED = 2         # random seed

# RL training
ALG_NAME = 'SAC'
TRAIN_EPISODES = 300   # total number of episodes for training
TEST_EPISODES = 1      # total number of episodes for training

state_dim = 18         # number of states
action_dim = 9         # number of actions
action_range = 1       # scale action, [-action_range, action_range]
num_itr = 10            # number of iterations on each episode
avg_N = 20              # number of values that used for averaging

BATCH_SIZE = 64         # update batch size
HIDDEN_DIM = 128        # size of hidden layers for networks
UPDATE_ITR = 3          # repeated updates for single step
SOFT_Q_LR = 1e-4        # q_net learning rate
POLICY_LR = 2e-4        # policy_net learning rate
ALPHA_LR = 1e-4         # alpha learning rate
POLICY_TARGET_UPDATE_INTERVAL = 3  # delayed update for the policy network and target networks
REWARD_SCALE = 10.0       # value range of reward
REPLAY_BUFFER_SIZE = 5e5  # size of the replay buffer

AUTO_ENTROPY = True     # automatically updating variable alpha for entropy

# alpha_array = []
itr_data_errors = []

###############################  SAC  ####################################

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

    def sample(self, BATCH_SIZE):
        batch = random.sample(self.buffer, BATCH_SIZE)
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


class SoftQNetwork(Model):
    """ the network for evaluate values of state-action pairs: Q(s,a) """

    def __init__(self, num_inputs, num_actions, hidden_dim, init_w=3e-3):
        super(SoftQNetwork, self).__init__()
        input_dim = num_inputs + num_actions
        w_init = tf.keras.initializers.glorot_normal(
            seed=None
        )  # glorot initialization is better than uniform in practice
        # w_init = tf.random_uniform_initializer(-init_w, init_w)

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

    def __init__(
            self, num_inputs, num_actions, hidden_dim, action_range=1., init_w=3e-3, log_std_min=-20, log_std_max=2
    ):
        super(PolicyNetwork, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        w_init = tf.keras.initializers.glorot_normal(seed=None)
        # w_init = tf.random_uniform_initializer(-init_w, init_w)

        self.linear1 = Dense(n_units=hidden_dim, act=tf.nn.relu, W_init=w_init, in_channels=num_inputs, name='policy1')
        self.linear2 = Dense(n_units=hidden_dim, act=tf.nn.relu, W_init=w_init, in_channels=hidden_dim, name='policy2')
        self.linear3 = Dense(n_units=hidden_dim, act=tf.nn.relu, W_init=w_init, in_channels=hidden_dim, name='policy3')

        self.mean_linear = Dense(
            n_units=num_actions, W_init=w_init, b_init=tf.random_uniform_initializer(-init_w, init_w),
            in_channels=hidden_dim, name='policy_mean'
        )
        self.log_std_linear = Dense(
            n_units=num_actions, W_init=w_init, b_init=tf.random_uniform_initializer(-init_w, init_w),
            in_channels=hidden_dim, name='policy_logstd'
        )

        self.action_range = action_range
        self.num_actions = num_actions

    def forward(self, state):
        x = self.linear1(state)
        x = self.linear2(x)
        x = self.linear3(x)

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = tf.clip_by_value(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def evaluate(self, state, epsilon=1e-6):
        """ generate action with state for calculating gradients """
        state = state.astype(np.float32)
        mean, log_std = self.forward(state)
        std = tf.math.exp(log_std)  # no clip in evaluation, clip affects gradients flow

        normal = Normal(0, 1)
        z = normal.sample(mean.shape)
        action_0 = tf.math.tanh(mean + std * z)  # TanhNormal distribution as actions; reparameterization trick
        action = self.action_range * action_0
        # according to original paper, with an extra last term for normalizing different action range
        log_prob = Normal(mean, std).log_prob(mean + std * z) - tf.math.log(1. - action_0**2 +
                                                                            epsilon) - np.log(self.action_range)
        # both dims of normal.log_prob and -log(1-a**2) are (N,dim_of_action);
        # the Normal.log_prob outputs the same dim of input features instead of 1 dim probability,
        # needs sum up across the dim of actions to get 1 dim probability; or else use Multivariate Normal.
        log_prob = tf.reduce_sum(log_prob, axis=1)[:, np.newaxis]  # expand dim as reduce_sum causes 1 dim reduced

        return action, log_prob, z, mean, log_std

    def get_action(self, state, greedy=False):
        """ generate action with state for interaction with envronment """
        # print('  state = {}'.format(type(state)))
        mean, log_std = self.forward([state])
        std = tf.math.exp(log_std)

        normal = Normal(0, 1)
        z = normal.sample(mean.shape)
        action = self.action_range * tf.math.tanh(
            mean + std * z
        )  # TanhNormal distribution as actions; reparameterization trick

        action = self.action_range * tf.math.tanh(mean) if greedy else action
        return action.numpy()[0]

    def sample_action(self, ):
        """ generate random actions for exploration """
        a = tf.random.uniform([self.num_actions], -1, 1)
        return self.action_range * a.numpy()


class SAC:

    def __init__(
            self, state_dim, action_dim, action_range, hidden_dim, replay_buffer, SOFT_Q_LR=3e-4, POLICY_LR=3e-4,
            ALPHA_LR=3e-4
    ):
        self.replay_buffer = replay_buffer

        # initialize all networks
        self.soft_q_net1 = SoftQNetwork(state_dim, action_dim, hidden_dim)
        self.soft_q_net2 = SoftQNetwork(state_dim, action_dim, hidden_dim)
        self.target_soft_q_net1 = SoftQNetwork(state_dim, action_dim, hidden_dim)
        self.target_soft_q_net2 = SoftQNetwork(state_dim, action_dim, hidden_dim)
        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim, action_range)
        self.soft_q_net1.train()
        self.soft_q_net2.train()
        self.target_soft_q_net1.eval()
        self.target_soft_q_net2.eval()
        self.policy_net.train()

        self.log_alpha = tf.Variable(0, dtype=np.float32, name='log_alpha')
        self.alpha = tf.math.exp(self.log_alpha)
        # print('Soft Q Network (1,2): ', self.soft_q_net1)
        # print('Policy Network: ', self.policy_net)
        # set mode
        self.soft_q_net1.train()
        self.soft_q_net2.train()
        self.target_soft_q_net1.eval()
        self.target_soft_q_net2.eval()
        self.policy_net.train()

        # initialize weights of target networks
        self.target_soft_q_net1 = self.target_ini(self.soft_q_net1, self.target_soft_q_net1)
        self.target_soft_q_net2 = self.target_ini(self.soft_q_net2, self.target_soft_q_net2)

        self.soft_q_optimizer1 = tf.optimizers.Adam(SOFT_Q_LR)
        self.soft_q_optimizer2 = tf.optimizers.Adam(SOFT_Q_LR)
        self.policy_optimizer = tf.optimizers.Adam(POLICY_LR)
        self.alpha_optimizer = tf.optimizers.Adam(ALPHA_LR)

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

    def update(self, batch_size, reward_scale=10., auto_entropy=True, target_entropy=-2, gamma=0.99, soft_tau=1e-2):
        """ update all networks in SAC """
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)

        reward = reward[:, np.newaxis]  # expand dim
        done = done[:, np.newaxis]

        reward = reward_scale * (reward - np.mean(reward, axis=0)) / (
            np.std(reward, axis=0) + 1e-6
        )  # normalize with batch mean and std; plus a small number to prevent numerical problem

        # Training Q Function
        new_next_action, next_log_prob, _, _, _ = self.policy_net.evaluate(next_state)
        target_q_input = tf.concat([next_state, new_next_action], 1)  # the dim 0 is number of samples
        target_q_min = tf.minimum(
            self.target_soft_q_net1(target_q_input), self.target_soft_q_net2(target_q_input)
        ) - self.alpha * next_log_prob
        target_q_value = reward + (1 - done) * gamma * target_q_min  # if done==1, only reward
        q_input = tf.concat([state, action], 1)  # the dim 0 is number of samples

        with tf.GradientTape() as q1_tape:
            predicted_q_value1 = self.soft_q_net1(q_input)
            q_value_loss1 = tf.reduce_mean(tf.losses.mean_squared_error(predicted_q_value1, target_q_value))
        q1_grad = q1_tape.gradient(q_value_loss1, self.soft_q_net1.trainable_weights)
        self.soft_q_optimizer1.apply_gradients(zip(q1_grad, self.soft_q_net1.trainable_weights))

        with tf.GradientTape() as q2_tape:
            predicted_q_value2 = self.soft_q_net2(q_input)
            q_value_loss2 = tf.reduce_mean(tf.losses.mean_squared_error(predicted_q_value2, target_q_value))
        q2_grad = q2_tape.gradient(q_value_loss2, self.soft_q_net2.trainable_weights)
        self.soft_q_optimizer2.apply_gradients(zip(q2_grad, self.soft_q_net2.trainable_weights))

        # Training Policy Function
        with tf.GradientTape() as p_tape:
            new_action, log_prob, z, mean, log_std = self.policy_net.evaluate(state)
            new_q_input = tf.concat([state, new_action], 1)  # the dim 0 is number of samples
            """ implementation 1 """
            predicted_new_q_value = tf.minimum(self.soft_q_net1(new_q_input), self.soft_q_net2(new_q_input))
            # """ implementation 2 """
            # predicted_new_q_value = self.soft_q_net1(new_q_input)
            policy_loss = tf.reduce_mean(self.alpha * log_prob - predicted_new_q_value)
        p_grad = p_tape.gradient(policy_loss, self.policy_net.trainable_weights)
        self.policy_optimizer.apply_gradients(zip(p_grad, self.policy_net.trainable_weights))

        # Updating alpha w.r.t entropy
        # alpha: trade-off between exploration (max entropy) and exploitation (max Q)
        if auto_entropy is True:
            with tf.GradientTape() as alpha_tape:
                alpha_loss = -tf.reduce_mean((self.log_alpha * (log_prob + target_entropy)))
            alpha_grad = alpha_tape.gradient(alpha_loss, [self.log_alpha])
            self.alpha_optimizer.apply_gradients(zip(alpha_grad, [self.log_alpha]))
            self.alpha = tf.math.exp(self.log_alpha)
        else:  # fixed alpha
            self.alpha = 1.
            alpha_loss = 0
        # alpha_array.append(np.array(self.alpha))        #YY
        # Soft update the target value nets
        self.target_soft_q_net1 = self.target_soft_update(self.soft_q_net1, self.target_soft_q_net1, soft_tau)
        self.target_soft_q_net2 = self.target_soft_update(self.soft_q_net2, self.target_soft_q_net2, soft_tau)

    def save(self):  # save trained weights
        path = os.path.join('Results', '_'.join([ALG_NAME, ENV_ID]))
        if not os.path.exists(path):
            os.makedirs(path)
        extend_path = lambda s: os.path.join(path, s)
        tl.files.save_npz(self.soft_q_net1.trainable_weights, extend_path('model_q_net1.npz'))
        tl.files.save_npz(self.soft_q_net2.trainable_weights, extend_path('model_q_net2.npz'))
        tl.files.save_npz(self.target_soft_q_net1.trainable_weights, extend_path('model_target_q_net1.npz'))
        tl.files.save_npz(self.target_soft_q_net2.trainable_weights, extend_path('model_target_q_net2.npz'))
        tl.files.save_npz(self.policy_net.trainable_weights, extend_path('model_policy_net.npz'))
        np.save(extend_path('log_alpha.npy'), self.log_alpha.numpy())  # save log_alpha variable

    def load_weights(self):  # load trained weights
        path = os.path.join('Results', '_'.join([ALG_NAME, ENV_ID]))
        extend_path = lambda s: os.path.join(path, s)
        tl.files.load_and_assign_npz(extend_path('model_q_net1.npz'), self.soft_q_net1)
        tl.files.load_and_assign_npz(extend_path('model_q_net2.npz'), self.soft_q_net2)
        tl.files.load_and_assign_npz(extend_path('model_target_q_net1.npz'), self.target_soft_q_net1)
        tl.files.load_and_assign_npz(extend_path('model_target_q_net2.npz'), self.target_soft_q_net2)
        tl.files.load_and_assign_npz(extend_path('model_policy_net.npz'), self.policy_net)
        self.log_alpha.assign(np.load(extend_path('log_alpha.npy')))  # load log_alpha variable

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

    # To store reward history of each episode
    ep_reward_list = []
    # To store average reward history of last few episodes
    avg_reward_list = []
    # To store [prev_state, action, reward, next_state] of each iteration 
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
    agent = SAC(state_dim, action_dim, action_range, HIDDEN_DIM, replay_buffer, SOFT_Q_LR, POLICY_LR, ALPHA_LR)

    bitwidth = [8, [2, 2, 2], [2, 2, 2], [2, 2, 2]]

    # training loop
    if args.train:
        # Training
        state = getState(bitwidth).astype(np.float32)   # need an extra call here to make inside
        agent.policy_net([state])                       # functions be able to use model.forward
        for ep in range(1, TRAIN_EPISODES+1):
            state = getState(bitwidth).astype(np.float32)
            # agent.policy_net([state])
            episodic_reward = 0
            for i in range(1, num_itr+1):
                print('<<<<<<<<<<<<<<<<<<<<<<<<<<<< Episode # {}-{} >>>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(ep,i))
                # tf_state = tf.make_tensor_proto(tf.expand_dims(tf.convert_to_tensor(state), 0))
                # action = agent.policy_net.get_action(tf.make_ndarray(tf_state))
                action = agent.policy_net.get_action(state)
                action_bits = np.floor((7 * np.array(action) + 9) / 2)

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
                next_state = getState(bitwidth).astype(np.float32)
                # print('  new_state  = {}'.format(state))
                print('  rewards: [accuracy, weight, avg_bit_para, bias, activation, kl-d] =\n           [{}, {}, {}, {}, {}, {}] \n  total_reward = {}'.format(
                        r_acc, r_w, r_bp, r_b, r_a, r_kl, reward))

                # add to the memory buffer
                if i == num_itr:
                    done = 1
                else: 
                    done = 0
                replay_buffer.push(state, action, reward, next_state, done)

                itr_data_state.append(state)
                itr_data_action.append(np.squeeze(np.array(action)))
                itr_data_action_bit.append(action_value)
                itr_data_reward.append(reward)
                itr_data_next_state.append(next_state)
                episodic_reward += reward

                # print('  replay_buffer_size = {}'.format(len(replay_buffer)))

                if len(replay_buffer) > BATCH_SIZE:
                    for j in range(UPDATE_ITR):
                        agent.update(BATCH_SIZE, reward_scale=REWARD_SCALE, auto_entropy=AUTO_ENTROPY, target_entropy=-1. * action_dim)

                state = next_state

            # print('  eps_counter     = {}'.format(counter))
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
    #     agent.load_weights()

    #     # need an extra call here to make inside functions be able to use model.forward
    #     state = env.reset().astype(np.float32)
    #     agent.policy_net([state])

    #     for episode in range(TEST_EPISODES):
    #         state = env.reset().astype(np.float32)
    #         episode_reward = 0
    #         for step in range(MAX_STEPS):
    #             env.render()
    #             state, reward, done, info = env.step(agent.policy_net.get_action(state, greedy=True))
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

    # if args.test:
    #     # Testing
    #     agent.load_weights()
    #     os.system('clear')
    #     updateSetpoint(goal)
    #     # updateSetpoint(goal)
    #     state = getState().astype(np.float32)   # need an extra call here to make inside
    #     agent.policy_net([state])               # functions be able to use model.forward
    #     for ep in range(1, TEST_EPISODES+1):
    #         # Reset the setpoint for the new episode
    #         goal.pose.position.x = 5.0
    #         goal.pose.position.y = 5.0
    #         goal.pose.position.z = 6.0
    #         # goal.pose.orientation.x = 0.5
    #         # goal.pose.orientation.y = 0.5
    #         # goal.pose.orientation.z = 0.0
    #         # goal.pose.orientation.w = 0.5
    #         resetDrone(altitude=6, velMinMax=1.6, duration=1.0)
    #         goal_publisher.publish(goal)
    #         updateSetpoint(goal)
    #         state = getState().astype(np.float32)
    #         episodic_reward = 0
    #
    #         for i in range(1, num_itr+1):
    #             print('<<<<<<<<<<<<<<<<<<<<<<<<<<<< Episode # {}-{} >>>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(ep,i))
    #             goal_publisher.publish(goal)
    #             # r = quaternion_to_euler([goal.pose.orientation.x, goal.pose.orientation.y, goal.pose.orientation.z, goal.pose.orientation.w])
    #             updateSetpoint(goal)
    #             rospy.sleep(0.05)
    #             state = getState().astype(np.float32)
    #             # tf_state = tf.make_tensor_proto(tf.expand_dims(tf.convert_to_tensor(state), 0))
    #             # action = agent.policy_net.get_action(tf.make_ndarray(tf_state))
    #             action = agent.policy_net.get_action(state)
    #
    #             action_pub = np.array(action)
    #             action_pub = (action_pub+1)/2
    #             state_ = [state[0], state[2], state[1]*10]
    #
    #             print('> current_state   = {}'.format(state_))
    #             # print('  selected_action = {}'.format(action))
    #             print("  action_value    = {}".format(action_pub*200))
    #             print("---- apply action ----")
    #
    #             W_ddpg = Float32MultiArray()
    #
    #             ################################################ Action ################################################
    #             sac_action = True
    #             if sac_action == True:
    #                 # A. Actions from SAC model:
    #                 W_ddpg.data = [ action_pub[0],action_pub[1],action_pub[2],action_pub[3],action_pub[4],action_pub[5],
    #                                 action_pub[6],action_pub[7],action_pub[8],action_pub[9],action_pub[10],action_pub[11] ];
    #             else:
    #                 # B. Constant actions for comparison purposes with SAC:
    #                 w_const = 50/200
    #                 W_ddpg.data = [ w_const, w_const, w_const, w_const*2, w_const*2, w_const*2, w_const, w_const, w_const, w_const, w_const, w_const];
    #             #########################################################################################################
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
    #             next_state = getState().astype(np.float32)
    #             # print('  new_state  = {}'.format(next_state))
    #             print('  reward_rms = {} ,  reward_sp = {} ,  reward_des = {}'.format(reward_rms, reward_sp, reward_des))
    #             # print('  reward     = {}'.format(reward))
    #
    #             # Save data
    #             itr_data_state.append(state_)
    #             itr_data_action.append(np.squeeze(np.array(action)))
    #             itr_data_reward.append([reward_rms, reward_sp, reward_des])
    #             itr_data_next_state.append(next_state)
    #             episodic_reward += reward
    #
    #             state = next_state
    #
    #             # # Training path _ zig-zag
    #             # goal.pose.position.x += 4.0
    #             # if (i % 2) == 0:
    #             #     goal.pose.position.y += 4.0
    #             # else:
    #             #     goal.pose.position.y -= 4.0
    #
    #             # # 3D path
    #             # goal.pose.position.x += 4.0
    #             # goal.pose.position.z += 2.0
    #             # if (i % 2) == 0:
    #             #     goal.pose.position.y += 4.0
    #             #     goal.pose.position.z -= 3.0
    #             # else:
    #             #     goal.pose.position.y -= 4.0
    #
    #             # # Square path
    #             # if i == 1:
    #             #     goal.pose.position.y -= 5.0
    #             # elif i == 2:
    #             #     goal.pose.position.x -= 5.0
    #             # elif i == 3:
    #             #     goal.pose.position.y += 5.0
    #             # elif i == 4:
    #             #     goal.pose.position.x += 5.0
    #             # else:
    #             #     goal.pose.position.x += 5.0
    #             #     goal.pose.position.y += 5.0
    #
    #             # # Square path _ Helical
    #             # if i == 1:
    #             #     goal.pose.position.y -= 5.0
    #             #     goal.pose.position.z += 0.75
    #             # elif i == 2:
    #             #     goal.pose.position.x -= 5.0
    #             #     goal.pose.position.z += 0.75
    #             # elif i == 3:
    #             #     goal.pose.position.y += 5.0
    #             #     goal.pose.position.z += 0.75
    #             # elif i == 4:
    #             #     goal.pose.position.x += 5.0
    #             #     goal.pose.position.z += 0.75
    #             # else:
    #             #     goal.pose.position.x += 5.0
    #             #     goal.pose.position.y += 5.0
    #
    #             # # Random 2D Exploration:
    #             # sp_val_x = random.choice([-6,-5,-4,4,5,6])
    #             # sp_val_y = random.choice([-6,-5,-4,4,5,6])
    #             # goal.pose.position.x += sp_val_x
    #             # goal.pose.position.y += sp_val_y
    #
    #             # Random 3D Exploration:
    #             sp_val_x = random.choice([-6,-5,-4,4,5,6])
    #             sp_val_y = random.choice([-6,-5,-4,4,5,6])
    #             sp_val_z = random.choice([-0.5,-1.0,0,0.5,1.0])
    #             goal.pose.position.x += sp_val_x
    #             goal.pose.position.y += sp_val_y
    #             goal.pose.position.z += sp_val_z
    #
    #             # q = euler_to_quaternion([0,0,])
    #
    #             rospy.sleep(0.05)
    #             sub.unregister()
    #
    #         # print('  eps_counter     = {}'.format(counter))
    #         ep_reward_list.append(episodic_reward)
    #         print('########################################################################')
    #         print('                     << Episode # {} is over! ... >>                    '.format(ep))
    #         print("> episode_reward = {}".format(episodic_reward))
    #         # Mean of last 40 episodes
    #         avg_reward = np.mean(ep_reward_list[-avg_N:])
    #         print("  avg reward is ==> {}".format(avg_reward))
    #         avg_reward_list.append(avg_reward)
    #         # if (ep % 5) == 0:
    #         #     np.savetxt('testing_ep_reward_list_{}.csv'.format(ep), ep_reward_list)
    #         #     np.savetxt('testing_itr_data_reward_{}.csv'.format(ep), itr_data_reward)
    #         #     np.savetxt('testing_itr_data_errors_{}.csv'.format(ep), itr_data_errors)
    #
    #     # Save Data
    #     np.savetxt('testing_ep_reward_list.csv', ep_reward_list)
    #     np.savetxt('testing_itr_data_reward.csv', itr_data_reward)
    #     np.savetxt('testing_itr_data_errors.csv', itr_data_errors)
    #
    #     np.savetxt('testing_itr_data_next_state.csv', itr_data_next_state)
    #     # np.savetxt('alpha_array.csv', alpha_array)
    #     np.savetxt('testing_itr_data_prev_state.csv', itr_data_state)
    #     np.savetxt('testing_itr_data_action.csv', itr_data_action)
    #     # np.savetxt('testing_avg_reward_list.csv', avg_reward_list)
    #
    #     # Plotting graph
    #     # Episodes versus Avg. Rewards
    #     # plt.plot(avg_reward_list)
    #     # plt.xlabel("Episode")
    #     # plt.ylabel("Avg. Epsiodic Reward")
    #     # plt.show()