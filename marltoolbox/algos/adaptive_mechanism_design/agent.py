##########
# Code from: https://github.com/tobiasbaumann1/Adaptive_Mechanism_Design
##########
import copy
import logging
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import math_ops

logging.basicConfig(filename='Agents.log', level=logging.DEBUG)

from enum import Enum, auto


def convert_from_rllib_env_format(data, player_ids, state: bool = False, n_states: int = None, coin_game=False):
    # for Done
    if "__all__" in data.keys():
        return data["__all__"]

    if state:
        obs = np.array(data[player_ids[0]])
        if not coin_game:
            obs_one_hot = np.zeros((n_states))
            obs_one_hot[obs] = 1
            return obs_one_hot
        else:
            return obs

    formated_data = []
    for player_id in player_ids:
        formated_data.append(data[player_id])
    return np.array(formated_data)


def convert_to_rllib_env_format(data, player_ids, coin_game=False):
    formated_data = {}
    for data_element, player_id in zip(data, player_ids):
        if not coin_game:
            if data_element == 1:
                formated_data[player_id] = 0
            elif data_element == 0:
                formated_data[player_id] = 1
            else:
                raise ValueError()
        else:
            formated_data[player_id] = data_element
    return formated_data


def var_shape(x):
    out = x.get_shape().as_list()
    return out


def intprod(x):
    return int(np.prod(x))


def numel(x):
    return intprod(var_shape(x))


class Critic_Variant(Enum):
    INDEPENDENT = auto()
    CENTRALIZED = auto()
    CENTRALIZED_APPROX = auto()


class Agent(object):
    def __init__(self, env, learning_rate=0.001, gamma=0.95, agent_idx=0):
        self.sess = tf.Session()
        self.env = env
        self.n_actions = env.NUM_ACTIONS
        self.n_features = env.NUM_STATES
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.agent_idx = agent_idx
        self.log = []  # logs action probabilities

    def choose_action(self, s):
        action_probs = self.calc_action_probs(s)
        # print("action_probs.shape[1]", action_probs.shape[1])
        # print("action_probs", action_probs)
        action = np.random.choice(range(action_probs.shape[1]),
                                  p=action_probs.ravel())  # select action w.r.t the actions prob
        # print("action", action)
        self.log.append(action_probs[0, 1])
        return action

    def learn_at_episode_end(self):
        pass

    def close(self):
        self.sess.close()
        tf.reset_default_graph()

    def reset(self):
        self.sess.run(tf.global_variables_initializer())

    def get_weights_norm(self):
        return self.sess.run(self.weights_norm, {})

class Actor_Critic_Agent(Agent):
    def __init__(self, env, learning_rate=0.001, n_units_actor=20,
                 n_units_critic=20, gamma=0.95, agent_idx=0, mean_theta=0.0,
                 critic_variant=Critic_Variant.INDEPENDENT, weight_decay=0.0, std_theta=0.1,
                 entropy_coeff=0.001, use_adam_optimizer=True, *args):
        super().__init__(env, learning_rate, gamma, agent_idx)
        self.actor = Actor(env, n_units_actor, learning_rate, agent_idx, weight_decay=weight_decay,
                           std_theta=std_theta, entropy_coeff=entropy_coeff, mean_theta=mean_theta,
                           use_adam_optimizer=use_adam_optimizer)
        self.critic = Critic(env, n_units_critic, learning_rate, gamma, agent_idx, critic_variant,
                             mean_theta=mean_theta, weight_decay=weight_decay, std_theta=std_theta,
                             use_adam_optimizer=use_adam_optimizer)
        self.sess.run(tf.global_variables_initializer())

    def learn(self, s, a, r, s_, done=False, *args):
        if done:
            pass
        else:
            td_error, critic_loss = self.critic.learn(self.sess, s, r, s_, *args)
            advantage = self.actor.learn(self.sess, s, a, td_error)
        return critic_loss, advantage

    def __str__(self):
        return 'Actor_Critic_Agent_' + str(self.agent_idx)

    def calc_action_probs(self, s, add_dim=True):
        return self.actor.calc_action_probs(self.sess, s, add_dim)

    def calcul_value(self, s, add_dim=True):
        return self.critic.calcul_value(self.sess, s, add_dim)

    def pass_agent_list(self, agent_list):
        self.critic.pass_agent_list(agent_list)

    def get_action_prob_variable(self):
        return self.actor.actions_prob

    def get_state_variable(self):
        return self.actor.s

    def get_policy_parameters(self):
        return [self.actor.w_l1, self.actor.b_l1, self.actor.w_pi1, self.actor.b_pi1]

    def calc_g_log_pi(self, s, a):
        return self.actor.calc_g_log_pi(self.sess, s, a)

    def get_weights_norm(self):
        return self.actor.get_weights_norm(self.sess), self.critic.get_weights_norm(self.sess)

class Actor(object):
    def __init__(self, env, n_units=20, learning_rate=0.001, agent_idx=0, weight_decay=0.0, training=True,
                 std_theta=0.1, entropy_coeff=0.001, mean_theta=0.0, use_adam_optimizer=True):
        self.s = tf.placeholder(tf.float32, [1, env.NUM_STATES], "state_ag")
        self.a = tf.placeholder(tf.int32, None, "act")
        # self.a = tf.placeholder(tf.int32, [None, env.NUM_ACTIONS], "act")
        self.td_error = tf.placeholder(tf.float32, None, "td_error")  # TD_error
        self.env_name = env.NAME
        with tf.variable_scope(f'Actor_{agent_idx}'):
            # n_units = n_units[0]
            # self.w_l1 = tf.Variable(tf.random_normal([env.NUM_STATES, n_units], stddev=0.1))
            # self.b_l1 = tf.Variable(tf.random_normal([n_units], stddev=0.1))
            # self.l1 = tf.nn.relu(tf.matmul(self.s, self.w_l1) + self.b_l1)
            #
            # self.w_pi1 = tf.Variable(tf.random_normal([n_units, env.NUM_ACTIONS], stddev=0.1))
            # self.b_pi1 = tf.Variable(tf.random_normal([env.NUM_ACTIONS], stddev=0.1))
            # self.actions_prob = tf.nn.softmax(tf.matmul(self.l1, self.w_pi1) + self.b_pi1)
            # var_list = [self.w_l1, self.b_l1, self.w_pi1, self.b_pi1]

            if not isinstance(n_units, list):
                units = [env.NUM_STATES, n_units,  env.NUM_ACTIONS]
            else:
                units = [env.NUM_STATES] + n_units + [env.NUM_ACTIONS]
            print("units", units)
            var_list = []
            input_ = self.s
            for i in range(len(units)):
                with tf.variable_scope(f"layer_{i}"):
                    n_in = units[i]
                    n_out = units[i + 1]
                    print("i", i)
                    print("n_in", n_in)
                    print("n_out", n_out)
                    if i + 1 == len(units) - 1:
                        break
                    w_l1 = tf.Variable(tf.random_normal([n_in, n_out], mean=0.0, stddev=std_theta))
                    b_l1 = tf.Variable(tf.random_normal([n_out], mean=0.0, stddev=std_theta))
                    l1 = tf.nn.leaky_relu(tf.matmul(input_, w_l1) + b_l1)
                    var_list.extend([w_l1, b_l1])
                    # l1 = tf.compat.v1.layers.batch_normalization(l1, training=training)
                    input_ = l1
            # print("std_theta",std_theta)
            self.w_pi1 = tf.Variable(tf.random_normal([n_in, n_out], mean=0.0, stddev=std_theta))
            self.b_pi1 = tf.Variable(tf.random_normal([n_out], mean=mean_theta, stddev=std_theta))
            self.actions_prob = tf.nn.softmax(tf.matmul(input_, self.w_pi1) + self.b_pi1)
            var_list.extend([self.w_pi1, self.b_pi1])
            # opt = tf.print("tf.matmul(input_, self.w_pi1) + self.b_pi1", tf.matmul(input_, self.w_pi1),
            #                     self.b_pi1, input_, self.w_pi1)
            # with tf.control_dependencies([opt]):
            self.entropy = tf.reduce_sum(tf.log(self.actions_prob) * self.actions_prob)

            self.parameters = tf.concat(axis=0, values=[tf.reshape(v, [numel(v)]) for v in var_list])
            weights_norm = math_ops.reduce_sum(self.parameters * self.parameters, None, keepdims=True)
            self.weights_norm = tf.sqrt(tf.reduce_sum(weights_norm))


            with tf.variable_scope('exp_v'):
                log_prob = tf.log(self.actions_prob[0, self.a])
                self.exp_v = tf.reduce_mean(log_prob * self.td_error)
                self.g_log_pi = tf.gradients(log_prob, self.s)

            with tf.variable_scope('trainActor'):
                # entropy_opt = tf.print("self.entropy", self.entropy)
                # with tf.control_dependencies([entropy_opt]):
                self.loss = -self.exp_v
                if weight_decay > 0.0:
                    self.loss += self.weights_norm * weight_decay
                if entropy_coeff > 0.0:
                    self.loss += entropy_coeff * self.entropy
                if use_adam_optimizer:
                    self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
                else:
                    self.train_op = tf.train.MomentumOptimizer(learning_rate, momentum=0.9).minimize(self.loss)
                # update_ops = tf.compat.v1.get_collection(tf.GraphKeys.UPDATE_OPS, scope=f'Actor_{agent_idx}')
                # self.train_op = tf.group([self.train_op, update_ops])



        # with tf.variable_scope('Actor'):
        #     self.theta = tf.Variable(tf.random_normal([1], mean=0.0, stddev=std_theta))
        #     self.actions_prob = tf.expand_dims(tf.concat([1 - tf.sigmoid(self.theta), tf.sigmoid(self.theta)], 0), 0)
        #
        # with tf.variable_scope('exp_v'):
        #     self.log_prob = tf.log(self.actions_prob[0, self.a])
        #     self.g_log_pi = tf.gradients(self.log_prob, self.theta)
        #     self.exp_v = tf.reduce_mean(self.log_prob * self.td_error)
        #
        # with tf.variable_scope('trainActor'):
        #     self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(-self.exp_v)
        #
        # self.weights_norm = self.theta

    def learn(self, sess, s, a, td):
        s = s[np.newaxis, :]
        feed_dict = {self.s: s, self.a: a, self.td_error: td}
        _, exp_v = sess.run([self.train_op, self.exp_v], feed_dict)
        return exp_v

    def calc_action_probs(self, sess, s, add_dim=True):
        if add_dim:
            s = s[np.newaxis, :]
        probs = sess.run(self.actions_prob, {self.s: s})  # get probabilities for all actions
        # print("actor probs", probs)
        return probs
        # return np.array([[0.75, 0.25]])

    def calc_g_log_pi(self, sess, s, a):
        return sess.run(self.g_log_pi, feed_dict={self.s: s, self.a: a})

    def get_weights_norm(self, sess):
        return sess.run(self.weights_norm, {})

class Critic(object):
    def __init__(self, env, n_units, learning_rate, gamma, agent_idx,
                 critic_variant=Critic_Variant.INDEPENDENT, weight_decay=0.0, std_theta=0.1, mean_theta=0.0,
                 use_adam_optimizer=True):
        self.critic_variant = critic_variant
        self.env = env

        self.s = tf.placeholder(tf.float32, [1, env.NUM_STATES], "state_critic")
        self.v_ = tf.placeholder(tf.float32, [1, 1], "v_next")
        self.r = tf.placeholder(tf.float32, None, 'r')

        if self.critic_variant is Critic_Variant.CENTRALIZED:
            self.act_probs = tf.placeholder(tf.float32, shape=[1, env.NUM_ACTIONS * env.NUM_AGENTS], name="act_probs")
            self.nn_inputs = tf.concat([self.s, self.act_probs], axis=1)
        else:
            self.nn_inputs = self.s

        with tf.variable_scope(f'Critic_{agent_idx}'):
            # l1 = tf.layers.dense(
            #     inputs=self.nn_inputs,
            #     units=n_units,  # number of hidden units
            #     activation=tf.nn.relu,  # None
            #     kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
            #     bias_initializer=tf.constant_initializer(0.1),  # biases
            #     name='l1' + str(agent_idx)
            # )


            # n_units = n_units[0]
            # self.w_l1 = tf.Variable(tf.random_normal([env.NUM_STATES, n_units], stddev=0.1))
            # self.b_l1 = tf.Variable(tf.random_normal([n_units], stddev=0.1))
            # self.l1 = tf.nn.relu(tf.matmul(self.s, self.w_l1) + self.b_l1)
            #
            # self.w_pi1 = tf.Variable(tf.random_normal([n_units, env.NUM_ACTIONS], stddev=0.1))
            # self.b_pi1 = tf.Variable(tf.random_normal([env.NUM_ACTIONS], stddev=0.1))
            # self.actions_prob = tf.nn.softmax(tf.matmul(self.l1, self.w_pi1) + self.b_pi1)
            # var_list = [self.w_l1, self.b_l1, self.w_pi1, self.b_pi1]

            if self.critic_variant is Critic_Variant.CENTRALIZED:
                if not isinstance(n_units, list):
                    units = [env.NUM_STATES + env.NUM_ACTIONS * env.NUM_AGENTS, n_units, 1]
                else:
                    units = [env.NUM_STATES + env.NUM_ACTIONS * env.NUM_AGENTS] + n_units + [1]
            else:
                if not isinstance(n_units, list):
                    units = [env.NUM_STATES, n_units,  1]
                else:
                    units = [env.NUM_STATES] + n_units + [1]
            print("units", units)
            var_list = []
            input_ = self.nn_inputs
            for i in range(len(units)):
                with tf.variable_scope(f"layer_{i}"):
                    n_in = units[i]
                    n_out = units[i + 1]
                    print("i", i)
                    print("n_in", n_in)
                    print("n_out", n_out)
                    if i + 1 == len(units) - 1:
                        break
                    w_l1 = tf.Variable(tf.random_normal([n_in, n_out], mean=0.0, stddev=std_theta))
                    b_l1 = tf.Variable(tf.random_normal([n_out], mean=0.0, stddev=std_theta))
                    l1 = tf.nn.leaky_relu(tf.matmul(input_, w_l1) + b_l1)
                    var_list.extend([w_l1, b_l1])
                    # l1 = tf.compat.v1.layers.batch_normalization(l1, training=True)
                    input_ = l1

            # TODO use this everywhere
            # self.v = tf.layers.dense(
            #     inputs=l1,
            #     units=1,  # output units
            #     activation=None,
            #     kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
            #     bias_initializer=tf.constant_initializer(0.1),  # biases
            #     name='V' + str(agent_idx)
            # )

            self.w_pi1 = tf.Variable(tf.random_normal([n_in, n_out], mean=0.0, stddev=std_theta))
            self.b_pi1 = tf.Variable(tf.random_normal([n_out], mean=mean_theta, stddev=std_theta))
            self.v = tf.matmul(input_, self.w_pi1) + self.b_pi1
            var_list.extend([self.w_pi1, self.b_pi1])

        self.parameters = tf.concat(axis=0, values=[tf.reshape(v, [numel(v)]) for v in var_list])
        weights_norm = math_ops.reduce_sum(self.parameters * self.parameters, None, keepdims=True)
        self.weights_norm = tf.sqrt(tf.reduce_sum(weights_norm))

        with tf.variable_scope('squared_TD_error'):
            # print_opt = tf.print("self.r + gamma * self.v_ - self.v", self.r, gamma, self.v_, self.v)
            # with tf.control_dependencies([print_opt]):
            self.td_error = self.r + gamma * self.v_ - self.v
            self.loss = tf.square(self.td_error)
        with tf.variable_scope('trainCritic'):
            if weight_decay > 0.0:
                self.loss += self.weights_norm * weight_decay
            if use_adam_optimizer:
                self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
            else:
                self.train_op = tf.train.MomentumOptimizer(learning_rate, momentum=0.9).minimize(self.loss)
            # update_ops = tf.compat.v1.get_collection(tf.GraphKeys.UPDATE_OPS, scope=f'Critic_{agent_idx}')
            # self.train_op = tf.group([self.train_op, update_ops])




    def pass_agent_list(self, agent_list):
        self.agent_list = agent_list

    def learn(self, sess, s, r, s_, *args):
        s, s_ = s.astype(np.float32), s_.astype(np.float32)

        if self.critic_variant is Critic_Variant.CENTRALIZED:
            if args:
                obslist = args[0]
                obs_list = args[1]
                act_probs = np.hstack(
                    [agent.calc_action_probs(obslist[idx]) for idx, agent in enumerate(self.agent_list)])
                act_probs_ = np.hstack(
                    [agent.calc_action_probs(obs_list[idx]) for idx, agent in enumerate(self.agent_list)])
            else:
                act_probs = np.hstack([agent.calc_action_probs(s) for idx, agent in enumerate(self.agent_list)])
                act_probs_ = np.hstack([agent.calc_action_probs(s_) for idx, agent in enumerate(self.agent_list)])
            # op1 = tf.print("act_probs", act_probs)
            # op2 = tf.print("act_probs_", act_probs_)
            # with tf.control_dependencies([op1, op2]):
            nn_inputs = np.hstack([s[np.newaxis, :], act_probs])
            nn_inputs_ = np.hstack([s_[np.newaxis, :], act_probs_])
        else:
            nn_inputs, nn_inputs_ = s[np.newaxis, :], s_[np.newaxis, :]
        # print("nn_inputs_", nn_inputs_)
        v_ = sess.run(self.v, {self.nn_inputs: nn_inputs_})
        td_error, _, critic_loss = sess.run([self.td_error, self.train_op, self.loss],
                               {self.nn_inputs: nn_inputs, self.v_: v_, self.r: r})
        return td_error, critic_loss

    def calcul_value(self, sess, s, add_dim=True, *args):
        s = s.astype(np.float32)

        if self.critic_variant is Critic_Variant.CENTRALIZED:
            if args:
                raise NotImplementedError()
            #     obslist = args[0]
            #     # obs_list = args[1]
            #     act_probs = np.hstack(
            #         [agent.calc_action_probs(obslist[idx]) for idx, agent in enumerate(self.agent_list)])
            # else:
            act_probs = np.hstack([agent.calc_action_probs(s, add_dim=add_dim) for idx, agent in enumerate(self.agent_list)])
            if add_dim:
                nn_inputs = np.hstack([s[np.newaxis, :], act_probs])
            else:
                nn_inputs = np.hstack([s, act_probs])
        else:
            if add_dim:
                nn_inputs = s[np.newaxis, :]
            else:
                nn_inputs = s

        v = sess.run(self.v, {self.nn_inputs: nn_inputs})
        return v

    def get_weights_norm(self, sess):
        return sess.run(self.weights_norm, {})

class Simple_Agent(Agent):  # plays games with 2 actions, using a single parameter
    def __init__(self, env, learning_rate=0.001, n_units_critic=20, gamma=0.95, agent_idx=0,
                 critic_variant=Critic_Variant.INDEPENDENT, mean_theta=-2.0, std_theta=0.5):
        super().__init__(env, learning_rate, gamma, agent_idx)
        self.s = tf.placeholder(tf.float32, [1, env.NUM_STATES], "state")  # dummy variable
        self.a = tf.placeholder(tf.int32, None, "act")
        self.td_error = tf.placeholder(tf.float32, None, "td_error")  # TD_error

        with tf.variable_scope('Actor'):
            self.theta = tf.Variable(tf.random_normal([1], mean=mean_theta, stddev=std_theta))
            # self.theta = tf.Variable(tf.random_normal([1], mean=0, stddev=0.5))
            self.actions_prob = tf.expand_dims(tf.concat([1 - tf.sigmoid(self.theta), tf.sigmoid(self.theta)], 0), 0)
            # self.theta = tf.Variable(tf.random_normal([env.NUM_ACTIONS], mean=-2, stddev=0.5))
            # self.actions_prob = tf.expand_dims(tf.nn.softmax(self.theta), 0)

        with tf.variable_scope('exp_v'):
            # a_opt_val = tf.print("self.a", self.a)
            # with tf.control_dependencies([a_opt_val]):
            self.log_prob = tf.log(self.actions_prob[0, self.a])
            # log_prob_opt_val = tf.print("self.log_prob", self.log_prob)
            # actions_prob_opt_val = tf.print("self.actions_prob", self.actions_prob)
            # with tf.control_dependencies([log_prob_opt_val, actions_prob_opt_val]):
            self.g_log_pi = tf.gradients(self.log_prob, self.theta)
            self.exp_v = tf.reduce_mean(self.log_prob * self.td_error)

        with tf.variable_scope('trainActor'):
            self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(-self.exp_v)

        self.critic = Critic(env, n_units_critic, learning_rate, gamma, agent_idx, critic_variant,
                             weight_decay=0.0, std_theta=0.1)

        self.sess.run(tf.global_variables_initializer())

    def learn(self, s, a, r, s_, done=False, *args):
        if done:
            pass
        else:
            td_error, critic_loss = self.critic.learn(self.sess, s, r, s_, *args)
            feed_dict = {self.a: a, self.td_error: td_error}
            _, advantage = self.sess.run([self.train_op, self.exp_v], feed_dict)
        return critic_loss, advantage

    def __str__(self):
        return 'Simple_Agent_' + str(self.agent_idx)

    def calc_action_probs(self, s, add_dim=None):
        s = s[np.newaxis, :]
        probs = self.sess.run(self.actions_prob)
        return probs
        # return np.array([[0.75, 0.25]])

    def pass_agent_list(self, agent_list):
        self.critic.pass_agent_list(agent_list)

    def get_state_variable(self):
        return self.s

    def calc_g_log_pi(self, s, a):
        return self.sess.run(self.g_log_pi, feed_dict={self.s: s, self.a: a})

    def get_weights_norm(self):
        return self.sess.run(self.theta, {}), self.critic.get_weights_norm(self.sess)
