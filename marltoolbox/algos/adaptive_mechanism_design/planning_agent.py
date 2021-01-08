##########
# Code from: https://github.com/tobiasbaumann1/Adaptive_Mechanism_Design
##########
import copy

import logging
import numpy as np
import tensorflow as tf

logging.basicConfig(filename='Planning_Agent.log', level=logging.DEBUG, filemode='w')
from marltoolbox.algos.adaptive_mechanism_design.agent import Agent, convert_from_rllib_env_format
from tensorflow.python.ops import math_ops


def var_shape(x):
    out = x.get_shape().as_list()
    return out


def intprod(x):
    return int(np.prod(x))


def numel(x):
    return intprod(var_shape(x))


class Planning_Agent(Agent):
    def __init__(self, env, underlying_agents, learning_rate=0.01,
                 gamma=0.95, max_reward_strength=None, cost_param=0, with_redistribution=False,
                 value_fn_variant='exact', n_units=None, weight_decay=0.0, convert_a_to_one_hot=False, mean_theta=0.0,
                 loss_mul_planner=1.0, training=True, std_theta=0.1, planner_clip_norm=0.5, normalize_planner=False,
                 add_state_grad=False, planner_momentum=0.9, use_adam_optimizer=True, use_softmax_hot=True):
        super().__init__(env, learning_rate, gamma)
        self.underlying_agents = underlying_agents
        self.log = []
        self.max_reward_strength = max_reward_strength
        n_players = len(underlying_agents)
        self.with_redistribution = with_redistribution
        self.value_fn_variant = value_fn_variant
        self.convert_a_to_one_hot = convert_a_to_one_hot
        # self.convert_a_to_one_hot = False
        self.env_name = env.NAME
        self.env = env
        self.loss_mul_planner = loss_mul_planner

        with tf.variable_scope('Planner'):
            self.s = tf.placeholder(tf.float32, [1, env.NUM_STATES], "state_pl")
            self.a_players = tf.placeholder(tf.float32, [1, n_players], "player_actions")
            # self.a_players_one_hot = tf.one_hot(tf.cast(self.a_players, tf.int32), self.env.NUM_ACTIONS, dtype=tf.float32)
            # self.a_players_one_hot = tf.nn.softmax(1 * tf.stack([
            #     -tf.math.abs(-self.a_players),
            #     -tf.math.abs(1-self.a_players)], 2))
            if "CoinGame" in self.env_name:
                # self.a_players_one_hot = tf.stack([
                #     -tf.math.abs(tf.clip_by_value(-self.a_players,
                #                                   clip_value_min=-1.0, clip_value_max=0.0, name=None)),
                #     -tf.math.abs(tf.clip_by_value(1 - self.a_players,
                #                                   clip_value_min=-1.0, clip_value_max=0.0, name=None)),
                #     -tf.math.abs(tf.clip_by_value(2 - self.a_players,
                #                                   clip_value_min=-1.0, clip_value_max=0.0, name=None)),
                #     -tf.math.abs(tf.clip_by_value(3 - self.a_players,
                #                                   clip_value_min=-1.0, clip_value_max=0.0, name=None)),
                # ], 2)
                values = tf.stack([
                    -tf.math.abs(-self.a_players),
                    -tf.math.abs(1 - self.a_players),
                    -tf.math.abs(2 - self.a_players),
                    -tf.math.abs(3 - self.a_players),
                ], 2)
                values = tf.where(tf.equal(values, -2), values, values+1)
                values = tf.where(tf.equal(values, -3), values, values+2)
                if use_softmax_hot:
                    self.a_players_one_hot = tf.nn.softmax(values)
                else:
                    self.a_players_one_hot = values + 1

            else:
                # self.a_players_one_hot = tf.nn.softmax( tf.stack([
                #     -tf.math.abs(-self.a_players),
                #     -tf.math.abs(1-self.a_players)], 2))
                # self.a_players_one_hot = tf.stack([
                #     -tf.math.abs(tf.clip_by_value(-self.a_players, clip_value_min=-1.0, clip_value_max=0.0, name=None)),
                #     -tf.math.abs(tf.clip_by_value(1-self.a_players, clip_value_min=-1.0, clip_value_max=0.0, name=None))], 2)
                values = tf.stack([
                    -tf.math.abs(-self.a_players),
                    -tf.math.abs(1 - self.a_players),
                ], 2)
                values = tf.where(tf.equal(values, -1), values, values)
                self.a_players_one_hot = tf.nn.softmax(values)
            # opt_1 = tf.print("self.a_players", self.a_players)
            # opt_2 = tf.print("self.a_players_one_hot", self.a_players_one_hot)
            # with tf.control_dependencies([opt_1, opt_2]):
            #     self.a_players_one_hot = self.a_players_one_hot[:,:,-1]
            self.a_players_one_hot_reshape = tf.reshape(self.a_players_one_hot, (1, -1))

            if value_fn_variant == 'exact':
                if self.convert_a_to_one_hot:
                    self.p_players = tf.placeholder(tf.float32, [1, n_players, env.NUM_ACTIONS], "player_action_probs")
                else:
                    self.p_players = tf.placeholder(tf.float32, [1, n_players], "player_action_probs")
                self.a_plan = tf.placeholder(tf.float32, [2, 2], "conditional_planning_actions")  # works only for matrix games
            self.r_players = tf.placeholder(tf.float32, [1, n_players], "player_rewards")

            if self.convert_a_to_one_hot:
                self.inputs = tf.concat([self.s, self.a_players_one_hot_reshape], 1)
            else:
                self.inputs = tf.concat([self.s, self.a_players], 1)

            if normalize_planner:
                self.inputs = self.inputs-0.5

            with tf.variable_scope('Policy_p'):

                if self.convert_a_to_one_hot:
                    ma_action_space_dim = 2 * env.NUM_ACTIONS
                else:
                    ma_action_space_dim = env.NUM_ACTIONS

                if not isinstance(n_units, list):
                    units = [env.NUM_STATES + ma_action_space_dim, n_units, n_players]
                else:
                    units = [env.NUM_STATES + ma_action_space_dim] + n_units + [n_players]
                print("units", units)
                var_list = []
                input_ = self.inputs
                for i in range(len(units)):
                    with tf.variable_scope("planner_layer_{}".format(i)):
                        n_in = units[i]
                        n_out = units[i + 1]
                        print("i", i)
                        print("n_in", n_in)
                        print("n_out", n_out)
                        if i + 1 == len(units) - 1:
                            break
                        w_l1 = tf.Variable(tf.random_normal([n_in, n_out], mean=0.0, stddev=std_theta))
                        b_l1 = tf.Variable(tf.random_normal([n_out], mean=0.0, stddev=std_theta))
                        # l1 = tf.nn.relu(tf.matmul(input_, w_l1) + b_l1)
                        l1 = tf.nn.leaky_relu(tf.matmul(input_, w_l1) + b_l1)
                        var_list.extend([w_l1, b_l1])
                        # l1 = tf.nn.relu(tf.matmul(input_, w_l1))
                        # var_list.extend([w_l1])
                        # l1 = tf.compat.v1.layers.batch_normalization(l1, training=training)
                        input_ = l1

                self.w_pi0 = tf.Variable(tf.random_normal([n_in, n_out], mean=0.0, stddev=std_theta))
                self.b_pi0 = tf.Variable(tf.random_normal([n_out], mean=mean_theta, stddev=std_theta))
                self.l1 = tf.matmul(input_, self.w_pi0) + self.b_pi0
                var_list.extend([self.w_pi0, self.b_pi0])

                self.parameters = tf.concat(axis=0, values=[tf.reshape(v, [numel(v)]) for v in var_list])
                weights_norm = math_ops.reduce_sum(self.parameters * self.parameters, None, keepdims=True)
                self.weights_norm = tf.sqrt(tf.reduce_sum(weights_norm))
                # else:
                #     self.l1 = tf.layers.dense(
                #         inputs=self.inputs,
                #         units=n_players,  # 1 output per agent
                #         activation=None,
                #         kernel_initializer=tf.random_normal_initializer(0, std_theta),  # weights
                #         bias_initializer=tf.random_normal_initializer(0, std_theta),  # biases
                #         name='actions_planning'
                #     )
                #     self.weights_norm = tf.Variable(tf.random_normal([n_players], stddev=std_theta))

                if max_reward_strength is None:
                    self.action_layer = self.l1
                else:
                    self.action_layer = tf.sigmoid(self.l1)

            with tf.variable_scope('Vp'):
                if max_reward_strength is not None:
                    self.vp = 2 * max_reward_strength * (self.action_layer - 0.5)
                else:
                    self.vp = self.action_layer

            # TODO something to change here
            with tf.variable_scope('V_total'):
                if value_fn_variant == 'proxy':
                    self.v = 2 * self.a_players - 1
                # if value_fn_variant == 'estimated':
                if value_fn_variant == 'estimated' or value_fn_variant == 'exact':
                    if "CoinGame" in self.env_name:
                        self.v = tf.reduce_sum(self.r_players)
                        # self.v = tf.reduce_sum(self.r_players) + 2
                    else:
                        self.v = tf.reduce_sum(self.r_players) - 1.9
                        # self.v = tf.reduce_sum(self.r_players) - 4
                # if value_fn_variant == 'exact':
                #     self.v = tf.placeholder(tf.float32, [1, n_players], "player_values")
            with tf.variable_scope('cost_function'):
                if value_fn_variant == 'estimated':
                    # if "CoinGame" in self.env_name:
                    self.g_log_pi = tf.placeholder(tf.float32, [env.NUM_STATES, n_players], "player_gradients")
                    # else:
                    #     self.g_log_pi = tf.placeholder(tf.float32, [1, n_players], "player_gradients")
                cost_list = []
                for underlying_agent in underlying_agents:
                    # policy gradient theorem
                    idx = underlying_agent.agent_idx
                    if value_fn_variant == 'estimated':
                        if "CoinGame" in self.env_name: # or True:
                            self.g_Vp = self.g_log_pi[:, idx] * self.vp[:, idx]
                            self.g_V = self.g_log_pi[:, idx] * (self.v[:, idx]
                                                                if value_fn_variant == 'proxy'
                                                                else self.v)
                        else:
                            self.g_Vp = self.g_log_pi[0, idx] * self.vp[0, idx]
                            self.g_V = self.g_log_pi[0, idx] * (self.v[0, idx]
                                                                if value_fn_variant == 'proxy'
                                                                else self.v)
                    if value_fn_variant == 'exact':
                        # if "CoinGame" in self.env_name:
                        #     # self.g_p = self.p_players[0, idx] * (1 - self.p_players[0, idx])
                        #     # 4 actions
                        #     # self.g_p = self.p_players[0, idx, 0] * self.p_players[0, idx, 1] * \
                        #     #            self.p_players[0, idx, 2] *self.p_players[0, idx, 3]
                        #     # 4 actionsd 1 used
                        #     # self.g_p = self.p_players[0, idx, self.a_players[0,idx]] * \
                        #     #            (1 - tf.reduce_sum(self.p_players[0, idx,:]) + self.p_players[0, idx, self.a_players[0,idx]])
                        #     self.g_p = (1 - self.p_players[0, idx, self.a_players[0, idx]]) * \
                        #                (tf.reduce_sum(self.p_players[0, idx, :]) - self.p_players[
                        #                    0, idx, self.a_players[0, idx]])
                        #
                        #     # print_opt = tf.print("tf.gradients(ys=self.vp[0, idx], xs=self.inputs)",
                        #     #                      tf.shape(tf.gradients(
                        #     #     ys=self.vp[0, idx], xs=self.inputs)))
                        #     # with tf.control_dependencies([print_opt]):
                        #     # self.g_Vp = self.g_p * tf.gradients(ys=self.vp[0, idx], xs=self.a_players)[0][0, idx]
                        #     # self.g_Vp = self.g_p * tf.gradients(ys=self.vp[0, idx], xs=self.inputs)
                        #     self.g_Vp = self.g_p * tf.reduce_sum(tf.gradients(ys=self.vp[0, idx], xs=self.inputs))
                        #     # self.g_V = self.g_p * (tf.reduce_sum(self.v) - 2*tf.reduce_sum(self.vp))
                        #     opt = tf.print("tf.reduce_sum(self.v)", tf.reduce_sum(self.v))
                        #     opt_g_p = tf.print("self.g_p", tf.reduce_sum(self.g_p))
                        #     opt_g_Vp = tf.print("self.g_Vp", tf.reduce_sum(self.g_Vp))
                        #     with tf.control_dependencies([opt, opt_g_p, opt_g_Vp]):
                        #         self.g_V = self.g_p * tf.reduce_sum(self.v)
                        # else:
                        # coop_act = -1
                        act_idx = tf.cast(self.a_players[0, idx], tf.int32)

                        # self.g_p = self.p_players[0, idx] * (1 - self.p_players[0, idx])
                        # self.p_opp = self.p_players[0, 1 - idx]

                        if self.convert_a_to_one_hot:
                            self.g_p = self.p_players[0, idx, act_idx] * (1 - self.p_players[0, idx, act_idx])

                            self.p_opp = self.p_players[0, 1 - idx, act_idx]
                            # opt_grad = tf.print("idx", idx,"convert self.g_p ", self.g_p , "self.p_opp", self.p_opp)
                            # opt_ac = tf.print("idx", idx,"convert self.a_players_one_hot", self.a_players_one_hot,
                            #                   "self.a_players", self.a_players)
                            # with tf.control_dependencies([opt_grad, opt_ac]):
                            grad = tf.gradients(ys=self.vp[0, idx], xs=self.a_players)
                            if add_state_grad:
                                grad_s = tf.gradients(ys=self.vp[0, idx], xs=self.s)
                                # grad = tf.gradients(ys=self.vp[0, idx], xs=self.a_players_one_hot)
                            # opt_grad = tf.print("idx", idx,"convert grad", grad, "self.vp", self.vp, "self.l1", self.l1)
                            # with tf.control_dependencies([opt_grad]):
                            self.g_Vp = self.g_p * grad[0][0, idx]
                            if add_state_grad:
                                self.g_Vp += self.g_p * tf.reduce_sum(grad_s)
                                # self.g_Vp = self.g_p * tf.reduce_sum(grad[0][0, idx, :])
                                # self.g_Vp = self.g_p * tf.reduce_sum(grad[0][0, 2*idx:2*(idx+1)])
                                # self.g_Vp = self.g_p * grad[0][0, idx, coop_act]
                                # self.g_Vp = self.g_p * grad[0][0, idx, tf.cast(self.a_players[0, idx], tf.int32)]
                                # self.g_Vp = self.g_Vp - \
                                #              self.g_p * grad[0][0, idx, tf.cast(1-self.a_players[0, idx], tf.int32)]
                                # self.g_Vp = - self.g_Vp + \
                                #              self.g_p * grad[0][0, idx, tf.cast(1-self.a_players[0, idx], tf.int32)]

                            # opt_grad = tf.print("idx", idx, "convert self.g_p ", self.g_p, "self.p_opp", self.p_opp)
                            # with tf.control_dependencies([opt_grad]):
                            #     grad = tf.gradients(ys=self.vp[0, idx], xs=self.a_players)
                            # opt_grad = tf.print("idx", idx, "convert grad", grad, "self.vp", self.vp, "self.l1",
                            #                     self.l1)
                            # with tf.control_dependencies([opt_grad]):
                            #     self.g_Vp = self.g_p * grad[0][0, idx]
                        else:
                            self.g_p = self.p_players[0, idx] * (1 - self.p_players[0, idx])
                            self.p_opp = self.p_players[0, 1 - idx]
                            # self.g_Vp = self.g_p * tf.gradients(ys=self.vp[0, idx], xs=self.a_players)[0][0, idx]
                            # opt_grad = tf.print("idx", idx,"not convert self.g_p ", self.g_p , "self.p_opp", self.p_opp)
                            # with tf.control_dependencies([opt_grad]):
                            grad = tf.gradients(ys=self.vp[0, idx], xs=self.a_players)
                            # opt_grad = tf.print("idx", idx,"not convert grad", grad, "self.vp", self.vp, "self.l1", self.l1)
                            # with tf.control_dependencies([opt_grad]):
                            self.g_Vp = self.g_p * grad[0][0, idx]
                                # grad = tf.gradients(ys=self.vp[0, idx], xs=self.inputs)
                                # self.g_Vp = self.g_p * tf.reduce_sum(grad)

                        if "CoinGame" in self.env_name:
                             self.g_V = self.g_p * tf.reduce_sum(self.v)
                        else:
                            self.g_V = self.g_p * (self.p_opp * (2 * env.R - env.T - env.S)
                                               + (1 - self.p_opp) * (env.T + env.S - 2 * env.P))

                    # opt_a = tf.print("self.a_players", self.a_players)
                    # opt_a_hot = tf.print("self.a_players_one_hot", self.a_players_one_hot)
                    # if self.convert_a_to_one_hot:
                    #     opt_p_p = tf.print("self.p_players[0, idx, coop_act]", self.p_players[0, idx, coop_act])
                    # else:
                    #     opt_p_p = tf.print("self.a_players_one_hot", self.a_players_one_hot)
                    # opt = tf.print("idx", idx, "self.g_p self.p_opp self.g_Vp self.g_V",
                    #                self.g_p, self.p_opp,
                    #                self.g_Vp,
                    #                self.g_V)
                    # opt_grad = tf.print("grad", tf.shape(grad))
                    # with tf.control_dependencies([opt,opt_grad, opt_a_hot, opt_a]):
                    cost_list.append(- underlying_agent.learning_rate * self.g_Vp * self.g_V)

                if with_redistribution:
                    self.extra_loss = cost_param * tf.norm(self.vp - tf.reduce_mean(self.vp))
                else:
                    # self.extra_loss = cost_param * tf.norm(self.vp)
                    self.extra_loss = cost_param * tf.norm(self.vp) * tf.norm(self.vp)

                self.cost = tf.reduce_sum(tf.stack(cost_list))
                if planner_clip_norm is not None:
                    self.cost = tf.clip_by_norm(self.cost, planner_clip_norm, axes=None, name=None)

                # self.loss = self.loss_mul_planner * (self.cost + self.extra_loss)
                self.loss = (self.cost + self.extra_loss)
                if weight_decay > 0.0:
                     self.loss += weight_decay * self.weights_norm

            with tf.variable_scope('trainPlanningAgent'):
                #AdamOptimizer
                if use_adam_optimizer:
                    self.train_op = tf.train.AdamOptimizer(self.loss_mul_planner *learning_rate).minimize(self.loss,
                                                                               var_list=tf.get_collection(
                                                                                   tf.GraphKeys.GLOBAL_VARIABLES,
                                                                                   scope='Planner/Policy_p'))
                else:
                    self.train_op = tf.train.MomentumOptimizer(self.loss_mul_planner *learning_rate,
                                                           momentum=planner_momentum).minimize(self.loss,
                                                                               var_list=tf.get_collection(
                                                                                   tf.GraphKeys.GLOBAL_VARIABLES,
                                                                                   scope='Planner/Policy_p'))
                # update_ops = tf.compat.v1.get_collection(tf.GraphKeys.UPDATE_OPS, scope='Planner/Policy_p')
                # self.train_op = tf.group([self.train_op, update_ops])
            self.sess.run(tf.global_variables_initializer())

    def get_weigths(self):
        return self.sess.run(self.parameters, {})

    def learn(self, s, a_players, coin_game=False, env_rewards=None):
        s = s[np.newaxis, :]
        if env_rewards is None:
            if coin_game:
                # TODO remove hardcoded policy_id
                actions = {"player_red": a_players[0], "player_blue": a_players[1]}
                r_players_rllib_format = self.env._compute_rewards(s, actions)
            else:
                r_players_rllib_format = self.env._compute_rewards(*a_players)
            r_players = convert_from_rllib_env_format(r_players_rllib_format, self.env.players_ids)
        else:
            r_players = env_rewards
        a_players = np.asarray(a_players)

        if self.convert_a_to_one_hot:
            a_players_one_hot = self.np_action_to_one_hot(a_players)
        #     feed_dict = {self.s: s,
        #                  # self.a_players_one_hot: a_players_one_hot[np.newaxis, ...],
        #                  self.a_players: a_players[np.newaxis, ...],
        #                  self.r_players: r_players[np.newaxis, :]}
        # else:
        feed_dict = {self.s: s,
                     self.a_players: a_players[np.newaxis, ...],
                     self.r_players: r_players[np.newaxis, :]}
        if self.value_fn_variant == 'estimated':
            g_log_pi_list = []
            for underlying_agent in self.underlying_agents:
                idx = underlying_agent.agent_idx
                # if "CoinGame" in self.env_name:
                g_log_pi_list.append(underlying_agent.calc_g_log_pi(s, a_players_one_hot[idx])[0][0, ...])
                # else:
                #     g_log_pi_list.append(underlying_agent.calc_g_log_pi(s, a_players[idx]))
            # if "CoinGame" in self.env_name:
            g_log_pi_arr = np.stack(g_log_pi_list, axis=1)
            # else:
            #     g_log_pi_arr = np.reshape(np.asarray(g_log_pi_list), [1, -1])
            # print("g_log_pi_arr", g_log_pi_arr.shape)
            feed_dict[self.g_log_pi] = g_log_pi_arr
        if self.value_fn_variant == 'exact':
            p_players_list = []
            v_list = []
            for underlying_agent in self.underlying_agents:
                idx = underlying_agent.agent_idx
                if self.convert_a_to_one_hot:
                    p_players_list.append(underlying_agent.calc_action_probs(s, add_dim=False))
                    # p_players_list.append(underlying_agent.calc_action_probs(s, add_dim=False)[0, -1])
                else:
                    p_players_list.append(underlying_agent.calc_action_probs(s)[0, -1])  # Only 2 actions
                # if "CoinGame" in self.env_name:
                #     v_list.append(underlying_agent.calcul_value(s, add_dim=False))
            if self.convert_a_to_one_hot:
                p_players_arr = np.stack(p_players_list, axis=1)
            else:
                p_players_arr = np.reshape(np.asarray(p_players_list), [1, -1])
            feed_dict[self.p_players] = p_players_arr
            # if "CoinGame" in self.env_name:
            #     v_players_arr = np.reshape(np.asarray(v_list), [1, -1])
            #     feed_dict[self.v] = v_players_arr
            # if "CoinGame" not in self.env_name:
            #     feed_dict[self.a_plan] = self.calc_conditional_planning_actions(s)
        self.sess.run([self.train_op], feed_dict)

        action, loss, g_Vp, g_V, cost, extra_loss, l1 = self.sess.run([self.vp, self.loss,
                                                 self.g_Vp, self.g_V,
                                                 self.cost, self.extra_loss, self.l1], feed_dict)
        # print('Learning step')
        # print('Planning_action: ' + str(action))
        vp, v = self.sess.run([self.vp, self.v], feed_dict)
        # print('Vp: ' + str(vp))
        # print('V: ' + str(v))
        # print('Loss: ' + str(loss))

        return action, loss, g_Vp, g_V, vp, v, r_players, cost, extra_loss, l1

    def get_log(self):
        return self.log

    def np_action_to_one_hot(self, a_players):
        a_players_one_hot = np.zeros((len(a_players), self.env.NUM_ACTIONS))
        for idx, act in enumerate(a_players.tolist()):
            a_players_one_hot[idx, act] = 1
        return a_players_one_hot

    def choose_action(self, s, a_players):
        # print('Player actions: ' + str(a_players))
        s = s[np.newaxis, :]
        a_players = np.asarray(a_players)

        a_plan = self.sess.run(self.vp, {self.s: s,
                                     self.a_players: a_players[np.newaxis, ...]})[0, :]

        # print('Planning action: ' + str(a_plan))
        # if "CoinGame" not in self.env_name:
        #     self.log.append(self.calc_conditional_planning_actions(s))
        return a_plan

    def calc_conditional_planning_actions(self, s):
        assert "CoinGame" not in self.env_name
        # Planning actions in each of the 4 cases: DD, CD, DC, CC
        a_plan_DD = self.sess.run(self.action_layer, {self.s: s, self.a_players: np.array([0, 0])[np.newaxis, :]})
        a_plan_CD = self.sess.run(self.action_layer, {self.s: s, self.a_players: np.array([1, 0])[np.newaxis, :]})
        a_plan_DC = self.sess.run(self.action_layer, {self.s: s, self.a_players: np.array([0, 1])[np.newaxis, :]})
        a_plan_CC = self.sess.run(self.action_layer, {self.s: s, self.a_players: np.array([1, 1])[np.newaxis, :]})
        l_temp = [a_plan_DD, a_plan_CD, a_plan_DC, a_plan_CC]
        if self.max_reward_strength is not None:
            l = [2 * self.max_reward_strength * (a_plan_X[0, 0] - 0.5) for a_plan_X in l_temp]
        else:
            l = [a_plan_X[0, 0] for a_plan_X in l_temp]
        if self.with_redistribution:
            if self.max_reward_strength is not None:
                l2 = [2 * self.max_reward_strength * (a_plan_X[0, 1] - 0.5) for a_plan_X in l_temp]
            else:
                l2 = [a_plan_X[0, 1] for a_plan_X in l_temp]
            l = [0.5 * (elt[0] - elt[1]) for elt in zip(l, l2)]
        return np.transpose(np.reshape(np.asarray(l), [2, 2]))
