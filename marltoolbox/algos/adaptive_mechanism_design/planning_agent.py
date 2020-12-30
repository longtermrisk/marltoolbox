##########
# Code from: https://github.com/tobiasbaumann1/Adaptive_Mechanism_Design
##########

import logging
import numpy as np
import tensorflow as tf

logging.basicConfig(filename='Planning_Agent.log', level=logging.DEBUG, filemode='w')
from marltoolbox.algos.adaptive_mechanism_design.agent import Agent, convert_from_rllib_env_format
from tensorflow.python.ops import math_ops

RANDOM_SEED = 5
np.random.seed(RANDOM_SEED)
tf.set_random_seed(RANDOM_SEED)


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
                 value_fn_variant='exact', n_units=None, weight_decay=0.0, convert_a_to_one_hot=False):
        super().__init__(env, learning_rate, gamma)
        self.underlying_agents = underlying_agents
        self.log = []
        self.max_reward_strength = max_reward_strength
        n_players = len(underlying_agents)
        self.with_redistribution = with_redistribution
        self.value_fn_variant = value_fn_variant
        self.convert_a_to_one_hot = convert_a_to_one_hot
        self.env_name = env.NAME
        self.env = env

        self.s = tf.placeholder(tf.float32, [1, env.NUM_STATES], "state")
        if self.env_name == "CoinGame":
            self.a_players = tf.placeholder(tf.float32, [1, n_players, env.NUM_ACTIONS], "player_actions")
        else:
            self.a_players = tf.placeholder(tf.float32, [1, n_players], "player_actions")

        if value_fn_variant == 'exact':
            self.p_players = tf.placeholder(tf.float32, [1, n_players], "player_action_probs")
            self.a_plan = tf.placeholder(tf.float32, [2, 2],
                                         "conditional_planning_actions")  # works only for matrix games
        self.r_players = tf.placeholder(tf.float32, [1, n_players], "player_rewards")
        if self.env_name == "CoinGame":
            self.inputs = tf.concat([self.s, tf.reshape(self.a_players, (1, -1))], 1)
        else:
            self.inputs = tf.concat([self.s,self.a_players],1)

        with tf.variable_scope('Policy_p'):
            if self.env_name == "CoinGame":
                print("USE PLANNER NEW NETWORK")
                self.w_l0 = tf.Variable(tf.random_normal([env.NUM_STATES + 2 * env.NUM_ACTIONS,
                                                          n_units], stddev=0.1))
                self.b_l0 = tf.Variable(tf.random_normal([n_units], stddev=0.1))

                self.l0 = tf.nn.relu(tf.matmul(self.inputs, self.w_l0) + self.b_l0)
                self.w_pi0 = tf.Variable(tf.random_normal([n_units, n_players], stddev=0.1))
                self.b_pi0 = tf.Variable(tf.random_normal([n_players], stddev=0.1))

                self.l1 = tf.matmul(self.l0, self.w_pi0) + self.b_pi0

                var_list = [self.w_l0, self.b_l0, self.w_pi0, self.b_pi0]
                self.parameters = tf.concat(axis=0, values=[tf.reshape(v, [numel(v)]) for v in var_list])
                weigths_norm = math_ops.reduce_sum(self.parameters * self.parameters, None, keepdims=True)
                self.weigths_norm = tf.sqrt(tf.reduce_sum(weigths_norm))
            else:
                self.l1 = tf.layers.dense(
                    inputs=self.inputs,
                    units=n_players,  # 1 output per agent
                    activation=None,
                    kernel_initializer=tf.random_normal_initializer(0, .1),  # weights
                    bias_initializer=tf.random_normal_initializer(0, .1),  # biases
                    name='actions_planning'
                )

            if max_reward_strength is None:
                self.action_layer = self.l1
            else:
                # self.action_layer = tf.sigmoid(self.l1)
                op0 = tf.print("self.l1", self.l1)
                with tf.control_dependencies([op0]):
                    # self.action_layer = tf.nn.softmax(self.l1)
                    self.action_layer = tf.clip_by_value(
                        self.l1, clip_value_min=-self.max_reward_strength / 2,
                        clip_value_max=self.max_reward_strength / 2, name=None
                    )

        with tf.variable_scope('Vp'):
            # Vp is trivial to calculate in this special case
            if max_reward_strength is not None:
                self.vp = 2 * max_reward_strength * (self.action_layer - 0.5)
            else:
                self.vp = self.action_layer

        # TODO something to change here
        with tf.variable_scope('V_total'):
            if value_fn_variant == 'proxy':
                self.v = 2 * self.a_players - 1
            if value_fn_variant == 'estimated':
                # self.v = tf.reduce_sum(self.r_players) - 1.9
                self.v = tf.reduce_sum(self.r_players)
        with tf.variable_scope('cost_function'):
            if value_fn_variant == 'estimated':
                if self.env_name == "CoinGame":
                    self.g_log_pi = tf.placeholder(tf.float32, [env.NUM_STATES, n_players], "player_gradients")
                else:
                    self.g_log_pi = tf.placeholder(tf.float32, [1, n_players], "player_gradients")
            cost_list = []
            for underlying_agent in underlying_agents:
                # policy gradient theorem
                idx = underlying_agent.agent_idx
                if value_fn_variant == 'estimated':
                    g_log_pi_opt = tf.print("self.g_log_pi", tf.shape(self.g_log_pi))
                    with tf.control_dependencies([g_log_pi_opt]):
                        if self.env_name == "CoinGame":
                            self.g_Vp = self.g_log_pi[:, idx] * self.vp[:, idx]
                            self.g_V = self.g_log_pi[:, idx] * (self.v[:, idx]
                                                            if value_fn_variant == 'proxy'
                                                            else self.v)
                        else:
                            self.g_Vp = self.g_log_pi[0,idx] * self.vp[0,idx]
                            self.g_V = self.g_log_pi[0,idx] * (self.v[0,idx]
                                                               if value_fn_variant == 'proxy'
                                                               else self.v)
                if value_fn_variant == 'exact':
                    self.g_p = self.p_players[0, idx] * (1 - self.p_players[0, idx])
                    self.p_opp = self.p_players[0, 1 - idx]
                    self.g_Vp = self.g_p * tf.gradients(ys=self.vp[0, idx], xs=self.a_players)[0][0, idx]
                    self.g_V = self.g_p * (self.p_opp * (2 * env.R - env.T - env.S)
                                           + (1 - self.p_opp) * (env.T + env.S - 2 * env.P))

                # cost_list.append(- underlying_agent.learning_rate * tf.tensordot(self.g_Vp,self.g_V,1))
                cost_list.append(- underlying_agent.learning_rate * self.g_Vp * self.g_V)

            if with_redistribution:
                extra_loss = cost_param * tf.norm(self.vp - tf.reduce_mean(self.vp))
            else:
                extra_loss = cost_param * tf.norm(self.vp)

            extra_loss_opt = tf.print("extra_loss", extra_loss)
            cost_list_opt = tf.print("cost_list", cost_list)
            with tf.control_dependencies([extra_loss_opt, cost_list_opt]):
                if self.env_name == "CoinGame":
                    self.loss = tf.reduce_sum(tf.stack(cost_list)) + extra_loss + weight_decay * self.weigths_norm
                else:
                    self.loss = tf.reduce_sum(tf.stack(cost_list)) + extra_loss

        with tf.variable_scope('trainPlanningAgent'):
            self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss,
                                                                           var_list=tf.get_collection(
                                                                               tf.GraphKeys.GLOBAL_VARIABLES,
                                                                               scope='Policy_p'))

        self.sess.run(tf.global_variables_initializer())

    def learn(self, s, a_players, coin_game=False, env_rewards=None):
        s = s[np.newaxis, :]
        # r_players = np.asarray(self.env.calculate_payoffs(a_players))
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
            a_players_one_hot = self.action_to_one_hot(a_players)

        feed_dict = {self.s: s, self.a_players: a_players_one_hot[np.newaxis, ...],
                     self.r_players: r_players[np.newaxis, :]}
        if self.value_fn_variant == 'estimated':
            g_log_pi_list = []
            for underlying_agent in self.underlying_agents:
                idx = underlying_agent.agent_idx
                if self.env_name == "CoinGame":
                    g_log_pi_list.append(underlying_agent.calc_g_log_pi(s, a_players[idx])[0][0, ...])
                else:
                    g_log_pi_list.append(underlying_agent.calc_g_log_pi(s,a_players[idx]))
            if self.env_name == "CoinGame":
                g_log_pi_arr = np.stack(g_log_pi_list, axis=1)
            else:
                g_log_pi_arr = np.reshape(np.asarray(g_log_pi_list),[1,-1])
            feed_dict[self.g_log_pi] = g_log_pi_arr
        if self.value_fn_variant == 'exact':
            p_players_list = []
            for underlying_agent in self.underlying_agents:
                idx = underlying_agent.agent_idx
                p_players_list.append(underlying_agent.calc_action_probs(s)[0, -1])
            p_players_arr = np.reshape(np.asarray(p_players_list), [1, -1])
            feed_dict[self.p_players] = p_players_arr
            if "CoinGame" not in self.env_name:
                feed_dict[self.a_plan] = self.calc_conditional_planning_actions(s)
        self.sess.run([self.train_op], feed_dict)

        action, loss, g_Vp, g_V = self.sess.run([self.action_layer, self.loss,
                                                 self.g_Vp, self.g_V], feed_dict)
        if self.max_reward_strength is not None:
            action = 2 * self.max_reward_strength * (action - 0.5)
        print('Learning step')
        print('Planning_action: ' + str(action))
        if self.value_fn_variant == 'estimated':
            vp, v = self.sess.run([self.vp, self.v], feed_dict)
            print('Vp: ' + str(vp))
            print('V: ' + str(v))
        print('Gradient of V_p: ' + str(g_Vp))
        print('Gradient of V: ' + str(g_V))
        print('Loss: ' + str(loss))

    def get_log(self):
        return self.log

    def action_to_one_hot(self, a_players):
        print("a_players", a_players)
        a_players_one_hot = np.zeros((len(a_players), self.env.NUM_ACTIONS))
        for idx, act in enumerate(a_players.tolist()):
            print("idx,act", idx, act)
            a_players_one_hot[idx, act] = 1
        return a_players_one_hot

    def choose_action(self, s, a_players):
        print('Player actions: ' + str(a_players))
        s = s[np.newaxis, :]
        a_players = np.asarray(a_players)

        if self.convert_a_to_one_hot:
            a_players = self.action_to_one_hot(a_players)

        # a_plan = self.sess.run(self.action_layer, {self.s: s, self.a_players: a_players[np.newaxis,:]})[0,:]
        a_plan = self.sess.run(self.vp, {self.s: s, self.a_players: a_players[np.newaxis, ...]})[0, :]
        # if self.max_reward_strength is not None:
        #     a_plan = 2 * self.max_reward_strength * (a_plan - 0.5)
        print('Planning action: ' + str(a_plan))
        if "CoinGame" not in self.env_name:
            self.log.append(self.calc_conditional_planning_actions(s))
        return a_plan

    def calc_conditional_planning_actions(self,s):
        assert "CoinGame" not in self.env_name
        # Planning actions in each of the 4 cases: DD, CD, DC, CC
        if self.env.NAME == "CoinGame":
            print("USE PLANNER NEW PLANNING")

        else:
            a_plan_DD = self.sess.run(self.action_layer, {self.s: s, self.a_players: np.array([0,0])[np.newaxis,:]})
            a_plan_CD = self.sess.run(self.action_layer, {self.s: s, self.a_players: np.array([1,0])[np.newaxis,:]})
            a_plan_DC = self.sess.run(self.action_layer, {self.s: s, self.a_players: np.array([0,1])[np.newaxis,:]})
            a_plan_CC = self.sess.run(self.action_layer, {self.s: s, self.a_players: np.array([1,1])[np.newaxis,:]})
        l_temp = [a_plan_DD,a_plan_CD,a_plan_DC,a_plan_CC]
        if self.max_reward_strength is not None:
            l = [2 * self.max_reward_strength * (a_plan_X[0,0]-0.5) for a_plan_X in l_temp]
        else:
            l = [a_plan_X[0,0] for a_plan_X in l_temp]
        if self.with_redistribution:
            if self.max_reward_strength is not None:
                l2 = [2 * self.max_reward_strength * (a_plan_X[0,1]-0.5) for a_plan_X in l_temp]
            else:
                l2 = [a_plan_X[0,1] for a_plan_X in l_temp]
            l = [0.5 * (elt[0]-elt[1]) for elt in zip(l,l2)]
        return np.transpose(np.reshape(np.asarray(l),[2,2]))
