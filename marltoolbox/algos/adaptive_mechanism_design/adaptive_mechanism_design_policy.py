##########
# Code from: https://github.com/tobiasbaumann1/Adaptive_Mechanism_Design
##########

import logging

logging.basicConfig(filename='Planning_Agent.log', level=logging.DEBUG, filemode='w')

# RANDOM_SEED = 5
# np.random.seed(RANDOM_SEED)
# tf.set_random_seed(RANDOM_SEED)
#
# class AdaptiveMeschanismDesignTfPolicy(TfPolicy):
#     def __init__(self, env, underlying_agents, learning_rate=0.01,
#         gamma = 0.95, max_reward_strength = None, cost_param = 0, with_redistribution = False,
#         value_fn_variant = 'exact'):
#         super().__init__(env, learning_rate, gamma)
#         self.underlying_agents = underlying_agents
#         self.log = []
#         self.max_reward_strength = max_reward_strength
#         n_players = len(underlying_agents)
#         self.with_redistribution = with_redistribution
#         self.value_fn_variant = value_fn_variant
#
#         self.s = tf.placeholder(tf.float32, [1, env.NUM_STATES], "state")
#         self.a_players = tf.placeholder(tf.float32, [1, n_players], "player_actions")
#         if value_fn_variant == 'exact':
#             self.p_players = tf.placeholder(tf.float32, [1, n_players], "player_action_probs")
#             self.a_plan = tf.placeholder(tf.float32, [2, 2], "conditional_planning_actions") # works only for matrix games
#         self.r_players = tf.placeholder(tf.float32, [1, n_players], "player_rewards")
#         self.inputs = tf.concat([self.s,self.a_players],1)
#
#         with tf.variable_scope('Policy_p'):
#             self.l1 = tf.layers.dense(
#                 inputs=self.inputs,
#                 units=n_players,    # 1 output per agent
#                 activation=None,
#                 kernel_initializer=tf.random_normal_initializer(0, .1),  # weights
#                 bias_initializer=tf.random_normal_initializer(0, .1),  # biases
#                 name='actions_planning'
#             )
#
#             if max_reward_strength is None:
#                 self.action_layer = self.l1
#             else:
#                 self.action_layer = tf.sigmoid(self.l1)
#
#         with tf.variable_scope('Vp'):
#             # Vp is trivial to calculate in this special case
#             if max_reward_strength is not None:
#                 self.vp = 2 * max_reward_strength * (self.action_layer - 0.5)
#             else:
#                 self.vp = self.action_layer
#
#         with tf.variable_scope('V_total'):
#             if value_fn_variant == 'proxy':
#                 self.v = 2 * self.a_players - 1
#             if value_fn_variant == 'estimated':
#                 self.v = tf.reduce_sum(self.r_players) - 1.9
#         with tf.variable_scope('cost_function'):
#             if value_fn_variant == 'estimated':
#                 self.g_log_pi = tf.placeholder(tf.float32, [1, n_players], "player_gradients")
#             cost_list = []
#             for underlying_agent in underlying_agents:
#                 # policy gradient theorem
#                 idx = underlying_agent.agent_idx
#                 if value_fn_variant == 'estimated':
#                     self.g_Vp = self.g_log_pi[0,idx] * self.vp[0,idx]
#                     self.g_V = self.g_log_pi[0,idx] * (self.v[0,idx] if value_fn_variant == 'proxy' else self.v)
#                 if value_fn_variant == 'exact':
#                     self.g_p = self.p_players[0,idx] * (1-self.p_players[0,idx])
#                     self.p_opp = self.p_players[0,1-idx]
#                     self.g_Vp = self.g_p * tf.gradients(ys = self.vp[0,idx],xs = self.a_players)[0][0,idx]
#                     self.g_V = self.g_p * (self.p_opp * (2 * env.R - env.T - env.S)
#                         + (1-self.p_opp) * (env.T + env.S - 2 * env.P))
#
#                 #cost_list.append(- underlying_agent.learning_rate * tf.tensordot(self.g_Vp,self.g_V,1))
#                 cost_list.append(- underlying_agent.learning_rate * self.g_Vp * self.g_V)
#
#             if with_redistribution:
#                 extra_loss = cost_param * tf.norm(self.vp-tf.reduce_mean(self.vp))
#             else:
#                 extra_loss = cost_param * tf.norm(self.vp)
#             self.loss = tf.reduce_sum(tf.stack(cost_list)) + extra_loss
#
#         with tf.variable_scope('trainPlanningAgent'):
#             self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss,
#                 var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Policy_p'))
#
#         self.sess.run(tf.global_variables_initializer())
#
#     def learn(self, s, a_players):
#         s = s[np.newaxis,:]
#         # r_players = np.asarray(self.env.calculate_payoffs(a_players))
#         r_players_rllib_format = self.env._compute_rewards(*a_players)
#         r_players = convert_from_rllib_env_format(r_players_rllib_format, self.env.players_ids)
#         a_players = np.asarray(a_players)
#         feed_dict = {self.s: s, self.a_players: a_players[np.newaxis,:],
#                     self.r_players: r_players[np.newaxis,:]}
#         if self.value_fn_variant == 'estimated':
#             g_log_pi_list = []
#             for underlying_agent in self.underlying_agents:
#                 idx = underlying_agent.agent_idx
#                 g_log_pi_list.append(underlying_agent.calc_g_log_pi(s,a_players[idx]))
#             g_log_pi_arr = np.reshape(np.asarray(g_log_pi_list),[1,-1])
#             feed_dict[self.g_log_pi] = g_log_pi_arr
#         if self.value_fn_variant == 'exact':
#             p_players_list = []
#             for underlying_agent in self.underlying_agents:
#                 idx = underlying_agent.agent_idx
#                 p_players_list.append(underlying_agent.calc_action_probs(s)[0,-1])
#             p_players_arr = np.reshape(np.asarray(p_players_list),[1,-1])
#             feed_dict[self.p_players] = p_players_arr
#             feed_dict[self.a_plan] = self.calc_conditional_planning_actions(s)
#         self.sess.run([self.train_op], feed_dict)
#
#         action,loss,g_Vp,g_V = self.sess.run([self.action_layer,self.loss,
#             self.g_Vp,self.g_V], feed_dict)
#         logging.info('Learning step')
#         logging.info('Planning_action: ' + str(action))
#         if self.value_fn_variant == 'estimated':
#             vp,v = self.sess.run([self.vp,self.v],feed_dict)
#             logging.info('Vp: ' + str(vp))
#             logging.info('V: ' + str(v))
#         logging.info('Gradient of V_p: ' + str(g_Vp))
#         logging.info('Gradient of V: ' + str(g_V))
#         logging.info('Loss: ' + str(loss))
#
#     def get_log(self):
#         return self.log
#
#     def choose_action(self, s, a_players):
#         logging.info('Player actions: ' + str(a_players))
#         s = s[np.newaxis, :]
#         a_players = np.asarray(a_players)
#         a_plan = self.sess.run(self.action_layer, {self.s: s, self.a_players: a_players[np.newaxis,:]})[0,:]
#         if self.max_reward_strength is not None:
#             a_plan = 2 * self.max_reward_strength * (a_plan - 0.5)
#         logging.info('Planning action: ' + str(a_plan))
#         self.log.append(self.calc_conditional_planning_actions(s))
#         return a_plan
#
#     def calc_conditional_planning_actions(self,s):
#         # Planning actions in each of the 4 cases: DD, CD, DC, CC
#         a_plan_DD = self.sess.run(self.action_layer, {self.s: s, self.a_players: np.array([0,0])[np.newaxis,:]})
#         a_plan_CD = self.sess.run(self.action_layer, {self.s: s, self.a_players: np.array([1,0])[np.newaxis,:]})
#         a_plan_DC = self.sess.run(self.action_layer, {self.s: s, self.a_players: np.array([0,1])[np.newaxis,:]})
#         a_plan_CC = self.sess.run(self.action_layer, {self.s: s, self.a_players: np.array([1,1])[np.newaxis,:]})
#         l_temp = [a_plan_DD,a_plan_CD,a_plan_DC,a_plan_CC]
#         if self.max_reward_strength is not None:
#             l = [2 * self.max_reward_strength * (a_plan_X[0,0]-0.5) for a_plan_X in l_temp]
#         else:
#             l = [a_plan_X[0,0] for a_plan_X in l_temp]
#         if self.with_redistribution:
#             if self.max_reward_strength is not None:
#                 l2 = [2 * self.max_reward_strength * (a_plan_X[0,1]-0.5) for a_plan_X in l_temp]
#             else:
#                 l2 = [a_plan_X[0,1] for a_plan_X in l_temp]
#             l = [0.5 * (elt[0]-elt[1]) for elt in zip(l,l2)]
#         return np.transpose(np.reshape(np.asarray(l),[2,2]))
