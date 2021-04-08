##########
# Code modified from: https://github.com/alshedivat/lola
##########

"""
Training function for the Coin Game.
"""
import copy
import json
import os
from collections import Iterable
from collections import deque

import numpy as np
import tensorflow as tf
from ray import tune

from marltoolbox.algos.lola.corrections import corrections_func, simple_actor_training_func
from marltoolbox.algos.lola.networks import Pnetwork, DQNAgent
from marltoolbox.algos.lola.utils import get_monte_carlo, make_cube
from marltoolbox.envs.vectorized_coin_game import VectorizedCoinGame, AsymVectorizedCoinGame
from marltoolbox.utils.full_epi_logger import FullEpisodeLogger

PLOT_KEYS = [
    "player_1_loss",
    "player_2_loss",
    "v_0_log",
    "v_1_log",
    "entropy_p_0",
    "entropy_p_1",
    "actor_loss_0",
    "actor_loss_1",
    "parameters_norm_0",
    "parameters_norm_1",
    "second_order0_sum",
    "second_order1_sum",
    "player_1_update_sum",
    "player_2_update_sum",
    "actor_grad_sum_0",
    "actor_grad_sum_1",
    "lr_decay_ratio",
    "pg_expl_player_loss",
    "pg_expl_v_log",
    "pg_expl_entropy",
    "pg_expl_actor_loss",
    "pg_expl_parameters_norm",
    "pg_expl_update_sum",
    "pg_expl_actor_grad_sum"
]

PLOT_ASSEMBLAGE_TAGS = [
    ("player_1_loss", "player_2_loss"),
    ("v_0_log", "v_1_log"),
    ("entropy_p_0", "entropy_p_1"),
    ("actor_loss_0", "actor_loss_1"),
    ("parameters_norm_0", "parameters_norm_1"),
    ("second_order0_sum", "second_order1_sum"),
    ("player_1_update_sum", "player_2_update_sum"),
    ("actor_grad_sum_0", "actor_grad_sum_1"),
    ("lr_decay_ratio",),
    ("pg_expl_player_loss",),
    ("player_1_loss", "player_2_loss", "pg_expl_player_loss"),
    ("v_0_log", "v_1_log", "pg_expl_v_log"),
    ("entropy_p_0", "entropy_p_1"),
    ("player_1_update_sum", "player_2_update_sum", "pg_expl_entropy"),
    ("actor_loss_0", "actor_loss_1", "pg_expl_actor_loss"),
    ("parameters_norm_0", "parameters_norm_1", "pg_expl_parameters_norm"),
    ("player_1_update_sum", "player_2_update_sum", "pg_expl_update_sum"),
    ("actor_grad_sum_0", "actor_grad_sum_1", "pg_expl_actor_grad_sum"),
]

def update(mainPN, lr, final_delta_1_v, final_delta_2_v, use_actions_from_exploiter=False):
    update_theta_1 = mainPN[0].setparams(
        mainPN[0].getparams() + lr * np.squeeze(final_delta_1_v))
    if not use_actions_from_exploiter:
        update_theta_2 = mainPN[1].setparams(
            mainPN[1].getparams() + lr * np.squeeze(final_delta_2_v))

def update_single(policy_network, lr, final_delta_1_v):
    update_theta_3 = policy_network.setparams(
        policy_network.getparams() + lr * np.squeeze(final_delta_1_v))

def copy_weigths(from_policy, to_policy, adding_scaled_weights=False):
    if not adding_scaled_weights:
        copy_weigths = to_policy.setparams(from_policy.getparams())
    else:
        mean_weigths = to_policy.setparams(
            (to_policy.getparams() + adding_scaled_weights * from_policy.getparams())/
            (1+adding_scaled_weights))


def clone_update(mainPN_clone):
    for i in range(2):
        mainPN_clone[i].log_pi_clone = tf.reduce_mean(
            mainPN_clone[i].log_pi_action_bs)
        mainPN_clone[i].clone_trainer = \
            tf.train.GradientDescentOptimizer(learning_rate=0.1)
        mainPN_clone[i].update = mainPN_clone[i].clone_trainer.minimize(
            -mainPN_clone[i].log_pi_clone, var_list=mainPN_clone[i].parameters)


def add_data_in_dqn_data_buffer():
    raise NotImplementedError()


def train_dqn_policy(dqn_data_buffer, dqn_exploiter):
    raise NotImplementedError()


class LOLAPGCG(tune.Trainable):

    def _init_lola(self, env_class, seed, num_episodes, trace_length, batch_size,
                   lola_update, opp_model, grid_size, gamma, hidden, bs_mul,
                   lr, env_config,
                   mem_efficient=True,
                   warmup=False,
                   changed_config=False,
                   ac_lr=1.0,
                   summary_len=20,
                   use_MAE=False,
                   clip_lola_update_norm=False,
                   clip_loss_norm=False,
                   entropy_coeff=0.0,
                   weigth_decay=0.0,
                   lola_correction_multiplier=1.0,
                   clip_lola_correction_norm=False,
                   clip_lola_actor_norm=False,
                   use_critic=False,
                   lr_decay=False,
                   correction_reward_baseline_per_step=False,
                   exploiter_rolling_avg=0.0,
                   exploiter_thresholds=None,
                   use_PG_exploiter=False,
                   start_using_exploiter_at_update_n=0,
                   every_n_updates_copy_weights = 100,
                   use_destabilizer= False,
                   adding_scaled_weights=False,
                   always_train_PG=False,
                   **kwargs):

        print("args not used:", kwargs)
        if opp_model:
            # Opponent modeling not tested nor supported when improving the stability of the algorithm
            raise NotImplementedError()

        corrections = lola_update

        # Instantiate the environment
        self.env = env_class(env_config)
        self.env.seed(seed)

        self.timestep = 0

        self.num_episodes = num_episodes
        self.trace_length = trace_length
        self.batch_size = batch_size
        self.corrections = corrections
        self.opp_model = opp_model
        if opp_model:
            raise NotImplementedError()
        self.grid_size = grid_size
        self.gamma = gamma
        self.hidden = hidden
        self.bs_mul = bs_mul
        self.lr = lr
        self.mem_efficient = mem_efficient
        self.asymmetry = env_class == AsymVectorizedCoinGame
        self.warmup = warmup
        self.changed_config = changed_config
        self.ac_lr = ac_lr
        self.summary_len = summary_len
        self.use_MAE = use_MAE
        self.clip_lola_update_norm = clip_lola_update_norm
        self.clip_loss_norm = clip_loss_norm
        self.entropy_coeff = entropy_coeff
        self.lola_correction_multiplier = lola_correction_multiplier
        self.lr_decay = lr_decay
        self.correction_reward_baseline_per_step = correction_reward_baseline_per_step
        self.use_critic = use_critic
        # Related to the possible exploiter
        self.start_using_exploiter_at_update_n = start_using_exploiter_at_update_n
        self.exploiter_rolling_avg_factor = exploiter_rolling_avg
        self.exploiter_rolling_avg_r_coop = 0.0
        self.exploiter_rolling_avg_r_selfish = 0.0
        self.exploiter_thresholds = exploiter_thresholds
        self.last_batch_opp_coop = 0.0
        self.last_batch_used_exploiter = False
        self.use_PG_exploiter = use_PG_exploiter
        self.use_destabilizer = use_destabilizer
        self.every_n_updates_copy_weights = every_n_updates_copy_weights if self.use_PG_exploiter else False
        self.adding_scaled_weights = adding_scaled_weights
        if self.use_PG_exploiter and self.adding_scaled_weights:
            assert self.adding_scaled_weights > 0.0
        self.always_train_PG=always_train_PG
        self.last_term_to_use = 0.0

        self.obs_batch = deque(maxlen=self.batch_size)
        self.full_episode_logger = FullEpisodeLogger(
            logdir=self._logdir,
            log_interval=100,
            log_ful_epi_one_hot_obs=True)

        # Setting the training parameters
        self.y = gamma
        self.n_agents = self.env.NUM_AGENTS
        if self.use_PG_exploiter:
            self.total_n_agents = self.n_agents + 1
        else:
            self.total_n_agents = self.n_agents
        self.h_size = [hidden] * self.total_n_agents
        self.max_epLength = trace_length + 1  # The max allowed length of our episode.

        graph = tf.Graph()

        with graph.as_default() as g:
            self.sess = tf.Session()

            self.mainPN = []
            self.mainPN_step = []

            self.agent_list = np.arange(self.total_n_agents)
            for agent in range(self.total_n_agents):
                print("mainPN", agent)
                self.mainPN.append(
                    Pnetwork(f'main_{agent}', self.h_size[agent], agent, self.env,
                             trace_length=trace_length, batch_size=batch_size,
                             changed_config=changed_config, ac_lr=ac_lr,
                             use_MAE=use_MAE,  # use_toolbox_env=use_toolbox_env,
                             clip_loss_norm=clip_loss_norm, sess=self.sess,
                             entropy_coeff=entropy_coeff, weigth_decay=weigth_decay,
                             use_critic=use_critic))
                print("mainPN_step", agent)
                self.mainPN_step.append(
                    Pnetwork(f'main_{agent}', self.h_size[agent], agent, self.env,
                             trace_length=trace_length, batch_size=batch_size,
                             reuse=True, step=True, use_MAE=use_MAE,
                             changed_config=changed_config, ac_lr=ac_lr,
                             clip_loss_norm=clip_loss_norm, sess=self.sess,
                             entropy_coeff=entropy_coeff, weigth_decay=weigth_decay,
                             use_critic=use_critic))
            # Clones of the opponents
            if opp_model:
                self.mainPN_clone = []
                for agent in range(self.total_n_agents):
                    self.mainPN_clone.append(
                        Pnetwork(f'clone_{agent}', self.h_size[agent], agent, self.env,
                                 trace_length=trace_length, batch_size=batch_size,
                                 changed_config=changed_config, ac_lr=ac_lr,
                                 use_MAE=use_MAE,
                                 clip_loss_norm=clip_loss_norm, sess=self.sess,
                                 entropy_coeff=entropy_coeff,
                                 use_critic=use_critic))

            if not mem_efficient:
                self.cube, self.cube_ops = make_cube(trace_length)
            else:
                self.cube, self.cube_ops = None, None

            if not opp_model:
                corrections_func(self.mainPN, batch_size, trace_length, corrections, self.cube,
                                 clip_lola_update_norm=clip_lola_update_norm,
                                 lola_correction_multiplier=self.lola_correction_multiplier,
                                 clip_lola_correction_norm=clip_lola_correction_norm,
                                 clip_lola_actor_norm=clip_lola_actor_norm,
                                 )
            else:
                corrections_func([self.mainPN[0], self.mainPN_clone[1]],
                                 batch_size, trace_length, corrections, self.cube,
                                 clip_lola_update_norm=clip_lola_update_norm,
                                 lola_correction_multiplier=self.lola_correction_multiplier,
                                 clip_lola_correction_norm=clip_lola_correction_norm,
                                 clip_lola_actor_norm=clip_lola_actor_norm,
                                 )
                corrections_func([self.mainPN[1], self.mainPN_clone[0]],
                                 batch_size, trace_length, corrections, self.cube,
                                 clip_lola_update_norm=clip_lola_update_norm,
                                 lola_correction_multiplier=self.lola_correction_multiplier,
                                 clip_lola_correction_norm=clip_lola_correction_norm,
                                 clip_lola_actor_norm=clip_lola_actor_norm,
                                 )
                clone_update(self.mainPN_clone)

            if self.use_PG_exploiter:
                simple_actor_training_func(self.mainPN[2], self.mainPN[0],
                                 batch_size, trace_length, self.cube)

            self.init = tf.global_variables_initializer()

            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

            self.trainables = tf.trainable_variables()

            # create lists to contain total rewards and steps per episode
            self.jList = []
            self.rList = []
            self.aList = []

            self.total_steps = 0


            pow_series = np.arange(trace_length)
            discount = np.array([pow(gamma, item) for item in pow_series])
            discount_array = gamma ** trace_length / discount
            self.discount = np.expand_dims(discount, 0)
            self.discount_array = np.reshape(discount_array, [1, -1])

            self.sess.run(self.init)
            if not self.mem_efficient:
                self.sess.run(self.cube_ops)

            _ = self.env.reset()
            self.updated = True
            self.warmup_step_n = 0
            self.n_epi_from_start = 0

            self.reset_compute_actions_state()

    # TODO add something to not load and create everything when only evaluating with RLLib

    def setup(self, config):
        self._init_lola(**config)

    def step(self):
        self.timestep += 1
        self.full_episode_logger.on_episode_start()
        to_report = {"episodes_total": self.timestep}

        episodeBuffer = []
        for ii in range(self.n_agents):
            episodeBuffer.append([])
        # np.random.shuffle(self.agent_list) # Not used and confusing
        if self.n_agents == self.total_n_agents:
            these_agents = range(self.total_n_agents)
        else:
            if not self.use_PG_exploiter:
                these_agents = sorted(self.agent_list[0:self.n_agents])
            else:
                these_agents = range(self.total_n_agents)

        if self.warmup_step_n < self.warmup:
            self.warmup_step_n += 1

        obs = self.env.reset()
        # Be careful this prevents to use
        #  _obs_invariant_to_the_player_trained in coin game
        sP = obs["player_red"]

        s = sP

        trainBatch0 = [[], [], [], [], [], []]
        trainBatch1 = [[], [], [], [], [], []]
        if self.use_PG_exploiter:
            expl_trainBatch1 = [[], [], [], [], [], []]

        d = False
        rAll = np.zeros((8))
        aAll = np.zeros((self.env.NUM_ACTIONS * self.total_n_agents))
        j = 0
        last_info = {}

        lstm_state = []
        for agent in these_agents:
            lstm_state.append(np.zeros((self.batch_size, self.h_size[agent] * 2)))

        while j < self.max_epLength:
            lstm_state_old = lstm_state
            j += 1
            a_all = []
            lstm_state = []
            for agent_role, agent in enumerate(these_agents):
                # Batch of observations => batch of actions
                a, lstm_s = self.sess.run(
                    [
                        self.mainPN_step[agent].predict,
                        self.mainPN_step[agent].lstm_state_output
                    ],
                    feed_dict={
                        self.mainPN_step[agent].state_input: s,
                        self.mainPN_step[agent].lstm_state: lstm_state_old[agent],
                        self.mainPN_step[agent].is_training: True,
                    }
                )
                lstm_state.append(lstm_s)
                a_all.append(a)

            trainBatch0[0].append(s)
            trainBatch1[0].append(s)

            actions = {"player_red": a_all[0],
                       "player_blue": a_all[1]}
            obs, r, d, info = self.env.step(actions)
            self._log_one_step_in_full_episode(s, r, actions, obs, info)
            d = np.array([d["__all__"] for _ in range(self.batch_size)])
            # TODO this prevents us to use
            #  _obs_invariant_to_the_player_trained
            s1P = obs["player_red"]
            if 'player_red' in info.keys():
                last_info.update({f"player_red_{k}": v for k, v in info['player_red'].items()})
            if 'player_blue' in info.keys():
                last_info.update({f"player_blue_{k}": v for k, v in info['player_blue'].items()})
            r = [r['player_red'], r['player_blue']]
            use_actions_from_exploiter = False


            if self.use_PG_exploiter:
                expl_trainBatch1[0].append(s)
                expl_trainBatch1[1].append(a_all[1])
                expl_trainBatch1[2].append(r[1])
                expl_trainBatch1[3].append(s1P)

                trainBatch1[1].append(a_all[1])
            else:
                trainBatch1[1].append(a_all[1])

            s1 = s1P

            # is the same as a_all_exploiter[0]
            trainBatch0[1].append(a_all[0])
            trainBatch0[2].append(r[0])
            trainBatch1[2].append(r[1])
            trainBatch0[3].append(s1)
            trainBatch1[3].append(s1)

            a_all = np.transpose(np.vstack(a_all))

            self.total_steps += 1

            for index in range(self.batch_size):
                r_pb = [r[0][index], r[1][index]]

                rAll[0] += r_pb[0]
                rAll[1] += r_pb[1]
                # Total reward for both agents over the episode
                rAll[6] += r_pb[0] + r_pb[1]
                # Count n steps in env (episode length)
                rAll[7] += 1

                for agent_n in range(a_all.shape[1]):
                    aAll[int(a_all[index, agent_n] + 4*agent_n)] += 1

            s_old = s
            s = s1
            sP = s1P
            if d.any():
                break

        self.jList.append(j)
        self.rList.append(rAll)
        self.aList.append(aAll)


        # need to multiple with
        pow_series = np.arange(self.trace_length)
        discount = np.array([pow(self.gamma, item) for item in pow_series])

        sample_return0, sample_reward0, sample_reward0_bis = \
            self.compute_centered_discounted_r(
                rewards=trainBatch0[2], discount=discount)
        sample_return1, sample_reward1, sample_reward1_bis = \
            self.compute_centered_discounted_r(
                rewards=trainBatch1[2], discount=discount)

        state_input0 = np.concatenate(trainBatch0[0], axis=0)
        state_input1 = np.concatenate(trainBatch1[0], axis=0)
        actions0 = np.concatenate(trainBatch0[1], axis=0)
        actions1 = np.concatenate(trainBatch1[1], axis=0)

        ob_space_shape = list(self.env.OBSERVATION_SPACE.shape)
        last_state = np.reshape(
            np.concatenate(trainBatch1[3], axis=0),
            [self.batch_size, self.trace_length, ob_space_shape[0],
             ob_space_shape[1], ob_space_shape[2]])[:, -1, :, :, :]

        value_0_next, value_1_next = self.sess.run(
            [self.mainPN_step[0].value, self.mainPN_step[1].value],
            feed_dict={
                self.mainPN_step[0].state_input: last_state,
                self.mainPN_step[1].state_input: last_state,
                self.mainPN_step[0].lstm_state: lstm_state[0],
                self.mainPN_step[1].lstm_state: lstm_state[1],
                self.mainPN_step[0].is_training: True,
                self.mainPN_step[1].is_training: True,
            })

        if self.use_PG_exploiter:
            expl_value_next = self.sess.run(
                self.mainPN_step[2].value,
                feed_dict={
                    self.mainPN_step[2].state_input: last_state,
                    self.mainPN_step[2].lstm_state: lstm_state[2],
                    self.mainPN_step[2].is_training: True,
                })

        if self.opp_model:
            ## update local clones
            update_clone = [self.mainPN_clone[0].update, self.mainPN_clone[1].update]
            feed_dict = {
                self.mainPN_clone[0].state_input: state_input1,
                self.mainPN_clone[0].actions: actions1,
                self.mainPN_clone[0].sample_return: sample_return1,
                self.mainPN_clone[0].sample_reward: sample_reward1,
                self.mainPN_clone[1].state_input: state_input0,
                self.mainPN_clone[1].actions: actions0,
                self.mainPN_clone[1].sample_return: sample_return0,
                self.mainPN_clone[1].sample_reward: sample_reward0,
                self.mainPN_clone[0].gamma_array: np.reshape(discount, [1, -1]),
                self.mainPN_clone[1].gamma_array: np.reshape(discount, [1, -1]),
                self.mainPN_clone[0].is_training: True,
                self.mainPN_clone[1].is_training: True,
            }
            num_loops = 50 if self.timestep == 0 else 1
            for _ in range(num_loops):
                self.sess.run(update_clone, feed_dict=feed_dict)

        if self.lr_decay:
            lr_decay = (self.num_episodes - self.timestep) / self.num_episodes
        else:
            lr_decay = 1.0

        # Update policy networks
        feed_dict = {
            self.mainPN[0].state_input: state_input0,
            self.mainPN[0].sample_return: sample_return0,
            self.mainPN[0].actions: actions0,
            self.mainPN[1].state_input: state_input1,
            self.mainPN[1].sample_return: sample_return1,
            self.mainPN[1].actions: actions1,
            self.mainPN[0].sample_reward: sample_reward0,
            self.mainPN[1].sample_reward: sample_reward1,
            self.mainPN[0].sample_reward_bis: sample_reward0_bis,
            self.mainPN[1].sample_reward_bis: sample_reward1_bis,
            self.mainPN[0].gamma_array: np.reshape(discount, [1, -1]),
            self.mainPN[1].gamma_array: np.reshape(discount, [1, -1]),
            self.mainPN[0].next_value: value_0_next,
            self.mainPN[1].next_value: value_1_next,
            self.mainPN[0].gamma_array_inverse:
                np.reshape(self.discount_array, [1, -1]),
            self.mainPN[1].gamma_array_inverse:
                np.reshape(self.discount_array, [1, -1]),
            self.mainPN[0].loss_multiplier: [lr_decay],
            self.mainPN[1].loss_multiplier: [lr_decay],
            self.mainPN[0].is_training: True,
            self.mainPN[1].is_training: True,
        }
        if self.opp_model:
            feed_dict.update({
                self.mainPN_clone[0].state_input: state_input1,
                self.mainPN_clone[0].actions: actions1,
                self.mainPN_clone[0].sample_return: sample_return1,
                self.mainPN_clone[0].sample_reward: sample_reward1,
                self.mainPN_clone[1].state_input: state_input0,
                self.mainPN_clone[1].actions: actions0,
                self.mainPN_clone[1].sample_return: sample_return0,
                self.mainPN_clone[1].sample_reward: sample_reward0,
                self.mainPN_clone[0].gamma_array: np.reshape(discount, [1, -1]),
                self.mainPN_clone[1].gamma_array: np.reshape(discount, [1, -1]),
            })

        lola_training_list = [
            self.mainPN[0].value,
            self.mainPN[0].updateModel,
            self.mainPN[0].delta,
            self.mainPN[0].value,
            self.mainPN[0].target,
            self.mainPN[0].loss,
            self.mainPN[0].entropy,
            self.mainPN[0].v_0_log,
            self.mainPN[0].actor_target_error,
            self.mainPN[0].actor_loss,
            self.mainPN[0].weigths_norm,
            self.mainPN[0].v_0_grad_01,
            self.mainPN[0].grad,
            self.mainPN[0].second_order,
            self.mainPN[0].grad_sum,
            self.mainPN[0].v_0_grad_01,
            self.mainPN[0].multiply0,
        ]

        lola_training_list.extend([
            self.mainPN[1].value,
            self.mainPN[1].updateModel,
            self.mainPN[1].delta,
            self.mainPN[1].value,
            self.mainPN[1].target,
            self.mainPN[1].loss,
            self.mainPN[1].entropy,
            self.mainPN[1].v_1_log,
            self.mainPN[1].actor_target_error,
            self.mainPN[1].actor_loss,
            self.mainPN[1].weigths_norm,
            self.mainPN[1].v_1_grad_10,
            self.mainPN[1].grad,
            self.mainPN[1].second_order,
            self.mainPN[1].grad_sum,
        ])

        (  # Player_red
            values, updateModel_1, update1, player_1_value, player_1_target,
            player_1_loss, entropy_p_0, v_0_log, actor_target_error_0, actor_loss_0,
            parameters_norm_0, second_order0, v_0_grad_theta_0, second_order0_sum,
            actor_grad_sum_0, v_0_grad_01, multiply0,
            # Player_blue
            values_1, updateModel_2, update2, player_2_value, player_2_target, player_2_loss,
            entropy_p_1, v_1_log, actor_target_error_1, actor_loss_1, parameters_norm_1, second_order1,
            v_1_grad_theta_1, second_order1_sum, actor_grad_sum_1
        ) = self.sess.run(lola_training_list, feed_dict=feed_dict)

        if self.warmup:
            update1 = update1 * self.warmup_step_n / self.warmup
            update2 = update2 * self.warmup_step_n / self.warmup
        if self.lr_decay:
            update1 = update1 * lr_decay
            update2 = update2 * lr_decay

        update1_sum = sum(update1) / self.bs_mul
        update2_sum = sum(update2) / self.bs_mul

        update(self.mainPN, self.lr, update1 / self.bs_mul, update2 / self.bs_mul, use_actions_from_exploiter)

        if self.use_PG_exploiter:
            # Update policy networks
            feed_dict = {
                self.mainPN[0].state_input: state_input0,
                self.mainPN[0].sample_return: sample_return0,
                self.mainPN[0].actions: actions0,
                self.mainPN[2].state_input: state_input1,
                self.mainPN[2].sample_return: sample_return1,
                self.mainPN[2].actions: actions1,
                self.mainPN[0].sample_reward: sample_reward0,
                self.mainPN[2].sample_reward: sample_reward1,
                self.mainPN[0].sample_reward_bis: sample_reward0_bis,
                self.mainPN[2].sample_reward_bis: sample_reward1_bis,
                self.mainPN[0].gamma_array: np.reshape(discount, [1, -1]),
                self.mainPN[2].gamma_array: np.reshape(discount, [1, -1]),
                self.mainPN[0].next_value: value_0_next,
                self.mainPN[2].next_value: expl_value_next,
                self.mainPN[0].gamma_array_inverse:
                    np.reshape(self.discount_array, [1, -1]),
                self.mainPN[2].gamma_array_inverse:
                    np.reshape(self.discount_array, [1, -1]),
                self.mainPN[0].loss_multiplier: [lr_decay],
                self.mainPN[2].loss_multiplier: [lr_decay],
                self.mainPN[0].is_training: True,
                self.mainPN[2].is_training: True,
            }

            lola_training_list = [
                self.mainPN[2].value,
                self.mainPN[2].updateModel,
                self.mainPN[2].delta,
                self.mainPN[2].value,
                self.mainPN[2].target,
                self.mainPN[2].loss,
                self.mainPN[2].entropy,
                self.mainPN[2].v_0_log,
                self.mainPN[2].actor_target_error,
                self.mainPN[2].actor_loss,
                self.mainPN[2].weigths_norm,
                self.mainPN[2].grad,
                self.mainPN[2].grad_sum,
            ]
            (pg_expl_values, pg_expl_updateModel, pg_expl_update, pg_expl_player_value, pg_expl_player_target,
             pg_expl_player_loss, pg_expl_entropy, pg_expl_v_log, pg_expl_actor_target_error, pg_expl_actor_loss,
             pg_expl_parameters_norm, pg_expl_v_grad_theta,
             pg_expl_actor_grad_sum) = self.sess.run(lola_training_list, feed_dict=feed_dict)

            if self.warmup:
                pg_expl_update = pg_expl_update * self.warmup_step_n / self.warmup
            if self.lr_decay:
                pg_expl_update = pg_expl_update * lr_decay

            # pg_expl_update_to_log = pg_expl_update
            pg_expl_update_sum = sum(pg_expl_update) / self.bs_mul

            update_single(self.mainPN[2], self.lr, pg_expl_update / self.bs_mul)

            if self.timestep >= self.start_using_exploiter_at_update_n:
                if self.timestep % self.every_n_updates_copy_weights == 0:
                    copy_weigths(from_policy= self.mainPN[2], to_policy=self.mainPN[1],
                                 adding_scaled_weights=self.adding_scaled_weights)

        print('update params')

        rlog = np.sum(self.rList[-self.summary_len:], 0)

        to_plot = {}
        for ii in range(len(rlog)):
            if ii == 0:
                to_plot['total_reward_player_red'] = rlog[ii]
            elif ii == 1:
                to_plot['total_reward_player_blue'] = rlog[ii]
            elif ii == 6:
                to_plot['total_reward'] = rlog[ii]
            elif ii == 7:
                to_plot['n_steps_per_summary'] = rlog[ii]

        for agent_n in range(int(len(self.aList[-1]) / 4)):
            actions_freq = {f"player_{agent_n}_act_{i}": self.aList[-1][int(i + agent_n*4)] / to_plot['n_steps_per_summary']
                            for i in range(0, 4, 1)}
            to_report.update(actions_freq)

        last_info.pop("available_actions", None)

        training_info = {
            "player_1_loss": player_1_loss,
            "player_2_loss": player_2_loss,
            "v_0_log": v_0_log,
            "v_1_log": v_1_log,
            "entropy_p_0": entropy_p_0,
            "entropy_p_1": entropy_p_1,
            "actor_loss_0": actor_loss_0,
            "actor_loss_1": actor_loss_1,
            "parameters_norm_0": parameters_norm_0,
            "parameters_norm_1": parameters_norm_1,
            "second_order0_sum": second_order0_sum,
            "second_order1_sum": second_order1_sum,
            "player_1_update_sum": update1_sum, # / self.summary_len,
            "player_2_update_sum": update2_sum, # / self.summary_len,
            "actor_grad_sum_0": actor_grad_sum_0,
            "actor_grad_sum_1": actor_grad_sum_1,
            "lr_decay_ratio": (self.num_episodes - self.timestep) / self.num_episodes,
        }
        # Logging distribution (can be a speed bottleneck)
        # training_info.update({
        #     "sample_return0": sample_return0,
        #     "sample_return1": sample_return1,
        #     "sample_reward0": sample_reward0,
        #     "sample_reward1": sample_reward1,
        #     "player_1_values": values,
        #     "player_2_values": values_1,
        #     "player_1_target": player_1_target,
        #     "player_2_target": player_2_target,
        # })

        # self.update1_list.clear()
        # self.update2_list.clear()

        to_report["finished"] = False if self.timestep < self.num_episodes else True
        to_report.update(to_plot)
        to_report.update(last_info)
        to_report.update(training_info)
        if self.use_PG_exploiter:
            expl_training_info = {
                "pg_expl_player_loss":  pg_expl_player_loss,
                "pg_expl_v_log": pg_expl_v_log,
                "pg_expl_entropy": pg_expl_entropy,
                "pg_expl_actor_loss": pg_expl_actor_loss,
                "pg_expl_parameters_norm": pg_expl_parameters_norm,
                "pg_expl_update_sum": pg_expl_update_sum,
                "pg_expl_actor_grad_sum": pg_expl_actor_grad_sum,
            }
            to_report.update(expl_training_info)

        self.full_episode_logger.on_episode_end()

        to_report["hp_search_objective"] = \
            to_report.get("player_blue_pick_speed", 0.0) + \
            to_report.get("player_red_pick_speed", 0.0) + \
            to_report.get("player_blue_pick_own_color", 0.0) + \
            to_report.get("player_red_pick_own_color", 0.0)

        return to_report

    def compute_centered_discounted_r(self, rewards, discount):
        sample_return = np.reshape(
            get_monte_carlo(rewards, self.y, self.trace_length, self.batch_size),
            [self.batch_size, -1])

        if self.correction_reward_baseline_per_step:
            sample_reward = discount * np.reshape(
                rewards - np.mean(np.array(rewards), axis=0), [-1, self.trace_length])
        else:
            sample_reward = discount * np.reshape(
                rewards - np.mean(rewards), [-1, self.trace_length])
        sample_reward_bis = discount * np.reshape(
            rewards, [-1, self.trace_length])
        return sample_return, sample_reward, sample_reward_bis

    def _log_one_step_in_full_episode(self, s, r, actions, obs, info):
        self.full_episode_logger.on_episode_step(
            step_data={
                "player_red":  {
                    "obs_before_act": s[0,...],
                    "obs_after_act": obs["player_red"][0,...],
                    "action": actions["player_red"][0],
                    "reward": r["player_red"][0],
                    "info": info["player_red"]
                    if 'player_red' in info.keys()
                    else None,
                    "epi": self.timestep
                },
                "player_blue": {
                    "obs_before_act": s[0,...],
                    "obs_after_act": obs["player_blue"][0,...],
                    "action": actions["player_blue"][0],
                    "reward": r["player_blue"][0],
                    "info": info["player_blue"]
                    if 'player_blue' in info.keys()
                    else None,
                    "epi": self.timestep
                },
           }
        )

    def save_checkpoint(self, checkpoint_dir):
        path = os.path.join(checkpoint_dir, "checkpoint.json")
        tf_checkpoint_path = os.path.join(checkpoint_dir, "checkpoint")
        tf_checkpoint_dir, tf_checkpoint_filename = os.path.split(tf_checkpoint_path)
        checkpoint = {
            "timestep": self.timestep,
            "tf_checkpoint_dir": tf_checkpoint_dir,
            "tf_checkpoint_filename": tf_checkpoint_filename,
        }
        with open(path, "w") as f:
            json.dump(checkpoint, f, sort_keys=True, indent=4)

        # TF v1
        save_path = self.saver.save(self.sess, f"{tf_checkpoint_path}.ckpt")

        return path

    def load_checkpoint(self, checkpoint_path):

        checkpoint_path = os.path.expanduser(checkpoint_path)
        print('Loading Model...', checkpoint_path)
        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)
        print("checkpoint", checkpoint)

        # Support VM and local (manual) loading
        tf_checkpoint_dir, _ = os.path.split(checkpoint_path)
        print("tf_checkpoint_dir", tf_checkpoint_dir)
        ckpt = tf.train.get_checkpoint_state(tf_checkpoint_dir,
                                             latest_filename=f'{checkpoint["tf_checkpoint_filename"]}')
        tail, head = os.path.split(ckpt.model_checkpoint_path)
        ckpt.model_checkpoint_path = os.path.join(tf_checkpoint_dir, head)
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)

    def cleanup(self):
        self.sess.close()
        super().cleanup()

    def compute_actions(self, policy_id: str, obs_batch: list):
        # because of the LSTM
        assert len(obs_batch) == 1

        for single_obs in obs_batch:
            agent_to_use = self._get_agent_to_use(policy_id)
            obs = self._preprocess_obs(single_obs, agent_to_use)
            a, lstm_s = self.sess.run(
                [
                    self.mainPN_step[agent_to_use].predict,
                    self.mainPN_step[agent_to_use].lstm_state_output
                ],
                feed_dict={
                    self.mainPN_step[agent_to_use].state_input: obs,
                    self.mainPN_step[agent_to_use].lstm_state:
                        self.lstm_state[agent_to_use],
                    self.mainPN_step[agent_to_use].is_training: False,
                }
            )
            self.lstm_state[agent_to_use] = lstm_s
        action = self._post_process_action(a)

        state_out = []
        extra_fetches = {}
        return action, state_out, extra_fetches

    def _get_agent_to_use(self, policy_id):
        if policy_id == "player_red":
            agent_n = 0
        elif policy_id == "player_blue":
            agent_n = 1
        else:
            raise ValueError(f"policy_id {policy_id}")
        return agent_n

    def _preprocess_obs(self, single_obs, agent_to_use):
        single_obs = single_obs[None, ...]  # add batch dim

        # Compensate for the batch norm not in evaluation mode
        while len(self.obs_batch) < self.batch_size:
            self.obs_batch.append(single_obs)
        self.obs_batch.append(single_obs)
        single_obs = np.concatenate(list(self.obs_batch), axis=0)
        return single_obs

    def _post_process_action(self, action):
        # Compensate for the batch norm not in evaluation mode
        if isinstance(action, Iterable):
            action = action[-1]

        return action[None, ...]  # add batch dim

    def reset_compute_actions_state(self):
        self.lstm_state = []
        for agent in range(self.n_agents):
            self.lstm_state.append(
                np.zeros((self.batch_size, self.h_size[agent] * 2)))
