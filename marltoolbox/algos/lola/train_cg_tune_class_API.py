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
import torch
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.agents.dqn import DEFAULT_CONFIG, DQNTorchPolicy
from ray.rllib.evaluation.sample_batch_builder import MultiAgentSampleBatchBuilder
from ray.rllib.execution.replay_buffer import LocalReplayBuffer

import marltoolbox.algos.lola_dice.envs as lola_dice_envs
from marltoolbox.algos.lola.corrections import corrections_func
from marltoolbox.algos.lola.networks import Pnetwork
from marltoolbox.algos.lola.utils import get_monte_carlo, make_cube
from marltoolbox.envs.coin_game import CoinGame, AsymCoinGame


def update(mainPN, lr, final_delta_1_v, final_delta_2_v):
    update_theta_1 = mainPN[0].setparams(
        mainPN[0].getparams() + lr * np.squeeze(final_delta_1_v))
    update_theta_2 = mainPN[1].setparams(
        mainPN[1].getparams() + lr * np.squeeze(final_delta_2_v))


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

    def _init_lola(self, env, seed, num_episodes, trace_length, batch_size,
                   lola_update, opp_model, grid_size, gamma, hidden, bs_mul, lr,
                   mem_efficient=True, asymmetry=False, warmup=False,
                   changed_config=False, ac_lr=1.0, summary_len=20, use_MAE=False,
                   use_toolbox_env=False,
                   clip_lola_update_norm=False, clip_loss_norm=False,
                   entropy_coeff=0.0, weigth_decay=0.0, lola_correction_multiplier=1.0,
                   clip_lola_correction_norm=False,
                   clip_lola_actor_norm=False, use_critic=False,
                   lr_decay=False, correction_reward_baseline_per_step=False,
                   playing_against_exploiter=False, debug=False, train_exploiter_n_times_per_epi=1,
                   exploiter_base_lr=0.1, exploiter_decay_lr_in_n_epi=1500,
                   exploiter_stop_training_after_n_epi=1500, exploiter_rolling_avg=0.0,
                   **kwargs):

        print("args not used:", kwargs)
        if opp_model:
            # Opponent modeling not tested nor supported when improving the stability of the algorithm
            raise NotImplementedError()

        corrections = lola_update

        # Instantiate the environment
        if env == CoinGame:
            if use_toolbox_env:
                self.env = CoinGame(config={
                    "batch_size": batch_size,
                    "max_steps": trace_length,
                    "grid_size": grid_size,
                    "get_additional_info": True,
                })
            else:
                self.env = lola_dice_envs.CG(trace_length, batch_size, grid_size)
            self.env.seed(seed)
        elif env == AsymCoinGame:
            if use_toolbox_env:
                self.env = AsymCoinGame(config={
                    "batch_size": batch_size,
                    "max_steps": trace_length,
                    "grid_size": grid_size,
                    "get_additional_info": True,
                })
            else:
                self.env = lola_dice_envs.AsymCG(trace_length, batch_size, grid_size)
            self.env.seed(seed)
        else:
            raise ValueError(f"exp_name: {env}")

        self.timestep = 0

        self.num_episodes = num_episodes
        self.trace_length = trace_length
        self.batch_size = batch_size
        self.corrections = corrections
        self.opp_model = opp_model
        self.grid_size = grid_size
        self.gamma = gamma
        self.hidden = hidden
        self.bs_mul = bs_mul
        self.lr = lr
        self.mem_efficient = mem_efficient
        self.asymmetry = env == AsymCoinGame
        self.warmup = warmup
        self.changed_config = changed_config
        self.ac_lr = ac_lr
        self.summary_len = summary_len
        self.use_MAE = use_MAE
        self.use_toolbox_env = use_toolbox_env
        self.clip_lola_update_norm = clip_lola_update_norm
        self.clip_loss_norm = clip_loss_norm
        self.entropy_coeff = entropy_coeff
        self.lola_correction_multiplier = lola_correction_multiplier
        self.lr_decay = lr_decay
        self.correction_reward_baseline_per_step = correction_reward_baseline_per_step
        self.use_critic = use_critic
        self.playing_against_exploiter = playing_against_exploiter
        self.train_exploiter_n_times_per_epi = train_exploiter_n_times_per_epi
        self.exploiter_stop_training_after_n_epi = exploiter_stop_training_after_n_epi

        self.obs_batch = deque(maxlen=self.batch_size)

        # Setting the training parameters
        self.y = gamma
        self.n_agents = self.env.NUM_AGENTS
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
                             use_MAE=use_MAE, use_toolbox_env=use_toolbox_env,
                             clip_loss_norm=clip_loss_norm, sess=self.sess,
                             entropy_coeff=entropy_coeff, weigth_decay=weigth_decay,
                             use_critic=use_critic))
                print("mainPN_step", agent)
                self.mainPN_step.append(
                    Pnetwork(f'main_{agent}', self.h_size[agent], agent, self.env,
                             trace_length=trace_length, batch_size=batch_size,
                             reuse=True, step=True, use_MAE=use_MAE,
                             changed_config=changed_config, ac_lr=ac_lr,
                             use_toolbox_env=use_toolbox_env,
                             clip_loss_norm=clip_loss_norm, sess=self.sess,
                             entropy_coeff=entropy_coeff, weigth_decay=weigth_decay,
                             use_critic=use_critic))

                if self.playing_against_exploiter and agent == 1:
                    self.create_dqn_exploiter(exploiter_base_lr, exploiter_decay_lr_in_n_epi, exploiter_rolling_avg)

            # Clones of the opponents
            if opp_model:
                self.mainPN_clone = []
                for agent in range(self.total_n_agents):
                    self.mainPN_clone.append(
                        Pnetwork(f'clone_{agent}', self.h_size[agent], agent, self.env,
                                 trace_length=trace_length, batch_size=batch_size,
                                 changed_config=changed_config, ac_lr=ac_lr,
                                 use_MAE=use_MAE, use_toolbox_env=use_toolbox_env,
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
                                 # playing_against_exploiter=self.playing_against_exploiter
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
                                 clip_lola_actor_norm=clip_lola_actor_norm
                                 )
                clone_update(self.mainPN_clone)

            self.init = tf.global_variables_initializer()

            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

            self.trainables = tf.trainable_variables()

            # create lists to contain total rewards and steps per episode
            self.jList = []
            self.rList = []
            self.aList = []
            self.update1_list = []
            self.update2_list = []

            self.total_steps = 0

            self.episodes_run = np.zeros(self.total_n_agents)
            self.episodes_run_counter = np.zeros(self.total_n_agents)
            self.episodes_reward = np.zeros((self.total_n_agents, batch_size))
            self.episodes_actions = np.zeros((self.total_n_agents, self.env.NUM_ACTIONS))

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
        print("_init_lola", config)
        self._init_lola(**config)

    def create_dqn_exploiter(self, exploiter_base_lr, exploiter_decay_lr_in_n_epi, exploiter_rolling_avg):
        # with tf.variable_scope(f"dqn_exploiter"):
        # Create the dqn policy for the exploiter
        dqn_config = copy.deepcopy(DEFAULT_CONFIG)
        dqn_config.update({
            "prioritized_replay": False,
            "double_q": True,
            "buffer_size": 50000,
            "dueling": False,
            "learning_starts": min(int((self.batch_size - 1) * (self.trace_length - 1)),64),
            "model": {
                "dim": self.grid_size,
                "conv_filters": [[16, [3, 3], 1], [32, [3, 3], 1]],  # [Channel, [Kernel, Kernel], Stride]]
                # "fcnet_hiddens": [self.env.NUM_ACTIONS],
                "max_seq_len": self.trace_length,
                # Number of hidden layers for fully connected net
                "fcnet_hiddens": [64],
                # Nonlinearity for fully connected net (tanh, relu)
                "fcnet_activation": "relu",
            },
            # Update the replay buffer with this many samples at once. Note that
            # this setting applies per-worker if num_workers > 1.
            "rollout_fragment_length": 1,
            # Size of a batch sampled from replay buffer for training. Note that
            # if async_updates is set, then each worker returns gradients for a
            # batch of this size.
            "train_batch_size": min(int((self.batch_size) * (self.trace_length)), 64),
            "explore": False,
            "grad_clip": 1,
            "gamma": 0.5,
            "lr": exploiter_base_lr,
            # Learning rate schedule
            "lr_schedule": [
                (0, exploiter_base_lr / 1000),
                (100, exploiter_base_lr),
                (exploiter_decay_lr_in_n_epi, exploiter_base_lr / 1e9)],
            "sgd_momentum": 0.9,
        })
        print("dqn_config", dqn_config)

        self.local_replay_buffer = LocalReplayBuffer(
            num_shards=1,
            learning_starts=dqn_config["learning_starts"],
            buffer_size=dqn_config["buffer_size"],
            replay_batch_size=dqn_config["train_batch_size"],
            replay_mode=dqn_config["multiagent"]["replay_mode"],
            replay_sequence_length=dqn_config["replay_sequence_length"])

        # self.dqn_exploiter = DQNTFPolicy(obs_space=self.env.OBSERVATION_SPACE,
        #                                  action_space=self.env.ACTION_SPACE,
        #                                  config=dqn_config)

        def sgd_optimizer_dqn(policy, config) -> "torch.optim.Optimizer":
            return torch.optim.SGD(policy.q_func_vars, lr=policy.cur_lr, momentum=config["sgd_momentum"])
        MyDQNTorchPolicy = DQNTorchPolicy.with_updates(optimizer_fn=sgd_optimizer_dqn)
        self.dqn_exploiter = MyDQNTorchPolicy(obs_space=self.env.OBSERVATION_SPACE,
                                            action_space=self.env.ACTION_SPACE,
                                            config=dqn_config)

        self.multi_agent_batch_builders = [MultiAgentSampleBatchBuilder(
            policy_map={"player_blue": self.dqn_exploiter},
            clip_rewards=False,
            callbacks=DefaultCallbacks()
        )
            # for _ in range(self.batch_size)
        ]

        self.exploiter_rolling_avg_factor = exploiter_rolling_avg
        self.exploiter_rolling_avg_r_coop = 0.0
        self.exploiter_rolling_avg_r_selfish = 0.0

    def add_data_in_rllib_batch_builder(self, s, s1P, trainBatch1, d):
        if self.timestep <= self.exploiter_stop_training_after_n_epi:
            # for i in range(self.batch_size):
            i = 0
            step_player_values = {
                "eps_id": self.timestep,
                "obs": s[i],
                "new_obs": s1P[i],
                "actions": trainBatch1[1][-1][i],
                "prev_actions": trainBatch1[1][-2][i] if len(trainBatch1[1]) > 1 else 0,
                "rewards": trainBatch1[2][-1][i],
                "prev_rewards": trainBatch1[2][-2][i] if len(trainBatch1[2]) > 1 else 0,
                "dones": d[0],  # done is the same for for every episodes in the batch
            }
            self.multi_agent_batch_builders[i].add_values(agent_id="player_blue", policy_id="player_blue",
                                                          **step_player_values)
            self.multi_agent_batch_builders[i].count += 1

    def train_dqn_policy(self):
        stats = {"learner_stats": {}}
        if self.timestep <= self.exploiter_stop_training_after_n_epi:
            # Add episodes in replay buffer
            # for i in range(self.batch_size):
            i = 0
            multiagent_batch = self.multi_agent_batch_builders[i].build_and_reset()
            self.local_replay_buffer.add_batch(multiagent_batch)

            # update lr in scheduler & in optimizer
            self.dqn_exploiter.on_global_var_update({
                "timestep": self.timestep
            })
            self.dqn_exploiter.optimizer()
            if hasattr(self.dqn_exploiter, "cur_lr"):
                for opt in self.dqn_exploiter._optimizers:
                    for p in opt.param_groups:
                        p["lr"] = self.dqn_exploiter.cur_lr
            # Generate training batch and train
            for _ in range(self.train_exploiter_n_times_per_epi):
                replay_batch = self.local_replay_buffer.replay()
                if replay_batch is not None: # is None when there is not enough step in the data buffer
                    stats = self.dqn_exploiter.learn_on_batch(replay_batch.policy_batches["player_blue"])

        stats["learner_stats"]["exploiter_lr_cur"] = self.dqn_exploiter.cur_lr
        for j, opt in enumerate(self.dqn_exploiter._optimizers):
            stats["learner_stats"]["exploiter_lr_from_params"] = [p["lr"] for p in opt.param_groups][0]
        return stats

    def step(self):
        self.timestep += 1

        episodeBuffer = []
        for ii in range(self.n_agents):
            episodeBuffer.append([])
        np.random.shuffle(self.agent_list)
        if self.n_agents == self.total_n_agents:
            these_agents = range(self.n_agents)
        else:
            these_agents = sorted(self.agent_list[0:self.n_agents])

        if self.warmup_step_n < self.warmup:
            self.warmup_step_n += 1

        if not self.use_toolbox_env:
            # using coin game from lola.envs
            # Reset environment and get first new observation
            # sP = env.reset()
            # using coin game from lola_dice.envs
            obs, _ = self.env.reset()
            sP = obs[0]
        else:
            obs = self.env.reset()
            sP = obs["player_red"]

        s = sP

        trainBatch0 = [[], [], [], [], [], []]
        trainBatch1 = [[], [], [], [], [], []]
        d = False
        rAll = np.zeros((8))
        aAll = np.zeros((self.env.NUM_ACTIONS * 2))
        j = 0
        last_info = {}

        lstm_state = []
        for agent in these_agents:
            self.episodes_run[agent] += 1
            self.episodes_run_counter[agent] += 1
            lstm_state.append(np.zeros((self.batch_size, self.h_size[agent] * 2)))

        while j < self.max_epLength:
            lstm_state_old = lstm_state
            j += 1
            a_all = []
            lstm_state = []
            for agent_role, agent in enumerate(these_agents):

                use_exploiter = False
                if agent_role == 1 and self.playing_against_exploiter:
                    if self.timestep > self.playing_against_exploiter:
                        use_exploiter = True

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

                if use_exploiter:
                    action, a2, a3 = self.dqn_exploiter.compute_actions(obs_batch=s)
                    a = action
                    # print("dqn action", action, a2, a3)

                a_all.append(a)

            trainBatch0[0].append(s)
            trainBatch1[0].append(s)
            trainBatch0[1].append(a_all[0])
            trainBatch1[1].append(a_all[1])

            if not self.use_toolbox_env:
                # using coin game from lola.envs
                # a_all = np.transpose(np.vstack(a_all))
                # s1P,r,d = env.step(actions=a_all)
                # using coin game from lola_dice.envs
                obs, r, d, info = self.env.step(a_all)
                d = np.array([d for _ in range(self.batch_size)])
                s1P = obs[0]
                last_info.update(info)
            else:
                actions = {"player_red": a_all[0],
                           "player_blue": a_all[1]}
                obs, r, d, info = self.env.step(actions)
                d = np.array([d["__all__"] for _ in range(self.batch_size)])
                s1P = obs["player_red"]
                if 'player_red' in info.keys():
                    last_info.update({f"player_red_{k}": v for k, v in info['player_red'].items()})
                if 'player_blue' in info.keys():
                    last_info.update({f"player_blue_{k}": v for k, v in info['player_blue'].items()})
                r = [r['player_red'], r['player_blue']]

            a_all = np.transpose(np.vstack(a_all))
            s1 = s1P

            trainBatch0[2].append(r[0])
            trainBatch1[2].append(r[1])
            trainBatch0[3].append(s1)
            trainBatch1[3].append(s1)
            # trainBatch0[4].append(d)
            # trainBatch1[4].append(d)
            # trainBatch0[5].append(lstm_state[0])
            # trainBatch1[5].append(lstm_state[1])

            if self.playing_against_exploiter:
                self.add_data_in_rllib_batch_builder(s, s1P, trainBatch1, d)

            self.total_steps += 1
            for agent_role, agent in enumerate(these_agents):
                self.episodes_reward[agent] += r[agent_role]

            for index in range(self.batch_size):
                r_pb = [r[0][index], r[1][index]]

                rAll[0] += r_pb[0]
                rAll[1] += r_pb[1]
                # Total reward for both agents over the episode
                rAll[6] += r_pb[0] + r_pb[1]
                # Count n steps in env (episode length)
                rAll[7] += 1

                aAll[a_all[index, 0]] += 1
                aAll[a_all[index, 1] + 4] += 1

            s_old = s
            s = s1
            sP = s1P
            if d.any():
                break

        self.jList.append(j)
        self.rList.append(rAll)
        self.aList.append(aAll)

        # training after one batch is obtained
        sample_return0 = np.reshape(
            get_monte_carlo(trainBatch0[2], self.y, self.trace_length, self.batch_size),
            [self.batch_size, -1])
        sample_return1 = np.reshape(
            get_monte_carlo(trainBatch1[2], self.y, self.trace_length, self.batch_size),
            [self.batch_size, -1])
        # need to multiple with
        pow_series = np.arange(self.trace_length)
        discount = np.array([pow(self.gamma, item) for item in pow_series])

        if self.correction_reward_baseline_per_step:
            sample_reward0 = discount * np.reshape(
                trainBatch0[2] - np.mean(np.array(trainBatch0[2]), axis=0), [-1, self.trace_length])
            sample_reward1 = discount * np.reshape(
                trainBatch1[2] - np.mean(np.array(trainBatch1[2]), axis=0), [-1, self.trace_length])
        else:
            sample_reward0 = discount * np.reshape(
                trainBatch0[2] - np.mean(trainBatch0[2]), [-1, self.trace_length])
            sample_reward1 = discount * np.reshape(
                trainBatch1[2] - np.mean(trainBatch1[2]), [-1, self.trace_length])
        sample_reward0_bis = discount * np.reshape(
            trainBatch0[2], [-1, self.trace_length])
        sample_reward1_bis = discount * np.reshape(
            trainBatch1[2], [-1, self.trace_length])

        state_input0 = np.concatenate(trainBatch0[0], axis=0)
        state_input1 = np.concatenate(trainBatch1[0], axis=0)
        actions0 = np.concatenate(trainBatch0[1], axis=0)
        actions1 = np.concatenate(trainBatch1[1], axis=0)

        if self.use_toolbox_env:
            ob_space_shape = list(self.env.OBSERVATION_SPACE.shape)
            last_state = np.reshape(
                np.concatenate(trainBatch1[3], axis=0),
                [self.batch_size, self.trace_length, ob_space_shape[0],
                 ob_space_shape[1], ob_space_shape[2]])[:, -1, :, :, :]
        else:
            last_state = np.reshape(
                np.concatenate(trainBatch1[3], axis=0),
                [self.batch_size, self.trace_length, self.env.ob_space_shape[0],
                 self.env.ob_space_shape[1], self.env.ob_space_shape[2]])[:, -1, :, :, :]

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

            theta_1_vals = self.mainPN[0].getparams()
            theta_2_vals = self.mainPN[1].getparams()
            theta_1_vals_clone = self.mainPN_clone[0].getparams()
            theta_2_vals_clone = self.mainPN_clone[1].getparams()

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

        (values, values_1, updateModel_1, updateModel_2,
         update1, update2,
         player_1_value, player_2_value, player_1_target, player_2_target,
         player_1_loss, player_2_loss, entropy_p_0, entropy_p_1, v_0_log, v_1_log,
         actor_target_error_0, actor_target_error_1, actor_loss_0, actor_loss_1,
         parameters_norm_0, parameters_norm_1,  # value_params_norm_0, value_params_norm_1,
         second_order0, second_order1, v_0_grad_theta_0, v_1_grad_theta_1,
         second_order0_sum, second_order1_sum,
         actor_grad_sum_0, actor_grad_sum_1) = self.sess.run(
            [
                self.mainPN[0].value,
                self.mainPN[1].value,
                self.mainPN[0].updateModel,
                self.mainPN[1].updateModel,
                self.mainPN[0].delta,
                self.mainPN[1].delta,

                self.mainPN[0].value,
                self.mainPN[1].value,
                self.mainPN[0].target,
                self.mainPN[1].target,
                self.mainPN[0].loss,
                self.mainPN[1].loss,
                self.mainPN[0].entropy,
                self.mainPN[1].entropy,

                self.mainPN[0].v_0_log,
                self.mainPN[1].v_1_log,

                self.mainPN[0].actor_target_error,
                self.mainPN[1].actor_target_error,
                self.mainPN[0].actor_loss,
                self.mainPN[1].actor_loss,

                self.mainPN[0].weigths_norm,
                self.mainPN[1].weigths_norm,

                self.mainPN[0].v_0_grad_01,
                self.mainPN[1].v_1_grad_10,

                self.mainPN[0].grad,
                self.mainPN[1].grad,

                self.mainPN[0].second_order,
                self.mainPN[1].second_order,
                self.mainPN[0].grad_sum,
                self.mainPN[1].grad_sum,
            ],
            feed_dict=feed_dict)

        if self.warmup:
            update1 = update1 * self.warmup_step_n / self.warmup
            update2 = update2 * self.warmup_step_n / self.warmup
        if self.lr_decay:
            update1 = update1 * lr_decay
            update2 = update2 * lr_decay

        update1_to_log = update1 / self.bs_mul
        update2_to_log = update2 / self.bs_mul
        print(len(update1), len(update2), "update1, update2", sum(update1_to_log), sum(update2_to_log))
        self.update1_list.append(sum(update1_to_log))
        self.update2_list.append(sum(update2_to_log))

        update(self.mainPN, self.lr, update1 / self.bs_mul, update2 / self.bs_mul)

        if self.playing_against_exploiter:
            dqn_exploiter_stats = self.train_dqn_policy()

        updated = True
        print('update params')

        self.episodes_run_counter[agent] = self.episodes_run_counter[agent] * 0
        self.episodes_actions[agent] = self.episodes_actions[agent] * 0
        self.episodes_reward[agent] = self.episodes_reward[agent] * 0

        updated = False
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

        action_log = np.sum(self.aList[-self.summary_len:], 0)
        actions_freq = {f"player_red_act_{i}": action_log[i] / to_plot['n_steps_per_summary']
                        for i in range(0, 4, 1)}
        actions_freq.update({f"player_blue_act_{i - 4}": action_log[i] / to_plot['n_steps_per_summary']
                             for i in range(4, 8, 1)})

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
            "player_1_update_sum": sum(self.update1_list) / self.summary_len,
            "player_2_update_sum": sum(self.update2_list) / self.summary_len,
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

        self.update1_list.clear()
        self.update2_list.clear()

        to_report = {"episodes_total": self.timestep}
        to_report["finished"] = False if self.timestep < self.num_episodes else True
        to_report.update(to_plot)
        to_report.update(last_info)
        to_report.update(training_info)
        to_report.update(actions_freq)
        if self.playing_against_exploiter:
            to_report.update(dqn_exploiter_stats["learner_stats"])
        return to_report

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
                    self.mainPN_step[agent_to_use].lstm_state: self.lstm_state[agent_to_use],
                    self.mainPN_step[agent_to_use].is_training: False,
                }
            )
            self.lstm_state[agent_to_use] = lstm_s
        action = self._post_process_action(a)

        state_out = []
        extra_fetches = {}
        return action, state_out, extra_fetches

    def reset_compute_actions_state(self):
        self.lstm_state = []
        for agent in range(self.n_agents):
            self.lstm_state.append(np.zeros((self.batch_size, self.h_size[agent] * 2)))
