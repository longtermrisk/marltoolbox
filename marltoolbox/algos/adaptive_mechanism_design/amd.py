##########
# Code from: https://github.com/tobiasbaumann1/Adaptive_Mechanism_Design
##########
import math
import os
import random

import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from ray import tune
from ray.rllib.agents.a3c.a3c import DEFAULT_CONFIG as a3c_config
from ray.rllib.agents.a3c.a3c_tf_policy import A3CTFPolicy
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.evaluation.sample_batch_builder import \
    MultiAgentSampleBatchBuilder

from marltoolbox.algos.adaptive_mechanism_design.agent import \
    Actor_Critic_Agent, Critic_Variant, Simple_Agent, \
    convert_from_rllib_env_format, convert_to_rllib_env_format
from marltoolbox.algos.adaptive_mechanism_design.planning_agent import \
    Planning_Agent
from marltoolbox.envs.matrix_sequential_social_dilemma import \
    define_greed_fear_matrix_game
from marltoolbox.envs.coin_game import CoinGame


def create_population(
        env, n_agents, n_units, use_simple_agents=False,
      lr=0.01, gamma=0.9, weight_decay=0.0, mean_theta=-2.0, std_theta=0.5,
      entropy_coeff=0.001, use_adam_optimizer=True, momentum=0.9,
        use_rllib_polcy=False):

    if use_rllib_polcy:
        agent_config = a3c_config
        agent_config.update({
            # If true, use the Generalized Advantage Estimator (GAE)
            # with a value function, see https://arxiv.org/pdf/1506.02438.pdf.
            "use_gae": False,
            # Size of rollout batch
            "rollout_fragment_length": 10,
            # Max global norm for each gradient calculated by worker
            "grad_clip": 40.0,
            # Learning rate
            "lr": lr * env.max_steps,  # Training every episode instead of every step
            # Learning rate schedule
            "lr_schedule": None,
            # Value Function Loss coefficient
            "vf_loss_coeff": 0.5,
            # Entropy coefficient
            "entropy_coeff": entropy_coeff,
            "gamma": gamma,
            "hiddens": n_units if isinstance(n_units, list) else [n_units],
        })
        l = []
        for i in range(n_agents):
            print("Going to create agent n°", i)
            with tf.variable_scope(f"player_{i}"):
                policy = A3CTFPolicy(
                    obs_space=gym.spaces.Box(low=0, high=1, shape=(env.n_features,), dtype='uint8'),
                    action_space=env.ACTION_SPACE,
                    config=agent_config)
                policy.__setattr__("agent_idx", i)
                l.append(policy)
                print(l[-1]._input_dict)
    else:
        critic_variant = Critic_Variant.CENTRALIZED
        if use_simple_agents:
            l = [Simple_Agent(env,
                              learning_rate=lr,
                              gamma=gamma,
                              agent_idx=i,
                              critic_variant=critic_variant,
                              mean_theta=mean_theta,
                              std_theta=std_theta) for i in range(n_agents)
                 ]
        else:
            l = [Actor_Critic_Agent(env,
                                    learning_rate=lr,
                                    gamma=gamma,
                                    n_units_actor=n_units,
                                    agent_idx=i,
                                    critic_variant=critic_variant,
                                    weight_decay=weight_decay,
                                    std_theta=std_theta,
                                    mean_theta=mean_theta,
                                    entropy_coeff=entropy_coeff,
                                    use_adam_optimizer=use_adam_optimizer,
                                    momentum=momentum) for i in range(n_agents)]
        # Pass list of agents for centralized critic
        if critic_variant is Critic_Variant.CENTRALIZED:
            for agent in l:
                agent.pass_agent_list(l)
    return l


class AdaptiveMechanismDesign(tune.Trainable):

    def _init_algo(self, fear, greed, n_players, use_simple_agents, action_flip_prob,
                   max_reward_strength, value_fn_variant, cost_param, with_redistribution,
                   n_planning_eps, env_config, n_units, n_episodes, env, lr, gamma, weight_decay,
                   loss_mul_planner, mean_theta, with_planner, std_theta, add_state_grad, planner_momentum,
                   planner_clip_norm, entropy_coeff, seed, normalize_planner, no_weights_decay_planner,
                   planner_std_theta_mul, use_adam_optimizer, use_softmax_hot, report_every_n, momentum,
                   weight_decay_pl_mul, square_cost, normalize_against_v, use_v_pl,
                   normalize_against_vp, normalize_vp_separated, use_rllib_polcy, **kwargs):

        if not use_simple_agents:
            speed_ratio = 5.0
            lr = lr / speed_ratio
            loss_mul_planner = loss_mul_planner * speed_ratio ** 2 / 2 / 2
            cost_param = cost_param * 1.5
            if n_units == 64:
                lr = lr / 8

        print("args not used:", kwargs)
        convert_a_to_one_hot = not use_simple_agents

        np.random.seed(seed)
        tf.set_random_seed(seed)
        random.seed(seed)

        if env == "FearGreedMatrix":
            env = define_greed_fear_matrix_game(fear=fear, greed=greed)(env_config)
        elif env == "CoinGame":
            env = CoinGame(env_config)

        env.seed(seed=seed)

        agents = create_population(env, n_players,
                                   use_simple_agents=use_simple_agents, n_units=n_units,
                                   lr=lr, gamma=gamma, weight_decay=weight_decay,
                                   mean_theta=mean_theta, std_theta=std_theta, entropy_coeff=entropy_coeff,
                                   use_adam_optimizer=use_adam_optimizer, momentum=momentum,
                                   use_rllib_polcy=use_rllib_polcy)
        np.random.seed(seed + 1)
        tf.set_random_seed(seed + 1)
        random.seed(seed + 1)

        if with_planner:
            std_theta = std_theta * planner_std_theta_mul
            weight_decay = weight_decay * weight_decay_pl_mul
            if no_weights_decay_planner:
                weight_decay = 0.0
            planning_agent = Planning_Agent(env, agents,
                                            learning_rate=lr,
                                            max_reward_strength=max_reward_strength,
                                            cost_param=cost_param,
                                            with_redistribution=with_redistribution,
                                            value_fn_variant=value_fn_variant,
                                            n_units=n_units, weight_decay=weight_decay,
                                            convert_a_to_one_hot=convert_a_to_one_hot,
                                            loss_mul_planner=loss_mul_planner,
                                            mean_theta=mean_theta,
                                            std_theta=std_theta, planner_clip_norm=planner_clip_norm,
                                            normalize_planner=normalize_planner,
                                            add_state_grad=add_state_grad,
                                            planner_momentum=planner_momentum,
                                            use_adam_optimizer=use_adam_optimizer,
                                            use_softmax_hot=use_softmax_hot,
                                            square_cost=square_cost,
                                            normalize_against_v=normalize_against_v,
                                            use_v_pl=use_v_pl,
                                            normalize_against_vp=normalize_against_vp,
                                            normalize_vp_separated=normalize_vp_separated)
        else:
            planning_agent = None

        self.epi_n = 0
        self.players = agents
        self.env = env
        self.action_flip_prob = action_flip_prob
        self.planning_agent = planning_agent
        self.with_redistribution = with_redistribution
        self.n_planning_eps = n_planning_eps
        self.player_ids = env.players_ids
        self.n_players = n_players
        self.n_episodes = n_episodes
        self.max_reward_strength = max_reward_strength
        self.cost_param = cost_param
        self.value_fn_variant = value_fn_variant
        self.fear = fear
        self.greed = greed
        self.report_every_n = report_every_n
        self.normalize_vp_separated = normalize_vp_separated
        self.use_rllib_polcy = use_rllib_polcy

        self.avg_planning_rewards_per_round = []
        self.episode_reward = []
        self.training_epi_avg_reward = []

        if self.use_rllib_polcy:
            self.multi_agent_batch_builder = MultiAgentSampleBatchBuilder(
                policy_map={idx: player for idx, player in enumerate(self.players)},
                clip_rewards=False,
                callbacks=DefaultCallbacks()
            )

    def setup(self, config):
        print("_init_algo", config)
        self._init_algo(**config)

    def step(self):
        (loss, cost, extra_loss, g_V, planning_rs, mean_v, mean_vp,
        planning_reward_when_pick_own_coin, planning_reward_when_pick_opp_coin,
        planning_reward_when_no_picking, planning_reward_when_specific_action) = [None] * 11

        for _ in range(self.report_every_n):
            self.epi_n += 1
            to_report = {"episodes_total": self.epi_n}

            s_rllib_format = self.env.reset()
            obs_before_act = convert_from_rllib_env_format(s_rllib_format, self.player_ids, state=True,
                                                           n_states=self.env.n_features,
                                                           coin_game=self.env.NAME == "CoinGame",)

            flag = isinstance(obs_before_act, list)

            cum_planning_rs = [0] * len(self.players)
            planning_reward_when_pick_own_coin = [None] * len(self.players)
            planning_reward_when_pick_opp_coin = [None] * len(self.players)
            planning_reward_when_no_picking = [None] * len(self.players)
            planning_reward_when_specific_action = [[None] * self.env.NUM_ACTIONS] * len(self.players)
            if self.use_rllib_polcy:
                prev_r = [0.0] * len(self.players)
                prev_a = [0] * len(self.players)
            done = False
            while not done:
                # choose action based on s
                if self.use_rllib_polcy:
                    actions = [player.compute_actions(obs_before_act[None, ...])[0][0] for player in self.players]
                else:
                    if flag:
                        actions = [player.choose_action(obs_before_act[idx]) for idx, player in enumerate(self.players)]
                    else:
                        actions = [player.choose_action(obs_before_act) for player in self.players]
                actions_rllib_format = convert_to_rllib_env_format(actions, self.player_ids,
                                                                   coin_game=self.env.NAME == "CoinGame")

                # take action and get next s and reward
                s_rllib_format, rewards_rllib_format, done_rllib_format, info_rllib_format = self.env.step(
                    actions_rllib_format)
                obs_after_act = convert_from_rllib_env_format(s_rllib_format, self.player_ids, state=True,
                                                              n_states=self.env.n_features,
                                                              coin_game=self.env.NAME == "CoinGame")
                rewards = convert_from_rllib_env_format(rewards_rllib_format, self.player_ids)
                done = convert_from_rllib_env_format(done_rllib_format, self.player_ids)
                self.episode_reward.append(rewards)

                # perturbed_actions = [(1 - a if np.random.binomial(1, self.action_flip_prob) else a) for a in actions]
                # Make it work for discrete action space of N
                perturbed_actions = []
                for a in actions:
                    if np.random.binomial(1, self.action_flip_prob):
                        perturbed_a = a
                        while perturbed_a == a:
                            perturbed_a = random.randint(0, self.env.n_features - 1)
                            print("perturbed_a == a", perturbed_a, a, perturbed_a == a)
                    else:
                        perturbed_actions.append(a)
                # print("perturbed_actions", perturbed_actions)

                env_rewards = rewards

                if self.planning_agent is not None and self.epi_n < self.n_planning_eps:
                    planning_rs = self.planning_agent.choose_action(obs_before_act, perturbed_actions)
                    if self.with_redistribution:
                        sum_planning_r = sum(planning_rs)
                        mean_planning_r = sum_planning_r / self.n_players
                        planning_rs = [r - mean_planning_r for r in planning_rs]
                    rewards = [sum(r) for r in zip(rewards, planning_rs)]
                    cum_planning_rs = [sum(r) for r in zip(cum_planning_rs, planning_rs)]
                    # Training planning agent
                    # TODO using the past rewards is not working since I perturbate the actions
                    (action, loss, g_Vp, g_V, r_players, cost, extra_loss, l1,
                     mean_v, vp, values, mean_vp) = self.planning_agent.learn(
                        obs_before_act, perturbed_actions,
                        coin_game=self.env.NAME == "CoinGame",
                        env_rewards=env_rewards)

                for idx, player in enumerate(self.players):
                    if self.use_rllib_polcy:
                        step_player_values = {
                            "eps_id": self.epi_n,
                            "obs": obs_before_act,  # [None,...]
                            "new_obs": obs_after_act,
                            "actions": actions[idx],
                            "prev_actions": prev_a[idx],
                            "rewards": rewards[idx],
                            "prev_rewards": prev_r[idx],
                            "dones": done,
                            "vf_preds": player._value(obs_before_act, prev_a[idx], prev_r[idx])
                        }
                        self.multi_agent_batch_builder.add_values(agent_id=idx, policy_id=idx, **step_player_values)

                    else:
                        if flag:
                            critic_loss, advantage = player.learn(obs_before_act[idx], actions[idx], rewards[idx],
                                                                  obs_after_act[idx],
                                                                  obs_before_act,
                                                                  obs_after_act)
                        else:
                            critic_loss, advantage = player.learn(obs_before_act, actions[idx], rewards[idx],
                                                                  obs_after_act)
                        to_report[f"critic_loss_p_{idx}"] = critic_loss[0, 0]
                        to_report[f"advantage_loss_p_{idx}"] = advantage

                    if self.planning_agent is not None:
                        opp_idx = (idx + 1) % 2
                        if env_rewards[idx] == 1.0 and env_rewards[opp_idx] == 0.0:
                            planning_reward_when_pick_own_coin[idx] = planning_rs[idx]
                        if env_rewards[idx] == 1.0 and env_rewards[opp_idx] == -2.0:
                            planning_reward_when_pick_opp_coin[idx] = planning_rs[idx]
                        if env_rewards[idx] == 0.0 and env_rewards[opp_idx] == 0.0:
                            planning_reward_when_no_picking[idx] = planning_rs[idx]

                        planning_reward_when_specific_action[idx][actions[idx]] = planning_rs[idx]

                if self.use_rllib_polcy and done:
                    multiagent_batch = self.multi_agent_batch_builder.build_and_reset()
                    for idx, player in enumerate(self.players):
                        stats = player.learn_on_batch(multiagent_batch.policy_batches[idx])
                        # Does slow down the training
                        # for k, v in stats["learner_stats"].items():
                        #     to_report[f"p{idx}_{k}"] = v

                    prev_r = rewards
                    prev_a = actions
                elif done:
                    for player in self.players:
                        player.learn_at_episode_end()

                # swap s
                obs_before_act = obs_after_act

            if self.planning_agent is not None and self.epi_n < self.n_planning_eps:
                self.avg_planning_rewards_per_round.append([r / self.env.step_count_in_current_episode for r in cum_planning_rs])
            epi_rewards = np.array(self.episode_reward)
            self.training_epi_avg_reward.append(np.mean(epi_rewards, axis=0))
            self.episode_reward.clear()

        to_report = self._add_info_to_log(to_report, actions_rllib_format, info_rllib_format, epi_rewards,
                                          loss, cost, extra_loss, g_V, planning_rs, mean_v, mean_vp,
                                          planning_reward_when_pick_own_coin, planning_reward_when_pick_opp_coin,
                                          planning_reward_when_no_picking, planning_reward_when_specific_action)
        return to_report

    def _add_info_to_log(self, to_report, actions_rllib_format, info_rllib_format, epi_rewards, loss, cost,
                         extra_loss, g_V, planning_rs, mean_v, mean_vp, planning_reward_when_pick_own_coin,
                         planning_reward_when_pick_opp_coin, planning_reward_when_no_picking,
                         planning_reward_when_specific_action):

        for k, v in actions_rllib_format.items():
            to_report[f"act_{k}"] = v
        to_report.update(info_rllib_format)
        to_report["mean_reward_p1"] = np.mean(epi_rewards, axis=0)[0]
        to_report["mean_reward_p2"] = np.mean(epi_rewards, axis=0)[1]
        to_report["mean_reward"] = np.sum(np.mean(epi_rewards, axis=0))

        if self.planning_agent is not None:
            to_report[f"loss_planner"] = loss
            to_report[f"loss_pl_grad"] = cost
            to_report[f"loss_rew_cost"] = extra_loss
            to_report[f"g_V"] = g_V
            planner_weights_norm = self.planning_agent.get_weights_norm()
            to_report[f"planner_weights_norm"] = planner_weights_norm
            to_report["planning_reward_player1"] = planning_rs[0]
            to_report["planning_reward_player2"] = planning_rs[1]
            to_report["mean_v"] = mean_v
            if not self.normalize_vp_separated:
                to_report["mean_vp"] = mean_vp
            else:
                to_report["mean_vp0"] = mean_vp[0]
                to_report["mean_vp1"] = mean_vp[1]

        for idx, player in enumerate(self.players):
            if not self.use_rllib_polcy:
                ac_weights_norm, cr_weights_norm = player.get_weights_norm()
                to_report[f"actor_weights_norm_p_{idx}"] = ac_weights_norm
                to_report[f"critic_weights_norm_p_{idx}"] = cr_weights_norm

            if planning_reward_when_pick_own_coin[idx] is not None:
                to_report[f"pl_rw_p{idx}_pick_own_coin"] = planning_reward_when_pick_own_coin[idx]
            if planning_reward_when_pick_opp_coin[idx] is not None:
                to_report[f"pl_rw_p{idx}_pick_opp_coin"] = planning_reward_when_pick_opp_coin[idx]
            if planning_reward_when_no_picking[idx] is not None:
                to_report[f"pl_rw_p{idx}_no_picking"] = planning_reward_when_no_picking[idx]

            for act_v in range(self.env.NUM_ACTIONS):
                if planning_reward_when_specific_action[idx][act_v] is not None:
                    to_report[f"pl_rw_p{idx}_a{act_v}"] = planning_reward_when_specific_action[idx][act_v]

        return to_report