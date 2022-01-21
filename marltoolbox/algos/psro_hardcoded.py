# Code modified from: https://github.com/deepmind/open_spiel/blob/master/open_spiel/python/examples/psro_v2_example.py


import copy
import os
import pickle
import collections

import numpy as np
import torch
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.agents.pg import PGTorchPolicy, DEFAULT_CONFIG
from ray.rllib.evaluation.sample_batch_builder import (
    MultiAgentSampleBatchBuilder,
    SampleBatchBuilder,
)
from ray.rllib.examples.policy.random_policy import RandomPolicy
from ray.rllib.policy.policy import clip_action
from ray.rllib.env.base_env import _DUMMY_AGENT_ID

from marltoolbox.experiments.tune_class_api.various_algo_meta_game import (
    _compute_policy_wt_alpha_rank,
)


class MySampleBatchBuilder(SampleBatchBuilder):
    def __init__(self):
        # deprecation_warning(
        #     old="SampleBatchBuilder",
        #     new="child class of `SampleCollector`",
        #     error=False)
        self.buffers = collections.defaultdict(list)
        self.count = 0


class MyMultiAgentSampleBatchBuilder(MultiAgentSampleBatchBuilder):
    def __init__(self, policy_map, clip_rewards, callbacks):
        # deprecation_warning(old="MultiAgentSampleBatchBuilder", error=False)
        self.policy_map = policy_map
        self.clip_rewards = clip_rewards
        # Build the Policies' SampleBatchBuilders.
        self.policy_builders = {
            k: MySampleBatchBuilder() for k in policy_map.keys()
        }
        # Whenever we observe a new agent, add a new SampleBatchBuilder for
        # this agent.
        self.agent_builders = {}
        # Internal agent-to-policy map.
        self.agent_to_policy = {}
        self.callbacks = callbacks
        # Number of "inference" steps taken in the environment.
        # Regardless of the number of agents involved in each of these steps.
        self.count = 0

    def add_values(self, agent_id, policy_id, **values) -> None:
        """Add the given dictionary (row) of values to this batch.

        Args:
            agent_id (obj): Unique id for the agent we are adding values for.
            policy_id (obj): Unique id for policy controlling the agent.
            values (dict): Row of values to add for this agent.
        """

        if agent_id not in self.agent_builders:
            self.agent_builders[agent_id] = MySampleBatchBuilder()
            self.agent_to_policy[agent_id] = policy_id

        # Include the current agent id for multi-agent algorithms.
        if agent_id != _DUMMY_AGENT_ID:
            values["agent_id"] = agent_id

        self.agent_builders[agent_id].add_values(**values)


class PSROTrainer(tune.Trainable):
    def setup(self, config):
        self.config = config
        self.env = self.config["env_class"](self.config["env_config"])
        self.eval_cell_over_n_epi = self.config["eval_cell_over_n_epi"]
        self.batch_size = self.config["oracle_config"]["train_batch_size"]
        self.policy_ids = self.config["env_config"]["players_ids"]
        self.n_steps_by_epi = self.config["env_config"]["n_steps_by_epi"]
        self.train_oracle_n_epi = self.config["train_oracle_n_epi"]
        self.num_iterations = self.config["num_iterations"]
        self.training = self.config["training"]
        self.verbose = self.config["verbose"]
        self.oracle_config = self.config["oracle_config"]
        self.center_returns = self.config["center_returns"]
        self.to_report_base_game = {}
        self.meta_policies = {
            policy_id: self._init_meta_player()
            for policy_id in self.policy_ids
        }
        self.init_obs = self.env.reset()
        if self.training:
            self._init_meta_payoff_matrix()

    def _init_meta_player(self):
        meta_policy = [1.0]
        return {
            "meta_policy": self._create_policy_sampler(meta_policy),
            "policies": [
                RandomPolicy(
                    self.env.OBSERVATION_SPACE, self.env.ACTION_SPACE, {}
                )
            ],
        }

    def _create_policy_sampler(self, meta_policy_proba):
        meta_policy_proba = torch.tensor(meta_policy_proba)
        return torch.distributions.Categorical(probs=meta_policy_proba)

    def _init_meta_payoff_matrix(self):
        self.meta_game_payoff_matrix = np.zeros((1, 1, len(self.policy_ids)))
        assert len(self.meta_policies[self.policy_ids[0]]["policies"]) == 1
        assert len(self.meta_policies[self.policy_ids[1]]["policies"]) == 1
        self.meta_game_payoff_matrix[0, 0, :] = self._compute_joint_payoffs(
            self.meta_policies[self.policy_ids[0]]["policies"][0],
            self.meta_policies[self.policy_ids[1]]["policies"][0],
        )

    def _compute_joint_payoffs(self, policy_pl_1, policy_pl_2):
        # self._init_to_report(in_cell_eval=True)
        self._reset_total_welfare()
        self.players = {
            self.policy_ids[0]: policy_pl_1,
            self.policy_ids[1]: policy_pl_2,
        }
        self.multi_agent_batch_builder = self._init_batch_builder()
        for epi_n in range(self.eval_cell_over_n_epi):
            self._play_one_episode()

        total_r_pl_1, total_r_pl_2 = self.total_welfare
        self._reset_total_welfare()
        n_steps_playerd = int(self.eval_cell_over_n_epi * self.n_steps_by_epi)
        mean_r_pl1 = total_r_pl_1 / n_steps_playerd
        mean_r_pl2 = total_r_pl_2 / n_steps_playerd
        return [mean_r_pl1, mean_r_pl2]

    def step(self):
        self._init_to_report(in_cell_eval=False)
        for policy_id in self.policy_ids:
            br_policy = self.train_one_br(policy_id)
            self.meta_policies[policy_id]["policies"].append(br_policy)
        self.compute_new_meta_policies()

        self.to_report["finished"] = (
            False
            if self.training_iteration < (self.num_iterations - 1)
            else True
        )
        self._fill_to_report()
        return self.to_report

    def _init_to_report(self, in_cell_eval, in_rllib_eval=False):
        self.to_report = {
            "training_iteration": self.training_iteration,
            "in_cell_eval": in_cell_eval,
            "in_rllib_eval": in_rllib_eval,
        }

    def _fill_to_report(self):
        # Add base policies actions
        for player_id, meta_policy in self.meta_policies.items():
            self.to_report[f"{player_id}_meta_policy"] = meta_policy[
                "meta_policy"
            ].probs
            for pi_idx, base_policy in enumerate(meta_policy["policies"]):
                actions = self._helper_compute_action(
                    base_policy, obs=self.init_obs[player_id]
                )
                self.to_report[f"{player_id}_pi_idx_{pi_idx}_act"] = actions

        # Add base policies actions
        if hasattr(self, "meta_game_payoff_matrix"):
            self.to_report[
                f"meta_game_payoff_matrix_pl0"
            ] = self.meta_game_payoff_matrix[..., 0]
            self.to_report[
                f"meta_game_payoff_matrix_pl1"
            ] = self.meta_game_payoff_matrix[..., 1]

    def compute_new_meta_policies(self):

        self._fill_new_meta_payoff_table()

        policy_player_1, policy_player_2 = _compute_policy_wt_alpha_rank(
            [
                self.meta_game_payoff_matrix[..., 0],
                self.meta_game_payoff_matrix[..., 1],
            ]
        )
        policy_player_1, policy_player_2 = self._clamp_policies_normalize(
            [policy_player_1, policy_player_2]
        )
        self.meta_policies[self.policy_ids[0]][
            "meta_policy"
        ] = self._create_policy_sampler(policy_player_1)
        self.meta_policies[self.policy_ids[1]][
            "meta_policy"
        ] = self._create_policy_sampler(policy_player_2)

    def _clamp_policies_normalize(self, policies):
        pi_clamped = []
        for pi in policies:
            assert not (any(pi > 1.01) or any(pi < -0.01)), f"pi {pi}"
            pi = pi / pi.sum()
            pi = pi.clamp(min=0.0, max=1.0)
            pi_clamped.append(pi)
        return pi_clamped

    def _fill_new_meta_payoff_table(self):
        prev_mat_shape = self.meta_game_payoff_matrix.shape
        new_mat_shape = list(prev_mat_shape)
        new_mat_shape[0] += 1
        new_mat_shape[1] += 1

        new_payoff_mat = np.zeros(new_mat_shape)
        new_payoff_mat[:-1, :-1, :] = self.meta_game_payoff_matrix
        self.meta_game_payoff_matrix = new_payoff_mat

        # Fill last row
        for pl_2_idx in range(new_payoff_mat.shape[1]):
            pl_1_idx = new_payoff_mat.shape[0] - 1
            self._fill_meta_game_cell(pl_1_idx, pl_2_idx)

        # Fill last col
        for pl_1_idx in range(new_payoff_mat.shape[0] - 1):
            pl_2_idx = new_payoff_mat.shape[1] - 1
            self._fill_meta_game_cell(pl_1_idx, pl_2_idx)

        print("meta_game_payoff_matrix")
        print("pl1", self.meta_game_payoff_matrix[..., 0])
        print("pl2", self.meta_game_payoff_matrix[..., 1])

    def _fill_meta_game_cell(self, pl_1_idx, pl_2_idx):
        policy_pl_1 = self.meta_policies[self.policy_ids[0]]["policies"][
            pl_1_idx
        ]
        policy_pl_2 = self.meta_policies[self.policy_ids[1]]["policies"][
            pl_2_idx
        ]
        self.meta_game_payoff_matrix[
            pl_1_idx, pl_2_idx, :
        ] = self._compute_joint_payoffs(policy_pl_1, policy_pl_2)

    def train_one_br(self, policy_id):
        policy = self._init_pg_policy()
        self._get_base_players(policy, policy_id)
        self.multi_agent_batch_builder = self._init_batch_builder()
        self.to_report_base_game = {}
        self._reset_total_welfare()
        self.n_steps_in_batch = 0
        for i in range(self.train_oracle_n_epi):
            self._play_one_episode()
            self.multi_agent_batch_builder.postprocess_batch_so_far()
            self.n_steps_in_batch += 1
            if self.n_steps_in_batch == self.batch_size:
                self._optimize_weights(policy, policy_id)
                self._get_base_players(policy, policy_id)
                self.multi_agent_batch_builder = self._init_batch_builder()
                if self.verbose:
                    print(
                        policy_id,
                        "self.total_welfare",
                        self.total_welfare[0] / self.batch_size,
                        self.total_welfare[1] / self.batch_size,
                    )
                self._reset_total_welfare()
                if self.verbose:
                    print(
                        policy_id,
                        "self.to_report_base_game",
                        self.to_report_base_game,
                    )
                self.to_report_base_game = {}
                self.n_steps_in_batch = 0

        return policy

    def _reset_total_welfare(self):
        self.total_welfare = [0.0] * len(self.policy_ids)

    def _init_batch_builder(self):
        return MyMultiAgentSampleBatchBuilder(
            policy_map={
                player_id: player for player_id, player in self.players.items()
            },
            clip_rewards=False,
            callbacks=DefaultCallbacks(),
        )

    def _init_pg_policy(self):
        pg_policy = PGTorchPolicy(
            self.env.OBSERVATION_SPACE,
            self.env.ACTION_SPACE,
            self.oracle_config,
        )
        # for v in dir(pg_policy):
        #     if isinstance(v, torch.nn.Module):
        #         print(v)
        return pg_policy

    def _get_base_players(self, policy, policy_id):
        opp_policy = self._sample_opponent_policy(policy_id)
        self.players = {}
        for one_policy_id in self.policy_ids:
            if one_policy_id == policy_id:
                self.players[one_policy_id] = policy
            else:
                self.players[one_policy_id] = opp_policy

    def _sample_opponent_policy(self, policy_id):
        self.players = {}
        for opp_policy_id in self.policy_ids:
            if opp_policy_id != policy_id:
                return self._sample_base_policy(opp_policy_id)

    def _sample_base_policy(self, policy_id):
        policy_idx = self.meta_policies[policy_id]["meta_policy"].sample()
        self.to_report[f"base_pi_idx_{policy_id}"] = policy_idx
        return self.meta_policies[policy_id]["policies"][policy_idx]

    def _play_one_episode(self):
        obs_before_act = self.env.reset()
        done = {"__all__": False}
        while not done["__all__"]:
            obs_after_act, actions, rewards, done = self._play_one_step(
                obs_before_act
            )
            self._add_step_in_batch_builder_buffer(
                obs_before_act, actions, rewards, done
            )
            obs_before_act = obs_after_act

    def _play_one_step(self, obs_before_act):
        actions = {
            player_id: self._helper_compute_action(
                player_policy, obs_before_act[player_id]
            )
            for player_id, player_policy in self.players.items()
        }
        obs_after_act, rewards, done, info = self.env.step(actions)
        self.to_report_base_game.update(info)

        return obs_after_act, actions, rewards, done

    def _helper_compute_action(self, player_policy, obs):
        return clip_action(
            player_policy.compute_actions([obs])[0][0],
            player_policy.action_space_struct,
        )

    def _add_step_in_batch_builder_buffer(
        self, obs_before_act, actions, rewards, done
    ):
        for i, policy_id in enumerate(self.policy_ids):
            self.total_welfare[i] += rewards[policy_id]

            step_player_values = {
                "eps_id": self.training_iteration,
                "obs": obs_before_act[policy_id],
                "actions": actions[policy_id],
                "rewards": rewards[policy_id],
                "dones": done[policy_id],
            }
            self.multi_agent_batch_builder.add_values(
                agent_id=policy_id, policy_id=policy_id, **step_player_values
            )

    def _optimize_weights(self, policy, policy_id):
        multiagent_batch = self.multi_agent_batch_builder.build_and_reset()
        if self.center_returns:
            multiagent_batch = self._center_reward(multiagent_batch, policy_id)
        stats = policy.learn_on_batch(
            multiagent_batch.policy_batches[policy_id]
        )

    def _center_reward(self, multiagent_batch, player_id):
        multiagent_batch.policy_batches[player_id]["rewards"] = (
            multiagent_batch.policy_batches[player_id]["rewards"]
            - multiagent_batch.policy_batches[player_id]["rewards"].mean()
        )
        return multiagent_batch

    def save_checkpoint(self, checkpoint_dir):
        all_weights = self.get_weights()
        checkpoint_dir = os.path.expanduser(checkpoint_dir)
        checkpoint_path = os.path.join(checkpoint_dir, "psro_hardcoded_ckpt.p")
        with open(checkpoint_path, "wb") as f:
            pickle.dump(all_weights, f, protocol=pickle.HIGHEST_PROTOCOL)
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path):
        with open(checkpoint_path, "rb") as f:
            all_weights = pickle.load(f)
        self.set_weights(all_weights)

    def get_weights(self):
        save_object = {
            k: {"meta_policy": v["meta_policy"]}
            for k, v in self.meta_policies.items()
        }

        for policy_id, meta_policy in save_object.items():
            weights = []
            for base_policy in self.meta_policies[policy_id]["policies"]:
                if not isinstance(base_policy, RandomPolicy):
                    weights.append(base_policy.get_weights())
                else:
                    weights.append(None)
            save_object[policy_id]["policies"] = weights
        return save_object

    def set_weights(self, save_object):
        meta_policies_save = copy.deepcopy(save_object)
        for policy_id, meta_policy in self.meta_policies.items():
            for pi_idx, weights in enumerate(
                meta_policies_save[policy_id]["policies"]
            ):
                if not len(meta_policy["policies"]) > pi_idx:
                    new_policy = self._init_pg_policy()
                    meta_policy["policies"].append(new_policy)
                base_policy = meta_policy["policies"][pi_idx]
                if not isinstance(base_policy, RandomPolicy):
                    base_policy.set_weights(weights)
            self.meta_policies[policy_id]["meta_policy"] = meta_policies_save[
                policy_id
            ]["meta_policy"]

    def cleanup(self):
        super().cleanup()

    def compute_actions(self, policy_id: str, obs_batch: list):
        assert len(obs_batch) == 1

        for single_obs in obs_batch:
            player_policy = self.base_policies[policy_id]
            obs = self._preprocess_obs(single_obs)
            a = self._helper_compute_action(player_policy, obs)

        action = self._post_process_action(a)

        state_out = []
        extra_fetches = {}
        return action, state_out, extra_fetches

    def _preprocess_obs(self, single_obs):
        # single_obs = single_obs[None, ...]  # add batch dim
        return np.array(single_obs)

    def _post_process_action(self, action):
        return action[None, ...]  # add batch dim

    def on_episode_start(self):
        self._init_to_report(in_cell_eval=False, in_rllib_eval=True)
        self.base_policies = {}
        for policy_id in self.policy_ids:
            self.base_policies[policy_id] = self._sample_base_policy(policy_id)
