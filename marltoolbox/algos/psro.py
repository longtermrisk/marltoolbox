# Code modified from: https://github.com/deepmind/open_spiel/blob/master/open_spiel/python/examples/psro_v2_example.py


import time

from absl import app
from absl import flags
import numpy as np

# pylint: disable=g-bad-import-order
import pyspiel
import tensorflow.compat.v1 as tf

# pylint: enable=g-bad-import-order

from open_spiel.python import policy
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import exploitability
from open_spiel.python.algorithms import get_all_states
from open_spiel.python.algorithms import policy_aggregator
from open_spiel.python.algorithms.psro_v2 import best_response_oracle
from open_spiel.python.algorithms.psro_v2 import psro_v2
from open_spiel.python.algorithms.psro_v2 import rl_oracle
from open_spiel.python.algorithms.psro_v2 import rl_policy
from open_spiel.python.algorithms.psro_v2 import strategy_selectors
from open_spiel.python.games import simple_bargaining, kuhn_poker, tic_tac_toe
from marltoolbox.utils.miscellaneous import get_random_seeds

from ray import tune

from open_spiel.python.examples.psro_v2_example import (
    _print_base_policies,
    print_policy_analysis,
)


class PSROTrainer(tune.Trainable):
    def setup(self, config):
        self.seed = config["seed"]
        self.n_players = config["n_players"]
        self.game_name = config["game_name"]
        self.oracle_type = config["oracle_type"]
        self.training_strategy_selector = config["training_strategy_selector"]
        self.rectifier = config["rectifier"]
        self.sims_per_entry = config["sims_per_entry"]
        self.number_policies_selected = config["number_policies_selected"]
        self.meta_strategy_method = config["meta_strategy_method"]
        self.symmetric_game = config["symmetric_game"]
        self.verbose = config["verbose"]
        self.loss_str = config["loss_str"]
        self.hidden_layer_size = config["hidden_layer_size"]
        self.n_hidden_layers = config["n_hidden_layers"]
        self.batch_size = config["batch_size"]
        self.entropy_cost = config["entropy_cost"]
        self.critic_learning_rate = config["critic_learning_rate"]
        self.pi_learning_rate = config["pi_learning_rate"]
        self.num_q_before_pi = config["num_q_before_pi"]
        self.optimizer_str = config["optimizer_str"]
        self.number_training_episodes = config["number_training_episodes"]
        self.self_play_proportion = config["self_play_proportion"]
        self.sigma = config["sigma"]
        self.dqn_learning_rate = config["dqn_learning_rate"]
        self.update_target_network_every = config[
            "update_target_network_every"
        ]
        self.learn_every = config["learn_every"]
        self.num_iterations = config["num_iterations"]

        np.random.seed(self.seed)
        try:
            game = pyspiel.load_game_as_turn_based(
                self.game_name,
                {"players": pyspiel.GameParameter(self.n_players)},
            )
        except pyspiel.SpielError:
            game = pyspiel.load_game_as_turn_based(self.game_name)

        self.env = rl_environment.Environment(game)

        # Initialize oracle and agents
        with tf.Session() as sess:
            if self.oracle_type == "DQN":
                oracle, agents = self.init_dqn_responder(sess, self.env)
            elif self.oracle_type == "PG":
                oracle, agents = self.init_pg_responder(sess, self.env)
            elif self.oracle_type == "BR":
                oracle, agents = self.init_br_responder(self.env)
            sess.run(tf.global_variables_initializer())

        sample_from_marginals = (
            True  # TODO(somidshafiei) set False for alpharank
        )
        training_strategy_selector = (
            self.training_strategy_selector or strategy_selectors.probabilistic
        )

        self.g_psro_solver = psro_v2.PSROSolver(
            self.env.game,
            oracle,
            initial_policies=agents,
            training_strategy_selector=training_strategy_selector,
            rectifier=self.rectifier,
            sims_per_entry=self.sims_per_entry,
            number_policies_selected=self.number_policies_selected,
            meta_strategy_method=self.meta_strategy_method,
            prd_iterations=50000,
            prd_gamma=1e-10,
            sample_from_marginals=sample_from_marginals,
            symmetric_game=self.symmetric_game,
        )

        self.start_time = time.time()

    def step(self):
        to_report = {"training_iteration": self.training_iteration}
        if self.verbose:
            print("Iteration : {}".format(self.training_iteration))
            print("Time so far: {}".format(time.time() - self.start_time))
        self.g_psro_solver.iteration()
        meta_game = self.g_psro_solver.get_meta_game()
        meta_probabilities = self.g_psro_solver.get_meta_strategies()
        policies = self.g_psro_solver.get_policies()

        _print_base_policies(policies)

        if self.verbose:
            print("Meta game : {}".format(meta_game))
            print("Probabilities : {}".format(meta_probabilities))
            to_report["meta_probabilities"] = meta_probabilities
            to_report["meta_game"] = meta_game
        # The following lines only work for sequential games for the moment.
        if (
            self.env.game.get_type().dynamics
            == pyspiel.GameType.Dynamics.SEQUENTIAL
        ):
            aggregator = policy_aggregator.PolicyAggregator(self.env.game)
            aggr_policies = aggregator.aggregate(
                range(self.n_players), policies, meta_probabilities
            )

            exploitabilities, expl_per_player = exploitability.nash_conv(
                self.env.game, aggr_policies, return_only_nash_conv=False
            )

            _ = print_policy_analysis(policies, self.env.game, self.verbose)
            if self.verbose:
                print("Exploitabilities : {}".format(exploitabilities))
                print(
                    "Exploitabilities per player : {}".format(expl_per_player)
                )
                to_report["exploitabilities"] = exploitabilities
                to_report["expl_per_player"] = expl_per_player

        to_report["finished"] = (
            False if self.training_iteration < self.num_iterations else True
        )
        return to_report

    def init_pg_responder(self, sess, env):
        """Initializes the Policy Gradient-based responder and agents."""
        info_state_size = env.observation_spec()["info_state"][0]
        num_actions = env.action_spec()["num_actions"]

        agent_class = rl_policy.PGPolicy

        agent_kwargs = {
            "session": sess,
            "info_state_size": info_state_size,
            "num_actions": num_actions,
            "loss_str": self.loss_str,
            "loss_class": False,
            "hidden_layers_sizes": [self.hidden_layer_size]
            * self.n_hidden_layers,
            "batch_size": self.batch_size,
            "entropy_cost": self.entropy_cost,
            "critic_learning_rate": self.critic_learning_rate,
            "pi_learning_rate": self.pi_learning_rate,
            "num_critic_before_pi": self.num_q_before_pi,
            "optimizer_str": self.optimizer_str,
        }
        oracle = rl_oracle.RLOracle(
            env,
            agent_class,
            agent_kwargs,
            number_training_episodes=self.number_training_episodes,
            self_play_proportion=self.self_play_proportion,
            sigma=self.sigma,
        )

        agents = [
            agent_class(  # pylint: disable=g-complex-comprehension
                env, player_id, **agent_kwargs
            )
            for player_id in range(self.n_players)
        ]
        for agent in agents:
            agent.freeze()
        return oracle, agents

    def init_br_responder(self, env):
        """Initializes the tabular best-response based responder and agents."""
        random_policy = policy.TabularPolicy(env.game)
        oracle = best_response_oracle.BestResponseOracle(
            game=env.game, policy=random_policy
        )
        agents = [random_policy.__copy__() for _ in range(self.n_players)]
        return oracle, agents

    def init_dqn_responder(self, sess, env):
        """Initializes the Policy Gradient-based responder and agents."""
        state_representation_size = env.observation_spec()["info_state"][0]
        num_actions = env.action_spec()["num_actions"]

        agent_class = rl_policy.DQNPolicy
        agent_kwargs = {
            "session": sess,
            "state_representation_size": state_representation_size,
            "num_actions": num_actions,
            "hidden_layers_sizes": [self.hidden_layer_size]
            * self.n_hidden_layers,
            "batch_size": self.batch_size,
            "learning_rate": self.dqn_learning_rate,
            "update_target_network_every": self.update_target_network_every,
            "learn_every": self.learn_every,
            "optimizer_str": self.optimizer_str,
        }
        oracle = rl_oracle.RLOracle(
            env,
            agent_class,
            agent_kwargs,
            number_training_episodes=self.number_training_episodes,
            self_play_proportion=self.self_play_proportion,
            sigma=self.sigma,
        )

        agents = [
            agent_class(  # pylint: disable=g-complex-comprehension
                env, player_id, **agent_kwargs
            )
            for player_id in range(self.n_players)
        ]
        for agent in agents:
            agent.freeze()
        return oracle, agents

    # def save_checkpoint(self, checkpoint_dir):
    #     path = os.path.join(checkpoint_dir, "checkpoint.json")
    #     tf_checkpoint_path = os.path.join(checkpoint_dir, "checkpoint")
    #     tf_checkpoint_dir, tf_checkpoint_filename = os.path.split(
    #         tf_checkpoint_path
    #     )
    #     checkpoint = {
    #         "timestep": self.timestep,
    #         "tf_checkpoint_dir": tf_checkpoint_dir,
    #         "tf_checkpoint_filename": tf_checkpoint_filename,
    #     }
    #     with open(path, "w") as f:
    #         json.dump(checkpoint, f, sort_keys=True, indent=4)
    #
    #     # TF v1
    #     save_path = self.saver.save(self.sess, f"{tf_checkpoint_path}.ckpt")
    #
    #     return path
    #
    # def load_checkpoint(self, checkpoint_path):
    #
    #     checkpoint_path = os.path.expanduser(checkpoint_path)
    #     print("Loading Model...", checkpoint_path)
    #     with open(checkpoint_path, "r") as f:
    #         checkpoint = json.load(f)
    #     print("checkpoint", checkpoint)
    #
    #     # Support VM and local (manual) loading
    #     tf_checkpoint_dir, _ = os.path.split(checkpoint_path)
    #     print("tf_checkpoint_dir", tf_checkpoint_dir)
    #     ckpt = tf.train.get_checkpoint_state(
    #         tf_checkpoint_dir,
    #         latest_filename=f'{checkpoint["tf_checkpoint_filename"]}',
    #     )
    #     tail, head = os.path.split(ckpt.model_checkpoint_path)
    #     ckpt.model_checkpoint_path = os.path.join(tf_checkpoint_dir, head)
    #     self.saver.restore(self.sess, ckpt.model_checkpoint_path)
    #
    # def cleanup(self):
    #     self.sess.close()
    #     super().cleanup()
    #
    # def compute_actions(self, policy_id: str, obs_batch: list):
    #     # because of the LSTM
    #     assert len(obs_batch) == 1
    #
    #     for single_obs in obs_batch:
    #         agent_to_use = self._get_agent_to_use(policy_id)
    #         obs = self._preprocess_obs(single_obs, agent_to_use)
    #         a, lstm_s = self.sess.run(
    #             [
    #                 self.mainPN_step[agent_to_use].predict,
    #                 self.mainPN_step[agent_to_use].lstm_state_output,
    #             ],
    #             feed_dict={
    #                 self.mainPN_step[agent_to_use].state_input: obs,
    #                 self.mainPN_step[agent_to_use].lstm_state: self.lstm_state[
    #                     agent_to_use
    #                 ],
    #                 self.mainPN_step[agent_to_use].is_training: False,
    #             },
    #         )
    #         self.lstm_state[agent_to_use] = lstm_s
    #     action = self._post_process_action(a)
    #
    #     state_out = []
    #     extra_fetches = {}
    #     return action, state_out, extra_fetches
    #
    # def _get_agent_to_use(self, policy_id):
    #     if policy_id == "player_red":
    #         agent_n = 0
    #     elif policy_id == "player_blue":
    #         agent_n = 1
    #     else:
    #         raise ValueError(f"policy_id {policy_id}")
    #     return agent_n
    #
    # def _preprocess_obs(self, single_obs, agent_to_use):
    #     single_obs = single_obs[None, ...]  # add batch dim
    #
    #     # Compensate for the batch norm not in evaluation mode
    #     while len(self.obs_batch) < self.batch_size:
    #         self.obs_batch.append(single_obs)
    #     self.obs_batch.append(single_obs)
    #     single_obs = np.concatenate(list(self.obs_batch), axis=0)
    #     return single_obs
    #
    # def _post_process_action(self, action):
    #     # Compensate for the batch norm not in evaluation mode
    #     if isinstance(action, Iterable):
    #         action = action[-1]
    #
    #     return action[None, ...]  # add batch dim
    #
    # def reset_compute_actions_state(self):
    #     self.lstm_state = []
    #     for agent in range(self.n_agents):
    #         self.lstm_state.append(
    #             np.zeros((self.batch_size, self.h_size[agent] * 2))
    #         )
