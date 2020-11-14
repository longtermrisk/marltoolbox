import os

import ray
import torch
from ray import tune
from ray.rllib.agents import Trainer
from ray.rllib.agents.a3c import A2CTrainer
from ray.rllib.agents.a3c.a3c_torch_policy import A3CTorchPolicy, ValueNetworkMixin, loss_and_entropy_stats
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.policy import Policy
from ray.rllib.utils.typing import TrainerConfigDict

from marltoolbox.algos.lola.LOLA_DICE_torch_unofficial import LolaDiceTorchPolicyMixin, LolaDiceTorchTrainerMixin, \
    init_lola, LOLACallbacks
from marltoolbox.envs.matrix_SSD import IteratedPrisonersDilemma
from marltoolbox.utils.logging import stats_fn_wt_additionnal_logs


class AllCallbacks(LOLACallbacks, DefaultCallbacks):
    pass


def after_init(trainer: Trainer) -> None:
    init_lola(trainer)
    init_models_weights(trainer)


def init_models_weights(trainer: Trainer) -> None:
    def torch_init_all_close_to_zero(policy, policy_id):
        [torch.nn.init.normal_(p, mean=0.0, std=0.1) for p in policy.model.parameters()]

    trainer.workers.foreach_policy(torch_init_all_close_to_zero)

LR = 0.0  # 0.04
MOMENTUM = 0.9


def create_custom_optimizer(policy: Policy, trainer_config: TrainerConfigDict) -> torch.optim.Optimizer:
    return torch.optim.SGD(policy.model.parameters(), lr=LR, momentum=MOMENTUM)


if __name__ == "__main__":
    ray.init(num_cpus=1, num_gpus=0)

    stop = {
        "training_iteration": 500,
    }

    env_config = {
        "players_ids": ["player_row", "player_col"],
        "max_steps": 150,
    }

    trainer_config_update = {
        "env": IteratedPrisonersDilemma,
        "env_config": env_config,
        "multiagent": {
            "policies": {
                env_config["players_ids"][0]: (None,
                                               IteratedPrisonersDilemma.OBSERVATION_SPACE,
                                               IteratedPrisonersDilemma.ACTION_SPACE,
                                               {"framework": "torch"}),
                env_config["players_ids"][1]: (None,
                                               IteratedPrisonersDilemma.OBSERVATION_SPACE,
                                               IteratedPrisonersDilemma.ACTION_SPACE,
                                               {"framework": "torch"}),

            },
            "policy_mapping_fn": (
                lambda agent_id: f"{agent_id}"),  # The list of agent_id is the possible keys of the env dicts
        },

        # === Exploration Settings ===
        # Default exploration behavior, iff `explore`=None is passed into
        # compute_action(s).
        # Set to False for no exploration behavior (e.g., for evaluation).
        "explore": True,
        # Provide a dict specifying the Exploration object's config.
        "exploration_config": {
            # The Exploration class to use. In the simplest case, this is the name
            # (str) of any class present in the `rllib.utils.exploration` package.
            # You can also provide the python class directly or the full location
            # of your class (e.g. "ray.rllib.utils.exploration.epsilon_greedy.
            # EpsilonGreedy").
            "type": "StochasticSampling",
            # Can't use EpsilonGreedy sinec then log_proba_action = 0.0
            # Add constructor kwargs here (if any).
        },

        # The dataflow here can vary per algorithm. For example, PPO further
        # divides the train batch into minibatches for multi-epoch SGD.
        "rollout_fragment_length": 1,
        # Whether to rollout "complete_episodes" or "truncate_episodes" to
        # `rollout_fragment_length` length unrolls. Episode truncation guarantees
        # evenly sized batches, but increases variance as the reward-to-go will
        # need to be estimated at truncation boundaries.
        "batch_mode": "truncate_episodes",

        # === Settings for the Trainer process ===
        # Training batch size, if applicable. Should be >= rollout_fragment_length.
        # Samples batches will be concatenated together to a batch of this size,
        # which is then passed to SGD.
        "train_batch_size": 60,

        # === Model ===
        # # Minimum env steps to optimize for per train call. This value does
        # # not affect learning, only the length of iterations.
        # "timesteps_per_iteration": 1,
        "model": {
            # Number of hidden layers for fully connected net
            "fcnet_hiddens": [],
            # Nonlinearity for fully connected net (tanh, relu)
            "fcnet_activation": "relu",
        },

        "framework": "torch",

        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        # Set the ray.rllib.* log level for the agent process and its workers.
        # Should be one of DEBUG, INFO, WARN, or ERROR. The DEBUG level will also
        # periodically print out summaries of relevant internal dataflow (this is
        # also printed out once at startup at the INFO level). When using the
        # `rllib train` command, you can also use the `-v` and `-vv` flags as
        # shorthand for INFO and DEBUG.
        "log_level": "INFO",
        # Callbacks that will be run during various phases of training. See the
        # `DefaultCallbacks` class and `examples/custom_metrics_and_callbacks.py`
        # for more usage information.
        "callbacks": AllCallbacks,

        # === Settings for Rollout Worker processes ===
        # Number of rollout worker actors to create for parallel sampling. Setting
        # this to 0 will force rollouts to be done in the trainer actor.
        "num_workers": 0,
    }

    A3CTorchPolicyWtLolaMixin = A3CTorchPolicy.with_updates(stats_fn=stats_fn_wt_additionnal_logs(loss_and_entropy_stats),
                                                            optimizer_fn=create_custom_optimizer,
                                                            mixins=[LolaDiceTorchPolicyMixin, ValueNetworkMixin])
    LOLAA2CTrainer = A2CTrainer.with_updates(name="LOLAA2CTrainer",
                                             default_policy=A3CTorchPolicyWtLolaMixin,
                                             get_policy_class=None,
                                             after_init=after_init,
                                             mixins=[LolaDiceTorchTrainerMixin])

    results = tune.run(LOLAA2CTrainer, config=trainer_config_update, stop=stop, verbose=1)

    ray.shutdown()
