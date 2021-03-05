import copy

import logging
from typing import List

from torch.nn import CrossEntropyLoss


from ray.rllib.utils.typing import PolicyID, SampleBatchType
from ray.rllib.execution.concurrency_ops import Concurrently
from ray.rllib.execution.replay_buffer import LocalReplayBuffer
from ray.rllib.execution.replay_ops import Replay, StoreToReplayBuffer
from ray.rllib.execution.train_ops import UpdateTargetNetwork
from ray.util.iter import LocalIterator
from ray.rllib.utils import merge_dicts
from ray.rllib.agents.dqn import DQNTorchPolicy, DQNTrainer
from ray.rllib.agents.dqn.dqn import calculate_rr_weights
from ray.rllib.agents.dqn.dqn import DEFAULT_CONFIG as DQN_DEFAULT_CONFIG
from ray.rllib.agents.dqn.dqn import validate_config as validate_config_dqn
from ray.rllib.execution.rollout_ops import ParallelRollouts, ConcatBatches
from ray.rllib.execution.train_ops import TrainOneStep
from ray.rllib.execution.metric_ops import StandardMetricsReporting
from ray.rllib.utils.typing import TrainerConfigDict
from ray.rllib.evaluation.metrics import LEARNER_STATS_KEY
from ray.rllib.evaluation.worker_set import WorkerSet

from marltoolbox.algos import hierarchical
from marltoolbox.algos.supervised_learning import SPLTorchPolicy
from marltoolbox.algos.ltft.ltft_torch_policy import \
    LTFTCallbacks, LTFTTorchPolicy
from marltoolbox.algos.ltft.utils import MyDQNTorchPolicy, sgd_optimizer_spl
from ray.rllib.utils.schedules import PiecewiseSchedule
from marltoolbox.utils import log, miscellaneous, exploration

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = merge_dicts(hierarchical.DEFAULT_CONFIG,
                             DQN_DEFAULT_CONFIG)
DEFAULT_CONFIG.update({
        # Percentile to use when performing the likelihood test on the
        # hypothesis that the opponent is cooperating.
        # If the observes behavior is below this percentile, then consider that
        # the opponent is cooperative.
        "percentile_for_likelihood_test": 95,
        # Number of episode during which to punish after detecting that the
        # opponent was not cooperating during the last episode
        "punishment_time": 1,
        # Min number of epi during which to cooperate after stopping punishment
        "min_coop_epi_after_punishing": 0,
        # Option to add some bias to the likelihood check.
        "defection_threshold": 0.0,
        # Average the defection value over several episodes
        "average_defection_value": True,
        "average_defection_value_len": 20,
        "use_last_step_for_search": True,
        # Length of the history (n of steps) to sample from to simulate the
        # opponent
        "length_of_history": 200,
        # Number of steps sampled to assemble an episode simulating the opponent
        # behavior
        "n_steps_in_bootstrap_replicates": 20,
        # Number of replicates of the simulation of the opponent behavior
        "n_bootstrap_replicates": 50,

        "callbacks": miscellaneous.merge_callbacks(
            LTFTCallbacks,
            log.get_logging_callbacks_class()),

        "sgd_momentum": 0.9,
        'nested_policies': [
            # Here the trainer need to be a DQNTrainer to provide the config
            # for the 3 DQNTorchPolicy
            {"Policy_class": MyDQNTorchPolicy, "config_update": {}},
            {"Policy_class": MyDQNTorchPolicy, "config_update": {}},
            {"Policy_class": MyDQNTorchPolicy, "config_update": {}},
            {"Policy_class": SPLTorchPolicy.with_updates(
                optimizer_fn=sgd_optimizer_spl),
             "config_update": {
                "learn_action": True,
                "learn_reward": False,
                "sgd_momentum": 0.75,
                "explore": False,
                "timesteps_per_iteration": None,  # To fill
                # === Optimization ===
                # Learning rate for adam optimizer
                "lr": None,  # To fill
                # Learning rate schedule
                "lr_schedule": [(0, None),
                                (None, 1e-12)],  # To fill
                "loss_fn": CrossEntropyLoss(
                    weight=None,
                    size_average=None,
                    ignore_index=-100,
                    reduce=None,
                    reduction='mean')
            }},
        ],

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
            "type": exploration.SoftQSchedule,
            # Add constructor kwargs here (if any).
            "temperature_schedule": PiecewiseSchedule(
                endpoints=[
                    (0, 1.0), (0, 0.1)],
                outside_value=0.1,
                framework="torch")
        },

        "log_level": "DEBUG",
    }
)

def prepare_default_config(lr, lr_spl, n_steps_per_epi, n_epi):
    default_config = copy.deepcopy(DEFAULT_CONFIG)
    default_config["exploration_config"]["temperature_schedule"] = \
        PiecewiseSchedule(
                endpoints=[(0, 1.0), (int(n_steps_per_epi*n_epi*0.75), 0.1)],
                outside_value=0.1,
                framework="torch")
    default_config.update({
        "lr": lr,
        "lr_schedule": [(0, lr),
                        (int(n_steps_per_epi * n_epi), lr / 1e9)],
    })
    default_config["nested_policies"][3]["config_update"].update({
        # === Optimization ===
        # Learning rate for adam optimizer
        "lr": lr_spl,
        # Learning rate schedule
        "lr_schedule": [(0, lr_spl),
                        (int(n_steps_per_epi * n_epi), lr_spl / 1e9)],

        "timesteps_per_iteration": n_steps_per_epi,
    })

    return default_config


def validate_config(config: TrainerConfigDict) -> None:
    validate_config_dqn(config)


def execution_plan(workers: WorkerSet,
                   config: TrainerConfigDict) -> LocalIterator[dict]:
    """
    Modified from the execution plan of the DQNTrainer
    """
    if config.get("prioritized_replay"):
        prio_args = {
            "prioritized_replay_alpha": config["prioritized_replay_alpha"],
            "prioritized_replay_beta": config["prioritized_replay_beta"],
            "prioritized_replay_eps": config["prioritized_replay_eps"],
        }
    else:
        prio_args = {}

    local_replay_buffer = LocalReplayBuffer(
        num_shards=1,
        learning_starts=config["learning_starts"],
        buffer_size=config["buffer_size"],
        replay_batch_size=config["train_batch_size"],
        replay_mode=config["multiagent"]["replay_mode"],
        replay_sequence_length=config["replay_sequence_length"],
        **prio_args)

    rollouts = ParallelRollouts(workers, mode="bulk_sync")

    # Create one batch to be added to the replay buffer
    # And one batch to be used to train the supervised learning policy in LTFT
    rollouts_dqn, rollouts_pg = rollouts.duplicate(n=2)

    # We execute the following steps concurrently:
    # (1) Generate rollouts and store them in our local replay buffer. Calling
    # next() on store_op drives this.
    store_op = rollouts_dqn.for_each(
        StoreToReplayBuffer(local_buffer=local_replay_buffer))

    def update_prio(item):
        samples, info_dict = item
        if config.get("prioritized_replay"):
            prio_dict = {}
            for policy_id, info in info_dict.items():
                # TODO(sven): This is currently structured differently for
                #  torch/tf. Clean up these results/info dicts across
                #  policies (note: fixing this in torch_policy.py will
                #  break e.g. DDPPO!).
                td_error = info.get("td_error",
                                    info[LEARNER_STATS_KEY].get("td_error"))
                prio_dict[policy_id] = (samples.policy_batches[policy_id]
                                        .data.get("batch_indexes"), td_error)
            local_replay_buffer.update_priorities(prio_dict)
        return info_dict

    # (2) Read and train on experiences from the replay buffer. Every batch
    # returned from the LocalReplay() iterator is passed to TrainOneStep to
    # take a SGD step, and then we decide whether to update the target network.
    post_fn = config.get("before_learn_on_batch") or (lambda b, *a: b)
    replay_op_dqn = Replay(local_buffer=local_replay_buffer) \
        .for_each(lambda x: post_fn(x, workers, config)) \
        .for_each(LocalTrainablePolicyModifier(workers, train_dqn_only)) \
        .for_each(TrainOneStep(workers)) \
        .for_each(update_prio) \
        .for_each(UpdateTargetNetwork(
        workers, config["target_network_update_freq"]))

    # Inform LTFT that we want to train the supervised learning policy only
    # Only train the LTFT policies with this batch
    ltft_policies_ids = _get_ltft_policies_ids(workers)
    train_op_pg = rollouts_pg \
        .combine(ConcatBatches(min_batch_size=config["train_batch_size"])) \
        .for_each(LocalTrainablePolicyModifier(workers, train_pg_only)) \
        .for_each(TrainOneStep(workers, policies=ltft_policies_ids))
    round_robin_weights = calculate_rr_weights(config)
    round_robin_weights.append(round_robin_weights[-1])

    # Alternate deterministically between (1) and (2). Only return the output
    # of (2) since training metrics are not available until (2) runs.
    train_op = Concurrently(
        [store_op, replay_op_dqn, train_op_pg],
        mode="round_robin",
        output_indexes=[1, 2],
        round_robin_weights=round_robin_weights)

    return StandardMetricsReporting(train_op, workers, config)

def _get_ltft_policies_ids(workers):
    ltft_policies_only = []
    for policy_id, policy in workers.local_worker().policy_map.items():
        if isinstance(policy, LTFTTorchPolicy):
            ltft_policies_only.append(policy)
    return ltft_policies_only

class LocalTrainablePolicyModifier:

    def __init__(self,
                 workers: WorkerSet,
                 fn_to_apply,
                 policies: List[PolicyID] = frozenset([])):
        self.workers = workers
        self.policies = policies or workers.local_worker().policies_to_train
        self.fn_to_apply = fn_to_apply

    def __call__(self, batch: SampleBatchType) -> SampleBatchType:
        w = self.workers.local_worker()
        w.foreach_trainable_policy(self.fn_to_apply)
        return batch

def train_pg_only(policy, pid, **kwargs):
    if isinstance(policy, LTFTTorchPolicy):
        policy.train_pg = True
        policy.train_dqn = False
    return policy

def train_dqn_only(policy, pid, **kwargs):
    if isinstance(policy, LTFTTorchPolicy):
        policy.train_pg = False
        policy.train_dqn = True
    return policy


LTFTTrainer = DQNTrainer.with_updates(
    name="LTFT",
    default_policy=LTFTTorchPolicy,
    default_config=DEFAULT_CONFIG,
    get_policy_class=None,
    validate_config=validate_config,
    execution_plan=execution_plan)
