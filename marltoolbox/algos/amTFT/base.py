import logging

from ray.rllib.agents.dqn.dqn_torch_policy import postprocess_nstep_and_prio
from ray.rllib.utils import merge_dicts

from marltoolbox.algos import hierarchical, augmented_dqn
from marltoolbox.utils import postprocessing, miscellaneous

logger = logging.getLogger(__name__)

APPROXIMATION_METHOD_Q_VALUE = "amTFT_use_Q_net"
APPROXIMATION_METHOD_ROLLOUTS = "amTFT_use_rollout"
APPROXIMATION_METHODS = (
    APPROXIMATION_METHOD_Q_VALUE,
    APPROXIMATION_METHOD_ROLLOUTS,
)
WORKING_STATES = (
    "train_coop",
    "train_selfish",
    "eval_amtft",
    "eval_naive_selfish",
    "eval_naive_coop",
)
WORKING_STATES_IN_EVALUATION = WORKING_STATES[2:]

OWN_COOP_POLICY_IDX = 0
OWN_SELFISH_POLICY_IDX = 1
OPP_COOP_POLICY_IDX = 2
OPP_SELFISH_POLICY_IDX = 3

DEFAULT_NESTED_POLICY_SELFISH = augmented_dqn.MyDQNTorchPolicy
DEFAULT_NESTED_POLICY_COOP = DEFAULT_NESTED_POLICY_SELFISH.with_updates(
    postprocess_fn=miscellaneous.merge_policy_postprocessing_fn(
        postprocessing.welfares_postprocessing_fn(
            add_utilitarian_welfare=True,
        ),
        postprocess_nstep_and_prio,
    )
)

DEFAULT_CONFIG = merge_dicts(
    hierarchical.DEFAULT_CONFIG,
    {
        # One of WORKING_STATES.
        "working_state": WORKING_STATES[0],
        "debit_threshold": 2.0,
        "punishment_multiplier": 6.0,
        "rollout_length": 40,
        "n_rollout_replicas": 20,
        "last_k": 1,
        # TODO use log level of RLLib instead of mine
        "verbose": 1,
        "auto_load_checkpoint": True,
        # To configure
        "own_policy_id": None,
        "opp_policy_id": None,
        "callbacks": None,
        # One from marltoolbox.utils.postprocessing.WELFARES
        "welfare_key": postprocessing.WELFARE_UTILITARIAN,
        "nested_policies": [
            # Here the trainer need to be a DQNTrainer
            # to provide the config for the 3 DQNTorchPolicy
            {"Policy_class": DEFAULT_NESTED_POLICY_COOP, "config_update": {}},
            {
                "Policy_class": DEFAULT_NESTED_POLICY_SELFISH,
                "config_update": {},
            },
            {"Policy_class": DEFAULT_NESTED_POLICY_COOP, "config_update": {}},
            {
                "Policy_class": DEFAULT_NESTED_POLICY_SELFISH,
                "config_update": {},
            },
        ],
        "optimizer": {
            "sgd_momentum": 0.0,
        },
        "batch_mode": "complete_episodes",
    },
)

PLOT_KEYS = [
    "punish",
    "debit",
    "debit_threshold",
    "summed_debit",
    "summed_n_steps_to_punish",
]

PLOT_ASSEMBLAGE_TAGS = [
    ("punish",),
    ("debit",),
    ("debit_threshold",),
    ("summed_debit",),
    ("summed_n_steps_to_punish",),
]


class AmTFTReferenceClass:
    pass
