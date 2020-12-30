import os
import ray
from ray import tune
from ray.rllib.agents.pg import PGTrainer, PGTorchPolicy, pg_torch_policy

# from marltoolbox.algos.inequity_aversion import InequityAversionTrainer
from marltoolbox.envs.matrix_SSD import IteratedBoSAndPD
from marltoolbox.utils import miscellaneous, log, postprocessing


def main(debug):
    ray.init(num_cpus=os.cpu_count(), num_gpus=0)

    stop = {"episodes_total": 10 if debug else 400}

    env_config = {
        "max_steps": 10,
        "players_ids": ["player_row", "player_col"],
    }

    policies = {
        env_config["players_ids"][0]: (None, IteratedBoSAndPD.OBSERVATION_SPACE, IteratedBoSAndPD.ACTION_SPACE, {}),
        env_config["players_ids"][1]: (None, IteratedBoSAndPD.OBSERVATION_SPACE, IteratedBoSAndPD.ACTION_SPACE, {})}

    rllib_config = {
        "env": IteratedBoSAndPD,
        "env_config": env_config,

        "num_gpus": 0,
        "num_workers": 1,

        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": (lambda agent_id: agent_id),
        },
        "framework": "torch",
        "gamma": 0.5,

        "callbacks": miscellaneous.merge_callbacks(
            log.get_logging_callbacks_class(), postprocessing.OverwriteRewardWtWelfareCallback),

    }

    MyPGTorchPolicy = PGTorchPolicy.with_updates(
        postprocess_fn=miscellaneous.merge_policy_postprocessing_fn(
            postprocessing.get_postprocessing_welfare_function(
                add_inequity_aversion_welfare=True,
                inequity_aversion_beta=1.0,
                inequity_aversion_alpha=0.0,
                inequity_aversion_gamma=1.0,
                inequity_aversion_lambda=0.5
            ),
            pg_torch_policy.post_process_advantages
        )
    )
    MyPGTrainer = PGTrainer.with_updates(default_policy=MyPGTorchPolicy, get_policy_class=None)
    tune.run(MyPGTrainer, stop=stop, checkpoint_freq=10, config=rllib_config)
    ray.shutdown()


if __name__ == "__main__":
    debug_mode = False
    main(debug_mode)
