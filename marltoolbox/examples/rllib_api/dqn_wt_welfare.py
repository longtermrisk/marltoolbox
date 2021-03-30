from ray.rllib.agents.dqn.dqn_torch_policy import postprocess_nstep_and_prio

from marltoolbox.algos import augmented_dqn
from marltoolbox.examples.rllib_api import dqn_coin_game
from marltoolbox.utils import log, miscellaneous, postprocessing


def main(debug, welfare=postprocessing.WELFARE_UTILITARIAN):
    train_n_replicates = 1 if debug else 1
    seeds = miscellaneous.get_random_seeds(train_n_replicates)
    exp_name, _ = log.log_in_current_day_dir("DQN_welfare_CG")

    hparams = dqn_coin_game._get_hyperparameters(seeds, debug, exp_name)
    rllib_config, stop_config = dqn_coin_game._get_rllib_configs(hparams)
    rllib_config = _modify_policy_to_use_welfare(rllib_config, welfare)

    tune_analysis = dqn_coin_game._train_dqn_and_plot_logs(
        hparams, rllib_config, stop_config)

    return tune_analysis


def _modify_policy_to_use_welfare(rllib_config, welfare):
    MyCoopDQNTorchPolicy = augmented_dqn.MyDQNTorchPolicy.with_updates(
        postprocess_fn=miscellaneous.merge_policy_postprocessing_fn(
            postprocessing.welfares_postprocessing_fn(),
            postprocess_nstep_and_prio
        )
    )

    policies = rllib_config["multiagent"]["policies"]
    new_policies = {}
    for policies_id, policy_tuple in policies.items():
        new_policies[policies_id] = list(policy_tuple)
        new_policies[policies_id][0] = MyCoopDQNTorchPolicy
        if welfare == postprocessing.WELFARE_UTILITARIAN:
            new_policies[policies_id][3].update(
                {postprocessing.ADD_UTILITARIAN_WELFARE: True}
            )
        elif welfare == postprocessing.WELFARE_INEQUITY_AVERSION:
            add_ia_w = True
            ia_alpha = 0.0
            ia_beta = 0.5
            ia_gamma = 0.96
            ia_lambda = 0.96
            inequity_aversion_parameters = (
                add_ia_w,
                ia_alpha,
                ia_beta,
                ia_gamma,
                ia_lambda,
            )
            new_policies[policies_id][3].update(
                {postprocessing.ADD_INEQUITY_AVERSION_WELFARE:
                     inequity_aversion_parameters}
            )
    rllib_config["multiagent"]["policies"] = new_policies

    return rllib_config


if __name__ == "__main__":
    debug_mode = True
    main(debug_mode)
