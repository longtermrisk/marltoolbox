# Run this directly with
# pytest file_path.py

import ray
from ray.rllib.utils.test_utils import check_learning_achieved


def check_learning_achieved(tune_results, reward,
                            max=False, min=False, equal=False):
    if min:
        if tune_results.trials[0].last_result["episode_reward_mean"] < reward:
            raise ValueError("`failed on min-reward` of {}!".format(reward))
    if max:
        if tune_results.trials[0].last_result["episode_reward_mean"] > max:
            raise ValueError("`failed on max-reward` of {}!".format(reward))
    if equal:
        if tune_results.trials[0].last_result["episode_reward_mean"] == max:
            raise ValueError("`failed on max-reward` of {}!".format(reward))


def test_pg_ipd():
    from marltoolbox.examples.rllib_api.pg_ipd import main
    # Restart Ray defensively in case the ray connection is lost.
    ray.shutdown()
    tune_analysis = main(debug=False)
    check_learning_achieved(tune_results=tune_analysis, reward=-75, max=True)


def test_ppo_asym_coin_game():
    from marltoolbox.examples.rllib_api.ppo_asymmetric_coin_game import main
    ray.shutdown()
    tune_analysis = main(debug=False, stop_iters=70)
    check_learning_achieved(tune_results=tune_analysis, reward=20, min=True)


def test_ltft_ipd():
    from marltoolbox.examples.rllib_api.ltft import main
    ray.shutdown()
    tune_analysis_self_play, tune_analysis_naive_opponent = main(debug=False)
    check_learning_achieved(
        tune_results=tune_analysis_self_play, reward=-42, min=True)
    check_learning_achieved(
        tune_results=tune_analysis_naive_opponent, reward=-78, max=True)


def test_amtft_ipd():
    from marltoolbox.examples.rllib_api.amtft_various_env import main
    ray.shutdown()
    tune_analysis_per_welfare, analysis_metrics_per_mode = main(
        debug=False, train_n_replicates=1, filter_utilitarian=False,
        env="IteratedPrisonersDilemma")
    for welfare_name, tune_analysis in tune_analysis_per_welfare.items():
        check_learning_achieved(
            tune_results=tune_analysis, reward=-204, min=True)
