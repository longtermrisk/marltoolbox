# Run this directly with
# pytest file_path.py

import ray

from marltoolbox.utils.miscellaneous import check_learning_achieved


def test_pg_ipd():
    from marltoolbox.examples.rllib_api.pg_ipd import main
    # Restart Ray defensively in case the ray connection is lost.
    ray.shutdown()
    tune_analysis = main(debug=False)
    check_learning_achieved(tune_results=tune_analysis, max_=-75)


def test_ppo_asym_coin_game():
    from marltoolbox.examples.rllib_api.ppo_asymmetric_coin_game import main
    ray.shutdown()
    tune_analysis = main(debug=False, stop_iters=70)
    check_learning_achieved(tune_results=tune_analysis, min_=20)


def test_ltft_ipd():
    from marltoolbox.examples.rllib_api.ltft import main
    ray.shutdown()
    tune_analysis_self_play, tune_analysis_naive_opponent = main(debug=False)
    check_learning_achieved(tune_results=tune_analysis_self_play, min_=-42)
    check_learning_achieved(tune_results=tune_analysis_naive_opponent, max_=-78)


def test_amtft_ipd():
    from marltoolbox.examples.rllib_api.amtft_various_env import main
    ray.shutdown()
    tune_analysis_per_welfare, analysis_metrics_per_mode = main(
        debug=False, train_n_replicates=1, filter_utilitarian=False,
        env="IteratedPrisonersDilemma")
    for welfare_name, tune_analysis in tune_analysis_per_welfare.items():
        check_learning_achieved(tune_results=tune_analysis, min_=-204)
