# Run this directly with
# pytest file_path.py

import ray

from marltoolbox.utils import postprocessing
from marltoolbox.utils.miscellaneous import check_learning_achieved


def print_metrics_available(tune_analysis):
    print(
        "metric available in tune_analysis:",
        tune_analysis.results_df.columns.tolist(),
    )


def test_pg_ipd():
    from marltoolbox.examples.rllib_api.pg_ipd import main

    # Restart Ray defensively in case the ray connection is lost.
    ray.shutdown()
    tune_analysis = main(debug=False)
    print_metrics_available(tune_analysis)
    check_learning_achieved(tune_results=tune_analysis, max_=-75)
    check_learning_achieved(
        tune_results=tune_analysis,
        min_=0.9,
        metric="custom_metrics.DD_freq/player_row_mean",
    )
    check_learning_achieved(
        tune_results=tune_analysis,
        min_=0.9,
        metric="custom_metrics.DD_freq/player_col_mean",
    )


def test_r2d2_ipd():
    from marltoolbox.examples.rllib_api.pg_ipd import main

    # Restart Ray defensively in case the ray connection is lost.
    ray.shutdown()
    tune_analysis = main(debug=False)
    print_metrics_available(tune_analysis)
    check_learning_achieved(tune_results=tune_analysis, max_=-75)
    check_learning_achieved(
        tune_results=tune_analysis,
        min_=0.9,
        metric="custom_metrics.DD_freq/player_row_mean",
    )
    check_learning_achieved(
        tune_results=tune_analysis,
        min_=0.9,
        metric="custom_metrics.DD_freq/player_col_mean",
    )


def test_ltft_ipd():
    from marltoolbox.experiments.rllib_api.ltft_various_env import main

    ray.shutdown()
    tune_analysis_self_play, tune_analysis_against_opponent = main(
        debug=False,
        env="IteratedPrisonersDilemma",
        train_n_replicates=1,
        against_naive_opp=True,
    )
    print_metrics_available(tune_analysis_self_play)
    check_learning_achieved(tune_results=tune_analysis_self_play, min_=-42)
    check_learning_achieved(
        tune_results=tune_analysis_self_play,
        min_=0.9,
        metric="custom_metrics.CC_freq/player_row_mean",
    )
    check_learning_achieved(
        tune_results=tune_analysis_self_play,
        min_=0.9,
        metric="custom_metrics.CC_freq/player_col_mean",
    )
    print_metrics_available(tune_analysis_against_opponent)
    check_learning_achieved(
        tune_results=tune_analysis_against_opponent, max_=-75
    )
    check_learning_achieved(
        tune_results=tune_analysis_against_opponent,
        min_=0.9,
        metric="custom_metrics.DD_freq/player_row_mean",
    )
    check_learning_achieved(
        tune_results=tune_analysis_against_opponent,
        min_=0.9,
        metric="custom_metrics.DD_freq/player_col_mean",
    )


def test_amtft_ipd():
    from marltoolbox.experiments.rllib_api.amtft_various_env import main

    ray.shutdown()
    tune_analysis_per_welfare, analysis_metrics_per_mode = main(
        debug=False,
        train_n_replicates=1,
        filter_utilitarian=False,
        env="IteratedPrisonersDilemma",
    )
    for welfare_name, tune_analysis in tune_analysis_per_welfare.items():
        print("welfare_name", welfare_name)
        print_metrics_available(tune_analysis)
        check_learning_achieved(tune_results=tune_analysis, min_=-204)
        check_learning_achieved(
            tune_results=tune_analysis,
            min_=0.9,
            metric="custom_metrics.CC_freq/player_row_mean",
        )
        check_learning_achieved(
            tune_results=tune_analysis,
            min_=0.9,
            metric="custom_metrics.CC_freq/player_col_mean",
        )


def test_ppo_asym_coin_game():
    from marltoolbox.examples.rllib_api.ppo_coin_game import main

    ray.shutdown()
    tune_analysis = main(debug=False, stop_iters=200)
    print_metrics_available(tune_analysis)
    check_learning_achieved(tune_results=tune_analysis, min_=15)
    check_learning_achieved(
        tune_results=tune_analysis,
        min_=0.30,
        metric="custom_metrics.pick_speed/player_red_mean",
    )
    check_learning_achieved(
        tune_results=tune_analysis,
        min_=0.30,
        metric="custom_metrics.pick_speed/player_blue_mean",
    )
    check_learning_achieved(
        tune_results=tune_analysis,
        min_=0.40,
        max_=0.60,
        metric="custom_metrics.pick_own_color/player_red_mean",
    )
    check_learning_achieved(
        tune_results=tune_analysis,
        min_=0.40,
        max_=0.60,
        metric="custom_metrics.pick_own_color/player_blue_mean",
    )


def test_dqn_coin_game():
    from marltoolbox.examples.rllib_api.dqn_coin_game import main

    ray.shutdown()
    tune_analysis = main(debug=False)
    print_metrics_available(tune_analysis)
    check_learning_achieved(tune_results=tune_analysis, max_=20)
    check_learning_achieved(
        tune_results=tune_analysis,
        min_=0.5,
        metric="custom_metrics.pick_speed/player_red_mean",
    )
    check_learning_achieved(
        tune_results=tune_analysis,
        min_=0.5,
        metric="custom_metrics.pick_speed/player_blue_mean",
    )
    check_learning_achieved(
        tune_results=tune_analysis,
        min_=0.40,
        max_=0.6,
        metric="custom_metrics.pick_own_color/player_red_mean",
    )
    check_learning_achieved(
        tune_results=tune_analysis,
        min_=0.40,
        max_=0.6,
        metric="custom_metrics.pick_own_color/player_blue_mean",
    )


def test_dqn_wt_utilitarian_welfare_coin_game():
    from marltoolbox.examples.rllib_api.dqn_wt_welfare import main

    ray.shutdown()
    tune_analysis = main(debug=False)
    print_metrics_available(tune_analysis)
    check_learning_achieved(tune_results=tune_analysis, min_=50)
    check_learning_achieved(
        tune_results=tune_analysis,
        min_=0.3,
        metric="custom_metrics.pick_speed/player_red_mean",
    )
    check_learning_achieved(
        tune_results=tune_analysis,
        min_=0.3,
        metric="custom_metrics.pick_speed/player_blue_mean",
    )
    check_learning_achieved(
        tune_results=tune_analysis,
        min_=0.95,
        metric="custom_metrics.pick_own_color/player_red_mean",
    )
    check_learning_achieved(
        tune_results=tune_analysis,
        min_=0.95,
        metric="custom_metrics.pick_own_color/player_blue_mean",
    )


def test_dqn_wt_inequity_aversion_welfare_coin_game():
    from marltoolbox.examples.rllib_api.dqn_wt_welfare import main

    ray.shutdown()
    tune_analysis = main(
        debug=False, welfare=postprocessing.WELFARE_INEQUITY_AVERSION
    )
    print_metrics_available(tune_analysis)
    check_learning_achieved(tune_results=tune_analysis, min_=50)
    check_learning_achieved(
        tune_results=tune_analysis,
        min_=0.25,
        metric="custom_metrics.pick_speed/player_red_mean",
    )
    check_learning_achieved(
        tune_results=tune_analysis,
        min_=0.25,
        metric="custom_metrics.pick_speed/player_blue_mean",
    )
    check_learning_achieved(
        tune_results=tune_analysis,
        min_=0.9,
        metric="custom_metrics.pick_own_color/player_red_mean",
    )
    check_learning_achieved(
        tune_results=tune_analysis,
        min_=0.9,
        metric="custom_metrics.pick_own_color/player_blue_mean",
    )


def test_ltft_coin_game():
    from marltoolbox.experiments.rllib_api.ltft_various_env import main

    ray.shutdown()
    tune_analysis_self_play, tune_analysis_against_opponent = main(
        debug=False,
        env="CoinGame",
        train_n_replicates=1,
        against_naive_opp=True,
    )
    print_metrics_available(tune_analysis_self_play)
    check_learning_achieved(tune_results=tune_analysis_self_play, min_=50)
    check_learning_achieved(
        tune_results=tune_analysis_self_play,
        min_=0.3,
        metric="custom_metrics.pick_speed/player_red_mean",
    )
    check_learning_achieved(
        tune_results=tune_analysis_self_play,
        min_=0.3,
        metric="custom_metrics.pick_speed/player_blue_mean",
    )
    check_learning_achieved(
        tune_results=tune_analysis_self_play,
        min_=0.9,
        metric="custom_metrics.pick_own_color/player_red_mean",
    )
    check_learning_achieved(
        tune_results=tune_analysis_self_play,
        min_=0.9,
        metric="custom_metrics.pick_own_color/player_blue_mean",
    )
    print_metrics_available(tune_analysis_against_opponent)
    check_learning_achieved(
        tune_results=tune_analysis_against_opponent, max_=20
    )
    check_learning_achieved(
        tune_results=tune_analysis_against_opponent,
        min_=0.3,
        metric="custom_metrics.pick_speed/player_red_mean",
    )
    check_learning_achieved(
        tune_results=tune_analysis_against_opponent,
        min_=0.3,
        metric="custom_metrics.pick_speed/player_blue_mean",
    )
    check_learning_achieved(
        tune_results=tune_analysis_against_opponent,
        max_=0.6,
        metric="custom_metrics.pick_own_color/player_red_mean",
    )
    check_learning_achieved(
        tune_results=tune_analysis_against_opponent,
        max_=0.6,
        metric="custom_metrics.pick_own_color/player_blue_mean",
    )


def test_amtft_coin_game():
    from marltoolbox.experiments.rllib_api.amtft_various_env import main

    ray.shutdown()
    tune_analysis_per_welfare, analysis_metrics_per_mode = main(
        debug=False,
        train_n_replicates=1,
        filter_utilitarian=False,
        env="CoinGame",
    )
    for welfare_name, tune_analysis in tune_analysis_per_welfare.items():
        print("welfare_name", welfare_name)
        print_metrics_available(tune_analysis)
        check_learning_achieved(tune_results=tune_analysis, min_=40)
        check_learning_achieved(
            tune_results=tune_analysis,
            min_=0.25,
            metric="custom_metrics.pick_speed/player_red_mean",
        )
