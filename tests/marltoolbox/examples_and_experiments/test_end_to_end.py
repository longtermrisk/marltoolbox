import ray


def test_pg_ipd():
    from examples.rllib_api.pg_ipd import main
    ray.shutdown()  # Restart Ray defensively in case the ray connection is lost.
    main(stop_iters=10, tf=False, debug=True)


def test_ppo_asym_coin_game():
    from examples.rllib_api.ppo_coin_game import main
    ray.shutdown()
    main(debug=True, stop_iters=3, tf=False)

def test_ppo_asym_coin_game():
    from examples.rllib_api.dqn_coin_game import main
    ray.shutdown()
    main(debug=True)

def test_ltft_ipd():
    from marltoolbox.experiments.rllib_api.ltft_various_env import main
    ray.shutdown()
    main(debug=True, env="IteratedPrisonersDilemma", train_n_replicates=1)


def test_ltft_coin_game():
    from marltoolbox.experiments.rllib_api.ltft_various_env import main
    ray.shutdown()
    main(debug=True, env="CoinGame", train_n_replicates=1)


def test_amtft_ipd():
    from marltoolbox.experiments.rllib_api.amtft_various_env import main
    ray.shutdown()
    main(debug=True, env="IteratedPrisonersDilemma")


def test_amtft_iasymbos():
    from marltoolbox.experiments.rllib_api.amtft_various_env import main
    ray.shutdown()
    main(debug=True, env="IteratedAsymBoS")


def test_amtft_coin_game():
    from marltoolbox.experiments.rllib_api.amtft_various_env import main
    ray.shutdown()
    main(debug=True, env="CoinGame")


def test_amtft_asym_coin_game():
    from marltoolbox.experiments.rllib_api.amtft_various_env import main
    ray.shutdown()
    main(debug=True, env="AsymCoinGame")


def test_amtft_mixed_motive_coin_game():
    from marltoolbox.experiments.rllib_api.amtft_various_env import main
    ray.shutdown()
    main(debug=True, env="MixedMotiveCoinGame")


def test_inequity_aversion():
    from examples.rllib_api.inequity_aversion import main
    ray.shutdown()
    main(debug=True)


def test_l1br_amtft_ipd():
    from marltoolbox.experiments.rllib_api.l1br_amtft import main
    ray.shutdown()
    main(debug=True, env="IteratedPrisonersDilemma")


def test_l1br_amtft_iasymbos():
    from marltoolbox.experiments.rllib_api.l1br_amtft import main
    ray.shutdown()
    main(debug=True, env="IteratedAsymBoS")


def test_l1br_amtft_coin_game():
    from marltoolbox.experiments.rllib_api.l1br_amtft import main
    ray.shutdown()
    main(debug=True, env="CoinGame")


def test_lola_dice_tune_fn_api():
    from marltoolbox.experiments.tune_function_api.lola_dice_official import main
    ray.shutdown()
    main(debug=True)


def test_lola_pg_tune_fn_api():
    from marltoolbox.experiments.tune_function_api.lola_pg_official import main
    ray.shutdown()
    main(debug=True)


def test_lola_pg_tune_class_api_ipd():
    from marltoolbox.experiments.tune_class_api.lola_pg_official import main
    ray.shutdown()
    main(debug=True, env="IteratedPrisonersDilemma")


def test_lola_pg_tune_class_api_ibos():
    from marltoolbox.experiments.tune_class_api.lola_pg_official import main
    ray.shutdown()
    main(debug=True, env="IteratedAsymBoS")


def test_lola_pg_tune_class_api_coin_game():
    from marltoolbox.experiments.tune_class_api.lola_pg_official import \
        main
    ray.shutdown()
    main(debug=True, env="VectorizedCoinGame")


def test_lola_pg_tune_class_api_asym_coin_game():
    from marltoolbox.experiments.tune_class_api.lola_pg_official import \
        main
    ray.shutdown()
    main(debug=True, env="AsymVectorizedCoinGame")


def test_lola_pg_tune_class_api_mixed_motive_coin_game():
    from marltoolbox.experiments.tune_class_api.lola_pg_official import \
        main
    ray.shutdown()
    main(debug=True, env="VectorizedMixedMotiveCoinGame")


def test_lola_exact_tune_class_api():
    from marltoolbox.experiments.tune_class_api.lola_exact_official import main
    ray.shutdown()
    main(debug=True)


def test_lola_dice_tune_class_api():
    from marltoolbox.experiments.tune_class_api.lola_dice_official import main
    ray.shutdown()
    main(debug=True)


def test_l1br_lola_pg_tune_class_api():
    from marltoolbox.experiments.tune_class_api.l1br_lola_pg import main
    ray.shutdown()
    main(debug=True)


def test_adaptive_mechanism_design_tune_class_api():
    from marltoolbox.experiments.tune_class_api.amd import main
    ray.shutdown()
    main(debug=True, use_rllib_policy=False)


def test_adaptive_mechanism_design_tune_class_api_wt_rllib_policy():
    from marltoolbox.experiments.tune_class_api.amd import main
    ray.shutdown()
    main(debug=True, use_rllib_policy=True)
