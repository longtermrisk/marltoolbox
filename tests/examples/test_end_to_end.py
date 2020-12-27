def test_pg_ipd():
    from marltoolbox.examples.rllib_api.pg_ipd import main
    main(stop_iters=10, tf=False, debug=True)


def test_ppo_asym_coin_game():
    from marltoolbox.examples.rllib_api.ppo_asymmetric_coin_game import main
    main(stop_iters=10, tf=False, debug=True)


def test_le_ipd():
    from marltoolbox.examples.rllib_api.le_ipd import main
    main(debug=True)


def test_amtft_various_env():
    from marltoolbox.examples.rllib_api.amtft_various_env import main
    main(debug=True)
