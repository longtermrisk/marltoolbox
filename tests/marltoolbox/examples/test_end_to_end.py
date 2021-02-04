def test_pg_ipd():
    from marltoolbox.examples.rllib_api.pg_ipd import main
    main(stop_iters=10, tf=False, debug=True)


def test_ppo_asym_coin_game():
    from marltoolbox.examples.rllib_api.ppo_asymmetric_coin_game import main
    main(stop_iters=3, tf=False, debug=True)


def test_le_ipd():
    from marltoolbox.examples.rllib_api.ltft_ipd import main
    main(debug=True)


def test_amtft_various_env():
    from marltoolbox.examples.rllib_api.amtft_various_env import main
    main(debug=True)


def test_inequity_aversion():
    from marltoolbox.examples.rllib_api.inequity_aversion import main
    main(debug=True)


def test_l1br_amtft():
    from marltoolbox.examples.rllib_api.l1br_amtft import main
    main(debug=True)


def test_lola_dice_tune_fn_api():
    from marltoolbox.examples.tune_function_api.lola_dice_official import main
    main(debug=True)


def test_lola_pg_tune_fn_api():
    from marltoolbox.examples.tune_function_api.lola_pg_official import main
    main(debug=True)


def test_lola_pg_tune_class_api():
    from marltoolbox.examples.tune_class_api.lola_pg_official import main
    main(debug=True)

def test_lola_exact_tune_class_api():
    from marltoolbox.examples.tune_class_api.lola_exact_official import main
    main(debug=True)

def test_lola_dice_tune_class_api():
    from marltoolbox.examples.tune_class_api.lola_dice_official import main
    main(debug=True)

def test_l1br_lola_pg_tune_class_api():
    from marltoolbox.examples.tune_class_api.l1br_lola_pg import main
    main(debug=True)

def test_adaptive_mechanism_design_tune_class_api():
    from marltoolbox.examples.tune_class_api.amd import main
    main(debug=True, use_rllib_policy=False)

def test_adaptive_mechanism_design_tune_class_api_wt_rllib_policy():
    from marltoolbox.examples.tune_class_api.amd import main
    main(debug=True, use_rllib_policy=True)