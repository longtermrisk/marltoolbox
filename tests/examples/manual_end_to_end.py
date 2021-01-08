from ray.rllib.utils.test_utils import check_learning_achieved

def check_learning_achieved(tune_results, reward, max=False, min=False, equal=False):
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
    tune_analysis = main(debug=False)
    check_learning_achieved(tune_results=tune_analysis, reward=-75, max=True)

def test_ppo_asym_coin_game():
    from marltoolbox.examples.rllib_api.ppo_asymmetric_coin_game import main
    tune_analysis = main(debug=False, stop_iters=70)
    check_learning_achieved(tune_results=tune_analysis, reward=20, min=True)

#
# def test_le_ipd():
#     from marltoolbox.examples.rllib_api.le_ipd import main
#     main(debug=True)
#
#
# def test_amtft_various_env():
#     from marltoolbox.examples.rllib_api.amtft_various_env import main
#     main(debug=True)
#
#
# def test_inequity_aversion():
#     from marltoolbox.examples.rllib_api.inequity_aversion import main
#     main(debug=True)
#
#
# def test_l1br_amtft():
#     from marltoolbox.examples.rllib_api.l1br_amtft import main
#     main(debug=True)
#
#
# def test_lola_dice_tune_fn_api():
#     from marltoolbox.examples.tune_function_api.lola_dice_official import main
#     main(debug=True)
#
#
# def test_lola_pg_tune_fn_api():
#     from marltoolbox.examples.tune_function_api.lola_pg_official import main
#     main(debug=True)
#
#
# def test_lola_pg_tune_class_api():
#     from marltoolbox.examples.tune_class_api.lola_pg_official import main
#     main(debug=True)
#
# def test_lola_exact_tune_class_api():
#     from marltoolbox.examples.tune_class_api.lola_exact_official import main
#     main(debug=True)
#
# def test_lola_dice_tune_class_api():
#     from marltoolbox.examples.tune_class_api.lola_dice_official import main
#     main(debug=True)
#
# def test_l1br_lola_pg_tune_class_api():
#     from marltoolbox.examples.tune_class_api.l1br_lola_pg import main
#     main(debug=True)
#
# def test_adaptive_mechanism_design_tune_class_api():
#     from marltoolbox.examples.tune_class_api.adaptive_mechanism_design import main
#     main(debug=True)


if __name__ == "__main__":
    test_pg_ipd()