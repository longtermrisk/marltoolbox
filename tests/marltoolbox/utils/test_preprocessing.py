from ray.rllib.policy.sample_batch import SampleBatch

from marltoolbox.utils import postprocessing


def generate_batch(own_rew, opp_rew):
    own_fake_data = {SampleBatch.REWARDS: own_rew}
    opp_fake_data = {SampleBatch.REWARDS: opp_rew}
    sample_batch = SampleBatch(**own_fake_data)
    opp_ag_batch = SampleBatch(**opp_fake_data)
    return sample_batch, opp_ag_batch


def test_add_inequity_aversion_welfare_to_batch_beta():
    gamma = 0.0
    lambda_ = 0.0
    # Disvalue lower than opp
    alpha = 0.0
    # Disvalue higher than opp
    beta = 1.0

    # postprocessing_fn = postprocessing.get_postprocessing_welfare_function(
    #     add_inequity_aversion_welfare=True,
    #     inequity_aversion_gamma=gamma,
    #     inequity_aversion_beta=beta,
    #     inequity_aversion_lambda=lambda_,
    #     inequity_aversion_alpha=alpha,
    # )

    sample_batch, opp_ag_batch = generate_batch(own_rew=[0, 1, 0, 10, 1, 0, -2, -888, -888],
                                                opp_rew=[0, 0, 1, 11, 0.5, -1, -4, -1888, 1888])
    postprocessing._add_inequity_aversion_welfare_to_batch(sample_batch, opp_ag_batch, alpha, beta, gamma, lambda_)
    assert (sample_batch[postprocessing.WELFARE_INEQUITY_AVERSION]
            == [0, 0, 0, 10, 0.5, -1, -4, -1888, -888]).all()


def test_add_inequity_aversion_welfare_to_batch_alpha():
    gamma = 0.0
    lambda_ = 0.0
    # Disvalue lower than opp
    alpha = 0.5
    # Disvalue higher than opp
    beta = 0.0

    # postprocessing_fn = postprocessing.get_postprocessing_welfare_function(
    #     add_inequity_aversion_welfare=True,
    #     inequity_aversion_gamma=gamma,
    #     inequity_aversion_beta=beta,
    #     inequity_aversion_lambda=lambda_,
    #     inequity_aversion_alpha=alpha,
    # )

    sample_batch, opp_ag_batch = generate_batch(own_rew=[0, 1, 0, 10, 1, 0, -2, -888, -500],
                                                opp_rew=[0, 0, 1, 11, 0.5, -1, -4, -1888, 1500])
    postprocessing._add_inequity_aversion_welfare_to_batch(sample_batch, opp_ag_batch, alpha, beta, gamma, lambda_)
    assert (sample_batch[postprocessing.WELFARE_INEQUITY_AVERSION]
            == [0, 1, -.5, 9.5, 1, 0, -2, -888, -1500]).all()


def test_add_inequity_aversion_welfare_to_batch_lambda():
    gamma = 1.0
    lambda_ = 0.9
    # Disvalue lower than opp
    alpha = 0.0
    # Disvalue higher than opp
    beta = 0.5

    # postprocessing_fn = postprocessing.get_postprocessing_welfare_function(
    #     add_inequity_aversion_welfare=True,
    #     inequity_aversion_gamma=gamma,
    #     inequity_aversion_beta=beta,
    #     inequity_aversion_lambda=lambda_,
    #     inequity_aversion_alpha=alpha,
    # )

    sample_batch, opp_ag_batch = generate_batch(
        #  smoothed cumulative rewards [0, 1, 0.9, 1.81]
        own_rew=[0, 1, 0, 1],
        #  smoothed cumulative rewards [0, 0, 0.5, 0.45]
        opp_rew=[0, 0, 0.5, 0])
    postprocessing._add_inequity_aversion_welfare_to_batch(sample_batch, opp_ag_batch, alpha, beta, gamma, lambda_)
    assert (sample_batch[postprocessing.WELFARE_INEQUITY_AVERSION]
            == [0.0, 0.5, -0.2, 1 - 0.68]).all()


def test_add_utilitarian_welfare_to_batch():
    # postprocessing_fn = postprocessing.get_postprocessing_welfare_function(
    #     add_utilitarian_welfare=True,
    # )

    sample_batch, opp_ag_batch = generate_batch(own_rew=[0, 1, 0, 10, 1, 0, -2, -888, -888],
                                                opp_rew=[0, 0, 1, 11, 0.5, -1, -4, -1888, 1888])
    postprocessing._add_utilitarian_welfare_to_batch(sample_batch, [opp_ag_batch])
    assert (sample_batch[postprocessing.WELFARE_UTILITARIAN]
            == [0, 1, 1, 21, 1.5, -1, -6, -2776, 1000]).all()
