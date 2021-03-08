import torch
from marltoolbox.utils.exploration import clusterize_by_distance

def assert_equal_wt_some_epsilon(v1, v2, epsilon = 1e-6):
    delta = torch.abs(v1 - v2)
    assert torch.all(delta < epsilon)


def test_clusterize_by_distance():
    output = clusterize_by_distance(
        torch.Tensor([0.0, 0.4, 1.0, 1.4, 1.8, 3.0]), 0.5)
    assert_equal_wt_some_epsilon(
        output,
        torch.Tensor([0.2000, 0.2000, 1.4000, 1.4000, 1.4000,3.0000]))

    output = clusterize_by_distance(
        torch.Tensor([0.0, 0.5, 1.0, 1.4, 1.8, 3.0]), 0.5)
    assert_equal_wt_some_epsilon(
        output,
        torch.Tensor([0.0000, 0.5000, 1.4000, 1.4000, 1.4000, 3.0000]))

    output = clusterize_by_distance(
        torch.Tensor([-10.0, -9.8, 1.0, 1.4, 1.8, 3.0]), 0.5)
    assert_equal_wt_some_epsilon(
        output,
        torch.Tensor([-9.9000, -9.9000, 1.4000, 1.4000, 1.4000, 3.0000]))

    output = clusterize_by_distance(
        torch.Tensor([-1.0, -0.51, -0.1, 0.0, 0.1, 0.51, 1.0]), 0.5)
    assert_equal_wt_some_epsilon(
        output,
        torch.Tensor([0., 0., 0., 0., 0., 0., 0.]))


