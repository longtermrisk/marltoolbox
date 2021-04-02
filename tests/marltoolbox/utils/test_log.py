import numpy as np
from marltoolbox.utils.log import _add_entropy_to_log

def test__add_entropy_to_log():
    to_log = {}
    train_batch = {"action_dist_inputs": np.array([[0.0, 1.0]])}
    to_log = _add_entropy_to_log(train_batch, to_log)
    assert_close(to_log[f"entropy_buffer_samples_avg"], 0.00, 0.001)
    assert_close(to_log[f"entropy_buffer_samples_single"], 0.00, 0.001)

    to_log = {}
    train_batch = {"action_dist_inputs": np.array([[0.75, 0.25]])}
    to_log = _add_entropy_to_log(train_batch, to_log)
    assert_close(to_log[f"entropy_buffer_samples_avg"], 0.562335145, 0.001)
    assert_close(to_log[f"entropy_buffer_samples_single"], 0.562335145, 0.001)

    to_log = {}
    train_batch = {"action_dist_inputs": np.array([[0.62, 0.12, 0.13, 0.13]])}
    to_log = _add_entropy_to_log(train_batch, to_log)
    assert_close(to_log[f"entropy_buffer_samples_avg"], 1.081271236, 0.001)
    assert_close(to_log[f"entropy_buffer_samples_single"], 1.081271236, 0.001)

    to_log = {}
    train_batch = {"action_dist_inputs": np.array([[0.62, 0.12, 0.13, 0.13],
                                                   [0.75, 0.25, 0.0, 0.0],
                                                   [0.0, 1.0, 0.0, 0.0]])}
    to_log = _add_entropy_to_log(train_batch, to_log)
    assert_close(to_log[f"entropy_buffer_samples_avg"], 0.547868794, 0.001)
    assert_close(to_log[f"entropy_buffer_samples_single"], 0.00, 0.001)

    return to_log

def assert_close(a,b,threshold):
    abs_diff = np.abs(a - b)
    assert abs_diff < threshold


