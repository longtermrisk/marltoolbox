"""
The magic corrections of LOLA.
"""
from functools import partial
import tensorflow as tf
from tensorflow.python.ops import math_ops

from marltoolbox.algos.lola.utils import flatgrad


def corrections_func(mainPN, batch_size, trace_length,
                     corrections=False, cube=None,
                     clip_lola_update_norm=False,
                     lola_correction_multiplier=1.0,
                     clip_lola_correction_norm=False,
                     clip_lola_actor_norm=False,
                     against_destabilizer_exploiter=False):
    """Computes corrections for policy gradients.

    Args:
    -----
        mainPN: list of policy/Q-networks
        batch_size: int
        trace_length: int
        corrections: bool (default: False)
            Whether policy networks should use corrections.
        cube: tf.Varialbe or None (default: None)
            If provided, should be constructed via `lola.utils.make_cube`.
            Used for variance reduction of the value estimation.
            When provided, the computation graph for corrections is faster to
            compile but is quite memory inefficient.
            When None, variance reduction graph is contructed dynamically,
            is a little longer to compile, but has lower memory footprint.
    """
    # not mem_efficient
    if cube is not None:
        ac_logp0 = tf.reshape(mainPN[0].log_pi_action_bs_t,
                              [batch_size, 1, trace_length])
        ac_logp1 = tf.reshape(mainPN[1].log_pi_action_bs_t,
                              [batch_size, trace_length, 1])
        mat_1 = tf.reshape(tf.squeeze(tf.matmul(ac_logp1, ac_logp0)),
                           [batch_size, 1, trace_length * trace_length])

        v_0 = tf.matmul(tf.reshape(mainPN[0].sample_reward, [batch_size, trace_length, 1]), mat_1)
        v_0 = tf.reshape(v_0, [batch_size, trace_length, trace_length, trace_length])

        v_1 = tf.matmul(tf.reshape(mainPN[1].sample_reward, [batch_size, trace_length, 1]), mat_1)
        v_1 = tf.reshape(v_1, [batch_size, trace_length, trace_length, trace_length])

        v_0 = 2 * tf.reduce_sum(v_0 * cube) / batch_size
        v_1 = 2 * tf.reduce_sum(v_1 * cube) / batch_size
    # wt mem_efficient
    else:
        ac_logp0 = tf.reshape(mainPN[0].log_pi_action_bs_t,
                              [batch_size, trace_length])
        ac_logp1 = tf.reshape(mainPN[1].log_pi_action_bs_t,
                              [batch_size, trace_length])

        # Static exclusive cumsum
        ac_logp0_cumsum = [tf.constant(0.)]
        ac_logp1_cumsum = [tf.constant(0.)]
        for i in range(trace_length - 1):
            ac_logp0_cumsum.append(tf.add(ac_logp0_cumsum[-1], ac_logp0[:, i]))
            ac_logp1_cumsum.append(tf.add(ac_logp1_cumsum[-1], ac_logp1[:, i]))

        # Compute v_0 and v_1
        mat_cumsum = ac_logp0[:, 0] * ac_logp1[:, 0]
        v_0 = mat_cumsum * mainPN[0].sample_reward[:, 0]
        v_1 = mat_cumsum * mainPN[1].sample_reward[:, 0]
        for i in range(1, trace_length):
            mat_cumsum = tf.add(mat_cumsum, ac_logp0[:, i] * ac_logp1[:, i])
            mat_cumsum = tf.add(mat_cumsum, ac_logp0_cumsum[i] * ac_logp1[:, i])
            mat_cumsum = tf.add(mat_cumsum, ac_logp1_cumsum[i] * ac_logp0[:, i])
            v_0 = tf.add(v_0, mat_cumsum * mainPN[0].sample_reward[:, i])
            v_1 = tf.add(v_1, mat_cumsum * mainPN[1].sample_reward[:, i])
        v_0 = 2 * tf.reduce_sum(v_0) / batch_size

        if against_destabilizer_exploiter:
            v_1 = 2 * v_1 / batch_size
        else:
            v_1 = 2 * tf.reduce_sum(v_1) / batch_size

    mainPN[0].v_0_log = v_0
    mainPN[1].v_1_log = v_1
    actor_target_error_0 = (mainPN[0].target-tf.stop_gradient(mainPN[0].value))
    v_0_pi_0 = 2*tf.reduce_sum((actor_target_error_0* mainPN[0].gamma_array) * mainPN[0].log_pi_action_bs_t) / \
               batch_size
    v_0_pi_1 = 2*tf.reduce_sum((actor_target_error_0 * mainPN[1].gamma_array) * mainPN[1].log_pi_action_bs_t) / \
               batch_size

    actor_target_error_1 = (mainPN[1].target-tf.stop_gradient(mainPN[1].value))

    v_1_pi_0 = 2*tf.reduce_sum((actor_target_error_1 * mainPN[0].gamma_array) * mainPN[0].log_pi_action_bs_t) / batch_size
    v_1_pi_1 = 2*tf.reduce_sum((actor_target_error_1 * mainPN[1].gamma_array) * mainPN[1].log_pi_action_bs_t) / batch_size

    mainPN[0].actor_target_error = actor_target_error_0
    mainPN[1].actor_target_error = actor_target_error_1
    mainPN[0].actor_loss = v_0_pi_0
    mainPN[1].actor_loss = v_1_pi_1
    mainPN[0].value_used_for_correction = v_0
    mainPN[1].value_used_for_correction = v_1

    v_0_grad_theta_0 = flatgrad(v_0_pi_0, mainPN[0].parameters)
    v_0_grad_theta_1 = flatgrad(v_0_pi_1, mainPN[1].parameters)

    v_1_grad_theta_0 = flatgrad(v_1_pi_0, mainPN[0].parameters)
    v_1_grad_theta_1 = flatgrad(v_1_pi_1, mainPN[1].parameters)

    mainPN[0].grad = v_0_grad_theta_0
    mainPN[1].grad = v_1_grad_theta_1
    mainPN[0].grad_sum = tf.math.reduce_sum(v_0_grad_theta_0)
    mainPN[1].grad_sum = tf.math.reduce_sum(v_1_grad_theta_1)

    mainPN[0].grad_v_1 = v_1_grad_theta_0
    mainPN[1].grad_v_0 = v_0_grad_theta_1

    if corrections:
        v_0_grad_theta_0_wrong = flatgrad(v_0, mainPN[0].parameters)
        if against_destabilizer_exploiter:
            # v_1_grad_theta_1_wrong_splits = [ flatgrad(v_1[i], mainPN[1].parameters) for i in range(batch_size)]
            # v_1_grad_theta_1_wrong = tf.stack(v_1_grad_theta_1_wrong_splits, axis=1)

            v_1_grad_theta_1_wrong = tf.vectorized_map(partial(flatgrad, var_list=mainPN[1].parameters), v_1)
        else:
            v_1_grad_theta_1_wrong = flatgrad(v_1, mainPN[1].parameters)

        param_len = v_0_grad_theta_0_wrong.get_shape()[0].value
        # param_len = -1

        if against_destabilizer_exploiter:
            multiply0 = tf.matmul(
                tf.reshape(tf.stop_gradient(v_0_grad_theta_1), [1, param_len]),
                tf.reshape(v_1_grad_theta_1_wrong, [param_len, batch_size])
            )
        else:
            multiply0 = tf.matmul(
                tf.reshape(tf.stop_gradient(v_0_grad_theta_1), [1, param_len]),
                tf.reshape(v_1_grad_theta_1_wrong, [param_len, 1])
            )
        multiply1 = tf.matmul(
            tf.reshape(tf.stop_gradient(v_1_grad_theta_0), [1, param_len]),
            tf.reshape(v_0_grad_theta_0_wrong, [param_len, 1])
        )

        if against_destabilizer_exploiter:
            second_order0 = flatgrad(multiply0, mainPN[0].parameters)
            second_order0 = second_order0[:, None]

            # second_order0_splits = [flatgrad(multiply0[:, i], mainPN[0].parameters) for i in range(batch_size)]
            # second_order0 = tf.stack(second_order0_splits, axis=1)

            # second_order0 = tf.vectorized_map(partial(flatgrad, var_list=mainPN[0].parameters), multiply0[0, :])
            # second_order0 = tf.reshape(second_order0, [param_len, batch_size])
        else:
            second_order0 = flatgrad(multiply0, mainPN[0].parameters)
        second_order1 = flatgrad(multiply1, mainPN[1].parameters)

        mainPN[0].multiply0 = multiply0
        mainPN[0].v_0_grad_01 = second_order0
        mainPN[1].v_1_grad_10 = second_order1
        mainPN[0].second_order = tf.math.reduce_sum(second_order0)
        mainPN[1].second_order = tf.math.reduce_sum(second_order1)

        if against_destabilizer_exploiter:
            second_order0 = tf.math.reduce_sum(second_order0, axis=1)

        second_order0 = (second_order0 * lola_correction_multiplier)
        second_order1 = (second_order1 * lola_correction_multiplier)
        if clip_lola_correction_norm:
            second_order0 = tf.clip_by_norm(second_order0, clip_lola_correction_norm, axes=None, name=None)
            second_order1 = tf.clip_by_norm(second_order1, clip_lola_correction_norm, axes=None, name=None)
        if clip_lola_actor_norm:
            v_0_grad_theta_0 = tf.clip_by_norm(v_0_grad_theta_0, clip_lola_actor_norm, axes=None, name=None)
            v_1_grad_theta_1 = tf.clip_by_norm(v_1_grad_theta_1, clip_lola_actor_norm, axes=None, name=None)



        delta_0 = v_0_grad_theta_0 + second_order0
        delta_1 = v_1_grad_theta_1 + second_order1

        if clip_lola_update_norm:
            delta_0 = tf.clip_by_norm(delta_0, clip_lola_update_norm, axes=None, name=None)
            delta_1 = tf.clip_by_norm(delta_1, clip_lola_update_norm, axes=None, name=None)

        mainPN[0].delta = delta_0
        mainPN[1].delta = delta_1
    else:
        mainPN[0].delta = v_0_grad_theta_0
        mainPN[1].delta = v_1_grad_theta_1

        # To prevent some logic about logging stuff
        mainPN[0].v_0_grad_01 = tf.reduce_sum(v_0_grad_theta_0) * 0.0
        mainPN[1].v_1_grad_10 = tf.reduce_sum(v_0_grad_theta_0) * 0.0





def simple_actor_training_func(policy_network, opp_policy_network, batch_size, trace_length, cube=None):

    # not mem_efficient
    if cube is not None:
        ac_logp0 = tf.reshape(policy_network.log_pi_action_bs_t,
                              [batch_size, 1, trace_length])
        ac_logp1 = tf.reshape(opp_policy_network.log_pi_action_bs_t,
                              [batch_size, trace_length, 1])
        mat_1 = tf.reshape(tf.squeeze(tf.matmul(ac_logp1, ac_logp0)),
                           [batch_size, 1, trace_length * trace_length])

        v_0 = tf.matmul(tf.reshape(policy_network.sample_reward, [batch_size, trace_length, 1]), mat_1)
        v_0 = tf.reshape(v_0, [batch_size, trace_length, trace_length, trace_length])

        v_1 = tf.matmul(tf.reshape(opp_policy_network.sample_reward, [batch_size, trace_length, 1]), mat_1)
        v_1 = tf.reshape(v_1, [batch_size, trace_length, trace_length, trace_length])

        v_0 = 2 * tf.reduce_sum(v_0 * cube) / batch_size
        v_1 = 2 * tf.reduce_sum(v_1 * cube) / batch_size
    # wt mem_efficient
    else:
        ac_logp0 = tf.reshape(policy_network.log_pi_action_bs_t,
                              [batch_size, trace_length])
        ac_logp1 = tf.reshape(opp_policy_network.log_pi_action_bs_t,
                              [batch_size, trace_length])

        # Static exclusive cumsum
        ac_logp0_cumsum = [tf.constant(0.)]
        ac_logp1_cumsum = [tf.constant(0.)]
        for i in range(trace_length - 1):
            ac_logp0_cumsum.append(tf.add(ac_logp0_cumsum[-1], ac_logp0[:, i]))
            ac_logp1_cumsum.append(tf.add(ac_logp1_cumsum[-1], ac_logp1[:, i]))

        # Compute v_0 and v_1
        mat_cumsum = ac_logp0[:, 0] * ac_logp1[:, 0]
        v_0 = mat_cumsum * policy_network.sample_reward[:, 0]
        v_1 = mat_cumsum * opp_policy_network.sample_reward[:, 0]
        for i in range(1, trace_length):
            mat_cumsum = tf.add(mat_cumsum, ac_logp0[:, i] * ac_logp1[:, i])
            mat_cumsum = tf.add(mat_cumsum, ac_logp0_cumsum[i] * ac_logp1[:, i])
            mat_cumsum = tf.add(mat_cumsum, ac_logp1_cumsum[i] * ac_logp0[:, i])
            v_0 = tf.add(v_0, mat_cumsum * policy_network.sample_reward[:, i])
            v_1 = tf.add(v_1, mat_cumsum * opp_policy_network.sample_reward[:, i])
        v_0 = 2 * tf.reduce_sum(v_0) / batch_size
        v_1 = 2 * tf.reduce_sum(v_1) / batch_size

    policy_network.v_0_log = v_0
    actor_target_error_0 = (policy_network.target-tf.stop_gradient(policy_network.value))
    v_0_pi_0 = 2*tf.reduce_sum((actor_target_error_0* policy_network.gamma_array) * policy_network.log_pi_action_bs_t) / \
               batch_size
    # v_1_pi_0 = 2*tf.reduce_sum((actor_target_error_1 * policy_network.gamma_array) * policy_network.log_pi_action_bs_t) / batch_size

    policy_network.actor_target_error = actor_target_error_0
    policy_network.actor_loss = v_0_pi_0
    policy_network.value_used_for_correction = v_0

    v_0_grad_theta_0 = flatgrad(v_0_pi_0, policy_network.parameters)

    # v_1_grad_theta_0 = flatgrad(v_1_pi_0, policy_network.parameters)

    policy_network.grad = v_0_grad_theta_0
    policy_network.grad_sum = tf.math.reduce_sum(v_0_grad_theta_0)

    # policy_network.grad_v_1 = v_1_grad_theta_0

    policy_network.delta = v_0_grad_theta_0
    # To prevent some logic about logging stuff
    policy_network.v_0_grad_01 = tf.reduce_sum(v_0_grad_theta_0) * 0.0

