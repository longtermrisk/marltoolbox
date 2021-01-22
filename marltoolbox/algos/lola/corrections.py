"""
The magic corrections of LOLA.
"""
import tensorflow as tf
from tensorflow.python.ops import math_ops

from marltoolbox.algos.lola.utils import flatgrad


def corrections_func(mainPN, batch_size, trace_length,
                     corrections=False, cube=None, clip_lola_update_norm=False,
                     lola_correction_multiplier=1.0,
                     clip_lola_correction_norm=False,
                     clip_lola_actor_norm=False):
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
        v_1 = 2 * tf.reduce_sum(v_1) / batch_size

    mainPN[0].v_0_log = v_0
    mainPN[1].v_1_log = v_1
    # print_op_15 = tf.print("v_0", tf.math.reduce_sum(v_0))
    # print_op_16 = tf.print("v_1", tf.math.reduce_sum(v_1))
    # print_op_17 = tf.print("mainPN[0].target", tf.math.reduce_sum(mainPN[0].target))
    # print_op_18 = tf.print("mainPN[1].target", tf.math.reduce_sum(mainPN[1].target))
    # print_op_19 = tf.print("mainPN[0].value", tf.math.reduce_sum(mainPN[0].value))
    # print_op_20 = tf.print("mainPN[1].value", tf.math.reduce_sum(mainPN[1].value))
    # print_op_21 = tf.print("mainPN[0].gamma_array", tf.math.reduce_sum(mainPN[0].gamma_array))
    # print_op_22 = tf.print("mainPN[1].gamma_array", tf.math.reduce_sum(mainPN[1].gamma_array))
    # with tf.control_dependencies([print_op_15, print_op_16, print_op_17, print_op_18, print_op_19, print_op_20,
    #                               print_op_21, print_op_22]):
    actor_target_error_0 = (mainPN[0].target-tf.stop_gradient(mainPN[0].value))
    # actor_target_error_0 = (mainPN[0].target)
    # actor_target_error_0 = (mainPN[0].target-tf.math.reduce_mean(mainPN[0].target, axis=0))
    v_0_pi_0 = 2*tf.reduce_sum((actor_target_error_0* mainPN[0].gamma_array) * mainPN[0].log_pi_action_bs_t) / \
               batch_size
    v_0_pi_1 = 2*tf.reduce_sum((actor_target_error_0 * mainPN[1].gamma_array) * mainPN[1].log_pi_action_bs_t) / \
               batch_size

    actor_target_error_1 = (mainPN[1].target-tf.stop_gradient(mainPN[1].value))
    # actor_target_error_1 = (mainPN[1].target)
    # actor_target_error_1 = (mainPN[1].target-tf.math.reduce_mean(mainPN[1].target, axis=0))
    v_1_pi_0 = 2*tf.reduce_sum((actor_target_error_1 * mainPN[0].gamma_array) * mainPN[0].log_pi_action_bs_t) / batch_size
    v_1_pi_1 = 2*tf.reduce_sum((actor_target_error_1 * mainPN[1].gamma_array) * mainPN[1].log_pi_action_bs_t) / batch_size

    mainPN[0].actor_target_error = actor_target_error_0
    mainPN[1].actor_target_error = actor_target_error_1
    mainPN[0].actor_loss = v_0_pi_0
    mainPN[1].actor_loss = v_1_pi_1
    mainPN[0].value_used_for_correction = v_0
    mainPN[1].value_used_for_correction = v_1
    # print_op_77 = tf.print("mainPN[0].log_pi_action_bs_t", tf.math.reduce_sum(mainPN[0].log_pi_action_bs_t))
    # print_op_78 = tf.print("mainPN[1].log_pi_action_bs_t", tf.math.reduce_sum(mainPN[1].log_pi_action_bs_t))
    # print_op_7 = tf.print("v_1_pi_0", tf.math.reduce_sum(v_1_pi_0))
    # print_op_8 = tf.print("v_1_pi_1", tf.math.reduce_sum(v_1_pi_1))
    # print_op_13 = tf.print("v_0_pi_0", tf.math.reduce_sum(v_0_pi_0))
    # print_op_14 = tf.print("v_0_pi_1", tf.math.reduce_sum(v_0_pi_1))
    # with tf.control_dependencies([print_op_7, print_op_8, print_op_13, print_op_14,
    #                               print_op_77, print_op_78]):
    v_0_grad_theta_0 = flatgrad(v_0_pi_0, mainPN[0].parameters)
    v_0_grad_theta_1 = flatgrad(v_0_pi_1, mainPN[1].parameters)

    v_1_grad_theta_0 = flatgrad(v_1_pi_0, mainPN[0].parameters)
    v_1_grad_theta_1 = flatgrad(v_1_pi_1, mainPN[1].parameters)

    # print_op_9 = tf.print("v_1_grad_theta_0", tf.math.reduce_sum(v_1_grad_theta_0))
    # print_op_10 = tf.print("v_1_grad_theta_1", tf.math.reduce_sum(v_1_grad_theta_1))
    # print_op_11 = tf.print("v_0_grad_theta_0", tf.math.reduce_sum(v_0_grad_theta_0))
    # print_op_12 = tf.print("v_0_grad_theta_1", tf.math.reduce_sum(v_0_grad_theta_1))

    mainPN[0].grad = v_0_grad_theta_0
    mainPN[1].grad = v_1_grad_theta_1
    mainPN[0].grad_sum = tf.math.reduce_sum(v_0_grad_theta_0)
    mainPN[1].grad_sum = tf.math.reduce_sum(v_1_grad_theta_1)

    # extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # with tf.control_dependencies(extra_update_ops):
    mainPN[0].grad_v_1 = v_1_grad_theta_0
    mainPN[1].grad_v_0 = v_0_grad_theta_1

    if corrections:
        # with tf.control_dependencies([print_op_9, print_op_10, print_op_11, print_op_12]):
        v_0_grad_theta_0_wrong = flatgrad(v_0, mainPN[0].parameters)
        v_1_grad_theta_1_wrong = flatgrad(v_1, mainPN[1].parameters)

        # print_op_5 = tf.print("v_0_grad_theta_0_wrong", tf.math.reduce_sum(v_0_grad_theta_0_wrong))
        # print_op_6 = tf.print("v_1_grad_theta_1_wrong", tf.math.reduce_sum(v_1_grad_theta_1_wrong))
        # with tf.control_dependencies([print_op_5,print_op_6]):

        param_len = v_0_grad_theta_0_wrong.get_shape()[0].value

        multiply0 = tf.matmul(
            tf.reshape(tf.stop_gradient(v_0_grad_theta_1), [1, param_len]),
            tf.reshape(v_1_grad_theta_1_wrong, [param_len, 1])
        )
        multiply1 = tf.matmul(
            tf.reshape(tf.stop_gradient(v_1_grad_theta_0), [1, param_len]),
            tf.reshape(v_0_grad_theta_0_wrong, [param_len, 1])
        )

        # print_op_3 = tf.print("multiply0", tf.math.reduce_sum(multiply0))
        # print_op_4 = tf.print("multiply1", tf.math.reduce_sum(multiply1))
        # with tf.control_dependencies([print_op_3,print_op_4]):
        second_order0 = flatgrad(multiply0, mainPN[0].parameters)
        second_order1 = flatgrad(multiply1, mainPN[1].parameters)

        mainPN[0].v_0_grad_01 = second_order0
        mainPN[1].v_1_grad_10 = second_order1
        mainPN[0].second_order = tf.math.reduce_sum(second_order0)
        mainPN[1].second_order = tf.math.reduce_sum(second_order1)

        # print_op_1 = tf.print("second_order0", tf.math.reduce_sum(second_order0))
        # print_op_2 = tf.print("second_order1", tf.math.reduce_sum(second_order1))
        # print_op_23 = tf.print("mainPN[0].loss", tf.math.reduce_sum(mainPN[0].loss))
        # print_op_24 = tf.print("mainPN[1].loss", tf.math.reduce_sum(mainPN[1].loss))
        # print_op_25 = tf.print("mainPN[0].next_v", tf.math.reduce_sum(mainPN[0].next_v))
        # print_op_26 = tf.print("mainPN[1].next_v", tf.math.reduce_sum(mainPN[1].next_v))
        # with tf.control_dependencies([print_op_1,print_op_2, print_op_23, print_op_24,
        #                               print_op_25, print_op_26]):

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
            # delta_0_l2sum = math_ops.reduce_sum(delta_0 * delta_0, None, keepdims=True)
            # delta_1_l2sum = math_ops.reduce_sum(delta_1 * delta_1, None, keepdims=True)
            # print_op_1 = tf.print("delta_0_l2sum", delta_0_l2sum)
            # print_op_2 = tf.print("delta_1_l2sum", delta_1_l2sum)
            # with tf.control_dependencies([print_op_1, print_op_2]):
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



