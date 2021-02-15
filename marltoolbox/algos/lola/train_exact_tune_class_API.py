##########
# Code modified from: https://github.com/alshedivat/lola
##########

"""
Trains LOLA on IPD or MatchingPennies with exact value functions.

Note: Interfaces are a little different form the code that estimates values,
      hence moved into a separate module.
"""
import json
import os
import random

import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
from ray import tune

from marltoolbox.algos.lola.utils import GetFlatWtSess, SetFromFlatWtSess, flatgrad


class Qnetwork:
    """
    Q-network that is either a look-up table or an MLP with 1 hidden layer.
    """

    def __init__(self, myScope, num_hidden, sess, simple_net=True):
        with tf.variable_scope(myScope):
            self.input_place = tf.placeholder(shape=[5], dtype=tf.int32)
            if simple_net:
                self.p_act = tf.Variable(tf.random_normal([5, 1]))
                # self.action = self.p_act[self.input_place]
            else:
                act = tf.nn.tanh(
                    layers.fully_connected(
                        tf.one_hot(self.input_place, 5, dtype=tf.float32),
                        num_outputs=num_hidden, activation_fn=None
                    )
                )
                self.p_act = layers.fully_connected(
                    act, num_outputs=1, activation_fn=None
                )
                # self.action = self.p_act
        self.parameters = []
        for i in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                   scope=myScope):
            self.parameters.append(i)  # i.name if you want just a name
        self.setparams = SetFromFlatWtSess(self.parameters, sess)
        self.getparams = GetFlatWtSess(self.parameters, sess)


def update(mainQN, lr, final_delta_1_v, final_delta_2_v):
    update_theta_1 = mainQN[0].setparams(
        mainQN[0].getparams() + lr * np.squeeze(final_delta_1_v))
    update_theta_2 = mainQN[1].setparams(
        mainQN[1].getparams() + lr * np.squeeze(final_delta_2_v))


def corrections_func(mainQN, corrections, gamma, pseudo, reg):
    mainQN[0].lr_correction = tf.placeholder(shape=[1], dtype=tf.float32)
    mainQN[1].lr_correction = tf.placeholder(shape=[1], dtype=tf.float32)

    theta_1_all = mainQN[0].p_act
    theta_2_all = mainQN[1].p_act
    theta_1 = tf.slice(theta_1_all, [0, 0], [4, 1])
    theta_2 = tf.slice(theta_2_all, [0, 0], [4, 1])

    theta_1_0 = tf.slice(theta_1_all, [4, 0], [1, 1])
    theta_2_0 = tf.slice(theta_2_all, [4, 0], [1, 1])

    p_1 = tf.nn.sigmoid(theta_1)
    p_2 = tf.nn.sigmoid(theta_2)
    mainQN[0].policy = tf.nn.sigmoid(theta_1_all)
    mainQN[1].policy = tf.nn.sigmoid(theta_2_all)

    p_1_0 = tf.nn.sigmoid(theta_1_0)
    p_2_0 = tf.nn.sigmoid(theta_2_0)

    p_1_0_v = tf.concat([p_1_0, (1 - p_1_0)], 0)
    p_2_0_v = tf.concat([p_2_0, (1 - p_2_0)], 0)

    s_0 = tf.reshape(tf.matmul(p_1_0_v, tf.transpose(p_2_0_v)), [-1, 1])

    # CC, CD, DC, DD
    P = tf.concat([
        tf.multiply(p_1, p_2),
        tf.multiply(p_1, 1 - p_2),
        tf.multiply(1 - p_1, p_2),
        tf.multiply(1 - p_1, 1 - p_2)
    ], 1)
    R_1 = tf.placeholder(shape=[4, 1], dtype=tf.float32)
    R_2 = tf.placeholder(shape=[4, 1], dtype=tf.float32)

    I_m_P = tf.diag([1.0, 1.0, 1.0, 1.0]) - P * gamma
    v_0 = tf.matmul(
        tf.matmul(tf.matrix_inverse(I_m_P), R_1), s_0,
        transpose_a=True
    )
    v_1 = tf.matmul(
        tf.matmul(tf.matrix_inverse(I_m_P), R_2), s_0,
        transpose_a=True
    )
    if reg > 0:
        for indx, _ in enumerate(mainQN[0].parameters):
            v_0 -= reg * tf.reduce_sum(
                tf.nn.l2_loss(tf.square(mainQN[0].parameters[indx]))
            )
            v_1 -= reg * tf.reduce_sum(
                tf.nn.l2_loss(tf.square(mainQN[1].parameters[indx]))
            )
    v_0_grad_theta_0 = flatgrad(v_0, mainQN[0].parameters)
    v_0_grad_theta_1 = flatgrad(v_0, mainQN[1].parameters)

    v_1_grad_theta_0 = flatgrad(v_1, mainQN[0].parameters)
    v_1_grad_theta_1 = flatgrad(v_1, mainQN[1].parameters)

    v_0_grad_theta_0_wrong = flatgrad(v_0, mainQN[0].parameters)
    v_1_grad_theta_1_wrong = flatgrad(v_1, mainQN[1].parameters)
    param_len = v_0_grad_theta_0_wrong.get_shape()[0].value

    if pseudo:
        multiply0 = tf.matmul(
            tf.reshape(v_0_grad_theta_1, [1, param_len]),
            tf.reshape(v_1_grad_theta_1, [param_len, 1])
        )
        multiply1 = tf.matmul(
            tf.reshape(v_1_grad_theta_0, [1, param_len]),
            tf.reshape(v_0_grad_theta_0, [param_len, 1])
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

    second_order0 = flatgrad(multiply0, mainQN[0].parameters)
    second_order1 = flatgrad(multiply1, mainQN[1].parameters)

    mainQN[0].R1 = R_1
    mainQN[1].R1 = R_2
    mainQN[0].v = v_0
    mainQN[1].v = v_1
    mainQN[0].delta = v_0_grad_theta_0
    mainQN[1].delta = v_1_grad_theta_1
    mainQN[0].delta += tf.multiply(second_order0, mainQN[0].lr_correction)
    mainQN[1].delta += tf.multiply(second_order1, mainQN[1].lr_correction)


class LOLAExact(tune.Trainable):

    def _init_lola(self, env, *, num_episodes=50, trace_length=200,
                   simple_net=True, corrections=True, pseudo=False,
                   num_hidden=10, reg=0.0, lr=1., lr_correction=0.5, gamma=0.96,
                   **kwargs):

        print("args not used:", kwargs)
        # print("is_training",is_training)

        self.num_episodes = num_episodes
        self.trace_length = trace_length
        self.simple_net = simple_net
        self.corrections = corrections
        self.pseudo = pseudo
        self.num_hidden = num_hidden
        self.reg = reg
        self.lr = lr
        self.lr_correction = lr_correction
        self.gamma = gamma

        self.timestep = 0

        graph = tf.Graph()

        with graph.as_default() as g:
            self.sess = tf.Session()
            # Get info about the env
            # self.payout_mat_1 = env.payout_mat
            # self.payout_mat_2 = env.payout_mat.T

            if env == "IPD":
                self.payout_mat_1 = np.array([[-1., 0.], [-3., -2.]])
                self.payout_mat_2 = np.array([[-1., -3], [+0., -2.]])
            elif env == "AsymBoS":
                self.payout_mat_1 = np.array([[+3.5, 0.], [0., +1]])
                self.payout_mat_2 = np.array([[+1., 0.], [0., +3.]])
            else:
                raise ValueError(f"exp_name: {env}")

            # Sanity

            # Q-networks
            self.mainQN = []
            for agent in range(2):
                self.mainQN.append(Qnetwork('main' + str(agent), self.num_hidden, self.sess, self.simple_net))

            # Corrections
            corrections_func(self.mainQN, self.corrections, self.gamma, self.pseudo, self.reg)

            self.results = []
            self.norm = 1 / (1 - self.gamma)
            self.init = tf.global_variables_initializer()

            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
            # + tf.get_collection_ref("batch_norm_non_trainable_variables_co‌​llection")

    # TODO add something to not load and create everything when only evaluating with RLLib

    def setup(self, config):
        print("_init_lola", config)
        self._init_lola(**config)

    def step(self):
        self.timestep += 1

        self.sess.run(self.init)
        lr_coor = np.ones(1) * self.lr_correction

        log_items = {}
        log_items['episode'] = self.timestep

        res = []
        params_time = []
        delta_time = []
        input_vals = np.reshape(np.array(range(5)) + 1, [-1])
        for i in range(self.trace_length):
            params0 = self.mainQN[0].getparams()
            params1 = self.mainQN[1].getparams()
            outputs = [
                self.mainQN[0].delta, self.mainQN[1].delta, self.mainQN[0].v, self.mainQN[1].v,
                self.mainQN[0].policy, self.mainQN[1].policy
            ]
            # print("input_vals", input_vals)
            update1, update2, v1, v2, policy1, policy2 = self.sess.run(
                outputs,
                feed_dict={
                    self.mainQN[0].input_place: input_vals,
                    self.mainQN[1].input_place: input_vals,
                    self.mainQN[0].R1: np.reshape(self.payout_mat_2, [-1, 1]),
                    self.mainQN[1].R1: np.reshape(self.payout_mat_1, [-1, 1]),
                    self.mainQN[0].lr_correction: lr_coor,
                    self.mainQN[1].lr_correction: lr_coor
                }
            )
            print("epi", self.timestep, "return_1", v1[0][0] / self.norm, "return_2", v2[0][0] / self.norm)
            update(self.mainQN, self.lr, update1, update2)
            params_time.append([params0, params1])
            delta_time.append([update1, update2])

            log_items['ret1'] = v1[0][0] / self.norm
            log_items['ret2'] = v2[0][0] / self.norm
            res.append([v1[0][0] / self.norm, v2[0][0] / self.norm])
        self.results.append(res)

        log_items["episodes_total"] = self.timestep

        return log_items

    def save_checkpoint(self, checkpoint_dir):
        path = os.path.join(checkpoint_dir, "checkpoint.json")
        tf_checkpoint_path = os.path.join(checkpoint_dir, "checkpoint")
        tf_checkpoint_dir, tf_checkpoint_filename = os.path.split(tf_checkpoint_path)
        checkpoint = {
            "timestep": self.timestep,
            "tf_checkpoint_dir": tf_checkpoint_dir,
            "tf_checkpoint_filename": tf_checkpoint_filename,
        }
        with open(path, "w") as f:
            json.dump(checkpoint, f, sort_keys=True, indent=4)

        # TF v1
        save_path = self.saver.save(self.sess, f"{tf_checkpoint_path}.ckpt")

        return path

    def load_checkpoint(self, checkpoint_path):

        checkpoint_path = os.path.expanduser(checkpoint_path)
        print('Loading Model...',checkpoint_path)
        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)
        print("checkpoint", checkpoint)

        # Support VM and local (manual) loading
        tf_checkpoint_dir, _ = os.path.split(checkpoint_path)
        print("tf_checkpoint_dir", tf_checkpoint_dir)
        ckpt = tf.train.get_checkpoint_state(tf_checkpoint_dir,
                                             latest_filename=f'{checkpoint["tf_checkpoint_filename"]}')
        tail, head = os.path.split(ckpt.model_checkpoint_path)
        ckpt.model_checkpoint_path = os.path.join(tf_checkpoint_dir, head)
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)


    def cleanup(self):
        self.sess.close()
        super().cleanup()

    def _get_agent_to_use(self, policy_id):
        if policy_id == "player_red":
            agent_n = 0
        elif policy_id == "player_blue":
            agent_n = 1
        elif policy_id == "player_row":
            agent_n = 0
        elif policy_id == "player_col":
            agent_n = 1
        else:
            raise ValueError(f"policy_id {policy_id}")
        return agent_n

    def _preprocess_obs(self, single_obs, agent_to_use):
        single_obs = np.where(single_obs == 1)[0][0]

        return single_obs

    def _post_process_action(self, action):
        return action[None, ...]  # add batch dim

    def compute_actions(self, policy_id:str, obs_batch:list):
        # because of the LSTM
        assert len(obs_batch) == 1


        for single_obs in obs_batch:
            agent_to_use = self._get_agent_to_use(policy_id)
            obs = self._preprocess_obs(single_obs, agent_to_use)
            input_vals = np.reshape(np.array(range(5)) + 1, [-1])
            policy = self.sess.run(
                [self.mainQN[agent_to_use].policy],
                feed_dict={
                    self.mainQN[agent_to_use].input_place: input_vals,
                }
            )
            coop_proba = policy[0][obs][0]
            if coop_proba > random.random():
                action = np.array(0)
            else:
                action = np.array(1)

        action = self._post_process_action(action)

        state_out = []
        extra_fetches = {}
        return action, state_out, extra_fetches
