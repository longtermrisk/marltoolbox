"""
Policy and value networks used in LOLA experiments.
"""
import copy

import tensorflow.contrib.layers as layers
import torch
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.agents.dqn import DEFAULT_CONFIG, DQNTorchPolicy
from ray.rllib.evaluation.sample_batch_builder import (
    MultiAgentSampleBatchBuilder,
)
from ray.rllib.execution.replay_buffer import LocalReplayBuffer
from tensorflow.python.ops import math_ops

from marltoolbox.algos.lola.utils import *


class Pnetwork:
    """
    Recurrent policy network used in Coin Game experiments.
    """

    def __init__(
        self,
        myScope,
        h_size,
        agent,
        env,
        trace_length,
        batch_size,
        reuse=None,
        step=False,
        changed_config=False,
        ac_lr=1.0,
        use_MAE=False,
        use_toolbox_env=False,
        clip_loss_norm=False,
        sess=None,
        entropy_coeff=1.0,
        weigth_decay=0.01,
        use_critic=False,
        use_destabilizer_in_policy=False,
    ):
        self.sess = sess

        # if use_toolbox_env:
        ob_space_shape = list(env.OBSERVATION_SPACE.shape)
        # else:
        #     ob_space_shape = env.ob_space_shape
        print("ob_space_shape", ob_space_shape)

        if step:
            trace_length = 1
        else:
            trace_length = trace_length
        with tf.variable_scope(myScope, reuse=reuse):
            self.batch_size = batch_size
            zero_state = tf.zeros((batch_size, h_size * 2), dtype=tf.float32)
            self.gamma_array = tf.placeholder(
                shape=[1, trace_length], dtype=tf.float32, name="gamma_array"
            )
            self.gamma_array_inverse = tf.placeholder(
                shape=[1, trace_length],
                dtype=tf.float32,
                name="gamma_array_inv",
            )

            self.lstm_state = tf.placeholder(
                shape=[batch_size, h_size * 2],
                dtype=tf.float32,
                name="lstm_state",
            )

            if step:
                self.state_input = tf.placeholder(
                    shape=[self.batch_size] + ob_space_shape,
                    dtype=tf.float32,
                    name="state_input",
                )
                lstm_state = self.lstm_state
            else:
                self.state_input = tf.placeholder(
                    shape=[batch_size * trace_length] + ob_space_shape,
                    dtype=tf.float32,
                    name="state_input",
                )
                lstm_state = zero_state

            self.sample_return = tf.placeholder(
                shape=[None, trace_length],
                dtype=tf.float32,
                name="sample_return",
            )
            self.sample_reward = tf.placeholder(
                shape=[None, trace_length],
                dtype=tf.float32,
                name="sample_reward",
            )
            self.sample_reward_bis = tf.placeholder(
                shape=[None, trace_length],
                dtype=tf.float32,
                name="sample_reward_bis",
            )

            self.j = tf.placeholder(shape=[None], dtype=tf.float32, name="j")

            self.loss_multiplier = tf.placeholder(
                shape=[1], dtype=tf.float32, name="loss_multiplier"
            )

            self.is_training = tf.placeholder(tf.bool, name="is_training")

            with tf.variable_scope("input_proc", reuse=reuse):
                output = layers.convolution2d(
                    self.state_input,
                    stride=1,
                    kernel_size=3,
                    num_outputs=20,
                    normalizer_fn=layers.batch_norm,
                    activation_fn=tf.nn.relu,
                    # weights_initializer=tf.contrib.layers.xavier_initializer(
                    #     uniform=False,
                    # ),
                    # biases_initializer=tf.contrib.layers.xavier_initializer(
                    #     # uniform=True,
                    #     uniform=False,
                    # )
                    # weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                    #     factor=1.0,
                    #     mode='FAN_AVG',
                    #     uniform=True,
                    #     # uniform=False,
                    # ),
                    # biases_initializer=tf.contrib.layers.variance_scaling_initializer(
                    #     factor=1.0,
                    #     mode='FAN_AVG',
                    #     uniform=True,
                    #     # uniform=False,
                    # ),
                )
                output = layers.convolution2d(
                    output,
                    stride=1,
                    kernel_size=3,
                    num_outputs=20,
                    normalizer_fn=layers.batch_norm,
                    activation_fn=tf.nn.relu,
                    # weights_initializer=tf.contrib.layers.xavier_initializer(
                    #     uniform=False,
                    # ),
                    # biases_initializer=tf.contrib.layers.xavier_initializer(
                    #     # uniform=True,
                    #     uniform=False,
                    # )
                    # weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                    #     factor=1.0,
                    #     mode='FAN_AVG',
                    #     uniform=True,
                    #     # uniform=False,
                    # ),
                    # biases_initializer=tf.contrib.layers.variance_scaling_initializer(
                    #     factor=1.0,
                    #     mode='FAN_AVG',
                    #     uniform=True,
                    #     # uniform=False,
                    # ),
                )

                output = layers.flatten(output)

                # Without a Critic
                if use_critic:
                    self.value = tf.reshape(
                        layers.fully_connected(tf.nn.relu(output), 1),
                        [-1, trace_length],
                    )
                else:
                    self.value = (
                        tf.reshape(
                            layers.fully_connected(tf.nn.relu(output), 1),
                            [-1, trace_length],
                        )
                        * 0.0
                    )

            if step:
                output_seq = batch_to_seq(output, self.batch_size, 1)
            else:
                output_seq = batch_to_seq(
                    output, self.batch_size, trace_length
                )
            output_seq, state_output = lstm(
                output_seq, lstm_state, scope="rnn", nh=h_size
            )
            output = seq_to_batch(output_seq)

            output = layers.fully_connected(
                output,
                num_outputs=env.NUM_ACTIONS,
                activation_fn=None,
                # weights_initializer=tf.contrib.layers.xavier_initializer(
                #     uniform=False,
                # ),
                # biases_initializer=tf.contrib.layers.xavier_initializer(
                #     # uniform=True,
                #     uniform=False,
                # )
                # weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                #     factor=1.0,
                #     mode='FAN_AVG',
                #     uniform=True,
                #     # uniform=False,
                # ),
                # biases_initializer=tf.contrib.layers.variance_scaling_initializer(
                #     factor=1.0,
                #     mode='FAN_AVG',
                #     uniform=True,
                #     # uniform=False,
                # ),
            )
            self.log_pi = tf.nn.log_softmax(output)
            self.lstm_state_output = state_output

            entropy_temp = self.log_pi * tf.exp(self.log_pi)
            self.entropy = tf.reduce_mean(entropy_temp) * 4

            self.actions = tf.placeholder(
                shape=[None], dtype=tf.int32, name="actions"
            )
            self.actions_onehot = tf.one_hot(
                self.actions, env.NUM_ACTIONS, dtype=tf.float32
            )

            predict = tf.multinomial(self.log_pi, 1)
            self.predict = tf.squeeze(predict)

            self.next_value = tf.placeholder(
                shape=[None, 1], dtype=tf.float32, name="next_value"
            )
            self.next_v = tf.matmul(self.next_value, self.gamma_array_inverse)

            if use_critic:
                self.target = self.sample_reward_bis + self.next_v
            else:
                self.target = self.sample_return + self.next_v

            if not use_MAE:
                self.td_error = tf.square(self.target - self.value) / 2
            else:
                self.td_error = tf.abs(self.target - self.value) / 2

            if use_critic:
                self.loss = tf.reduce_mean(self.td_error)
            else:
                self.loss = tf.reduce_mean(self.td_error) * 0.0

        self.parameters = []
        self.value_params = []
        for i in tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=myScope
        ):
            if not ("value_params" in i.name):
                self.parameters.append(i)  # i.name if you want just a name
            if "input_proc" in i.name:
                self.value_params.append(i)
        print("myScope", myScope)
        print("self.parameters", self.parameters)
        print("self.value_params", self.value_params)
        self.parameters_norm = tf.reduce_sum(
            [tf.reduce_sum(p * p) for p in self.parameters]
        )
        self.value_params_norm = tf.reduce_sum(
            [tf.reduce_sum(p * p) for p in self.value_params]
        )

        if self.sess is not None:
            self.getparams = GetFlatWtSess(self.parameters, self.sess)
            self.setparams = SetFromFlatWtSess(self.parameters, self.sess)
        else:
            self.getparams = GetFlat(self.parameters)
            self.setparams = SetFromFlat(self.parameters)

        if not step:
            self.log_pi_action = tf.reduce_mean(
                tf.multiply(self.log_pi, self.actions_onehot),
                reduction_indices=1,
            )
            self.log_pi_action_bs = tf.reduce_sum(
                tf.reshape(self.log_pi_action, [-1, trace_length]), 1
            )
            self.log_pi_action_bs_t = tf.reshape(
                self.log_pi_action, [self.batch_size, trace_length]
            )
            self.trainer = tf.train.GradientDescentOptimizer(
                learning_rate=ac_lr
            )

            weigths_norm = tf.concat(
                axis=0,
                values=[tf.reshape(v, [numel(v)]) for v in self.parameters],
            )
            weigths_norm = math_ops.reduce_sum(
                weigths_norm * weigths_norm, None, keepdims=True
            )
            weigths_norm = tf.reduce_sum(weigths_norm)
            self.weigths_norm = tf.sqrt(weigths_norm)

            if clip_loss_norm:
                l2sum = math_ops.reduce_sum(
                    self.loss * self.loss, None, keepdims=True
                )
                print_op_1 = tf.print("loss l2sum", l2sum)
                with tf.control_dependencies([print_op_1]):
                    self.loss = tf.clip_by_norm(
                        self.loss, clip_loss_norm, axes=None, name=None
                    )
            if use_critic:
                total_loss = (
                    self.loss
                    + entropy_coeff * self.entropy
                    + weigth_decay * self.weigths_norm
                ) * self.loss_multiplier
            else:
                total_loss = (
                    entropy_coeff * self.entropy
                    + weigth_decay * self.weigths_norm
                ) * self.loss_multiplier

            self.updateModel = self.trainer.minimize(
                total_loss, var_list=self.value_params
            )

        self.param_len = len(self.parameters)

        for var in self.parameters:
            print(var.name, var.get_shape())


class Qnetwork:
    """
    Simple Q-network used in IPD experiments.
    """

    def __init__(
        self,
        myScope,
        agent,
        env,
        batch_size,
        gamma,
        trace_length,
        hidden,
        simple_net,
        lr,
        sess,
    ):

        self.sess = sess

        # The network recieves a frame from the game, flattened into an array.
        # It then resizes it and processes it through four convolutional layers.
        with tf.variable_scope(myScope):
            self.scalarInput = tf.placeholder(
                shape=[None, env.NUM_STATES], dtype=tf.float32
            )
            self.gamma_array = tf.placeholder(
                shape=[1, trace_length], dtype=tf.float32, name="gamma_array"
            )
            self.gamma_array_inverse = tf.placeholder(
                shape=[1, trace_length], dtype=tf.float32, name="gamma_array"
            )

            if simple_net:
                self.logit_vals = tf.Variable(tf.random_normal([5, 1]))
                self.temp = tf.matmul(self.scalarInput, self.logit_vals)
                temp_concat = tf.concat([self.temp, self.temp * 0], 1)
                self.log_pi = tf.nn.log_softmax(temp_concat)
            else:
                act = tf.nn.leaky_relu(
                    layers.fully_connected(
                        self.scalarInput,
                        num_outputs=hidden,
                        activation_fn=None,
                    )
                )
                self.log_pi = tf.nn.log_softmax(
                    layers.fully_connected(
                        act, num_outputs=2, activation_fn=None
                    )
                )
            self.values = tf.Variable(
                tf.random_normal([5, 1]), name="value_params"
            )
            self.value = tf.reshape(
                tf.matmul(self.scalarInput, self.values), [batch_size, -1]
            )
            self.sample_return = tf.placeholder(
                shape=[None, trace_length],
                dtype=tf.float32,
                name="sample_return",
            )
            self.sample_reward = tf.placeholder(
                shape=[None, trace_length],
                dtype=tf.float32,
                name="sample_reward_new",
            )

            self.next_value = tf.placeholder(
                shape=[None, 1], dtype=tf.float32, name="next_value"
            )
            self.next_v = tf.matmul(self.next_value, self.gamma_array_inverse)
            self.target = self.sample_return + self.next_v
            self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
            self.actions_onehot = tf.one_hot(
                self.actions, env.NUM_ACTIONS, dtype=tf.float32
            )

            self.predict = tf.multinomial(self.log_pi, 1)
            self.predict = tf.squeeze(self.predict)
            self.log_pi_action = tf.reduce_mean(
                tf.multiply(self.log_pi, self.actions_onehot),
                reduction_indices=1,
            )

            self.td_error = tf.square(self.target - self.value) / 2
            self.loss = tf.reduce_mean(self.td_error)

        self.parameters = []
        self.value_params = []
        for i in tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=myScope
        ):
            if not ("value_params" in i.name):
                self.parameters.append(i)  # i.name if you want just a name
            else:
                self.value_params.append(i)

        self.trainer = tf.train.GradientDescentOptimizer(
            learning_rate=1
        )  # / arglist.bs)
        self.updateModel = self.trainer.minimize(
            self.loss, var_list=self.value_params
        )

        self.log_pi_action_bs = tf.reduce_sum(
            tf.reshape(self.log_pi_action, [-1, trace_length]), 1
        )
        self.log_pi_action_bs_t = tf.reshape(
            self.log_pi_action, [batch_size, trace_length]
        )
        # self.setparams= SetFromFlat(self.parameters)
        # self.getparams= GetFlat(self.parameters)

        if self.sess is not None:
            self.getparams = GetFlatWtSess(self.parameters, self.sess)
            self.setparams = SetFromFlatWtSess(self.parameters, self.sess)
        else:
            self.getparams = GetFlat(self.parameters)
            self.setparams = SetFromFlat(self.parameters)


class ExperienceBuffer:
    def __init__(self, buffer_size=1000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        if len(self.buffer) + 1 >= self.buffer_size:
            self.buffer[0 : (1 + len(self.buffer)) - self.buffer_size] = []
        self.buffer.append(experience)

    def sample(self, batch_size, trace_length):
        sampled_episodes = self.buffer
        sampledTraces = []
        for episode in sampled_episodes:
            this_episode = list(copy.deepcopy(episode))
            point = np.random.randint(0, len(this_episode) + 1 - trace_length)
            sampledTraces.append(this_episode[point : point + trace_length])
        sampledTraces = np.array(sampledTraces)
        return np.reshape(sampledTraces, [batch_size * trace_length, 6])


class DQNAgent:
    def __init__(
        self,
        env,
        batch_size,
        trace_length,
        grid_size,
        exploiter_base_lr,
        exploiter_decay_lr_in_n_epi,
        exploiter_stop_training_after_n_epi,
        train_exploiter_n_times_per_epi,
    ):

        self.stop_training_after_n_epi = exploiter_stop_training_after_n_epi
        self.train_exploiter_n_times_per_epi = train_exploiter_n_times_per_epi

        # with tf.variable_scope(f"dqn_exploiter"):
        # Create the dqn policy for the exploiter
        dqn_config = copy.deepcopy(DEFAULT_CONFIG)
        dqn_config.update(
            {
                "prioritized_replay": False,
                "double_q": True,
                "buffer_size": 50000,
                "dueling": False,
                "learning_starts": min(
                    int((batch_size - 1) * (trace_length - 1)), 64
                ),
                "model": {
                    "dim": grid_size,
                    "conv_filters": [[16, [3, 3], 1], [32, [3, 3], 1]],
                    # [Channel, [Kernel, Kernel], Stride]]
                    # "fcnet_hiddens": [self.env.NUM_ACTIONS],
                    "max_seq_len": trace_length,
                    # Number of hidden layers for fully connected net
                    "fcnet_hiddens": [64],
                    # Nonlinearity for fully connected net (tanh, relu)
                    "fcnet_activation": "relu",
                },
                # Update the replay buffer with this many samples at once. Note that
                # this setting applies per-worker if num_workers > 1.
                "rollout_fragment_length": 1,
                # Size of a batch sampled from replay buffer for training. Note that
                # if async_updates is set, then each worker returns gradients for a
                # batch of this size.
                "train_batch_size": min(
                    int((batch_size) * (trace_length)), 64
                ),
                "explore": False,
                "grad_clip": 1,
                "gamma": 0.5,
                "lr": exploiter_base_lr,
                # Learning rate schedule
                "lr_schedule": [
                    (0, exploiter_base_lr / 1000),
                    (100, exploiter_base_lr),
                    (exploiter_decay_lr_in_n_epi, exploiter_base_lr / 1e9),
                ],
                "sgd_momentum": 0.9,
            }
        )
        print("dqn_config", dqn_config)

        self.local_replay_buffer = LocalReplayBuffer(
            num_shards=1,
            learning_starts=dqn_config["learning_starts"],
            buffer_size=dqn_config["buffer_size"],
            replay_batch_size=dqn_config["train_batch_size"],
            replay_mode=dqn_config["multiagent"]["replay_mode"],
            replay_sequence_length=dqn_config["replay_sequence_length"],
        )

        # self.dqn_exploiter = DQNTFPolicy(obs_space=self.env.OBSERVATION_SPACE,
        #                                  action_space=self.env.ACTION_SPACE,
        #                                  config=dqn_config)

        def sgd_optimizer_dqn(policy, config) -> "torch.optim.Optimizer":
            return torch.optim.SGD(
                policy.q_func_vars,
                lr=policy.cur_lr,
                momentum=config["sgd_momentum"],
            )

        MyDQNTorchPolicy = DQNTorchPolicy.with_updates(
            optimizer_fn=sgd_optimizer_dqn
        )
        self.dqn_policy = MyDQNTorchPolicy(
            obs_space=env.OBSERVATION_SPACE,
            action_space=env.ACTION_SPACE,
            config=dqn_config,
        )

        self.multi_agent_batch_builders = [
            MultiAgentSampleBatchBuilder(
                policy_map={"player_blue": self.dqn_policy},
                clip_rewards=False,
                callbacks=DefaultCallbacks(),
            )
            # for _ in range(self.batch_size)
        ]

    def compute_actions(self, obs_batch):
        action, a2, a3 = self.dqn_policy.compute_actions(obs_batch=obs_batch)
        return action, a2, a3

    def add_data_in_rllib_batch_builder(
        self, s, s1P, trainBatch1, d, timestep
    ):
        if timestep <= self.stop_training_after_n_epi:
            # for i in range(self.batch_size):
            i = 0
            step_player_values = {
                "eps_id": timestep,
                "obs": s[i],
                "new_obs": s1P[i],
                "actions": trainBatch1[1][-1][i],
                "prev_actions": trainBatch1[1][-2][i]
                if len(trainBatch1[1]) > 1
                else 0,
                "rewards": trainBatch1[2][-1][i],
                "prev_rewards": trainBatch1[2][-2][i]
                if len(trainBatch1[2]) > 1
                else 0,
                "dones": d[0],
                # done is the same for for every episodes in the batch
            }
            self.multi_agent_batch_builders[i].add_values(
                agent_id="player_blue",
                policy_id="player_blue",
                **step_player_values,
            )
            self.multi_agent_batch_builders[i].count += 1

    def train_dqn_policy(self, timestep):
        stats = {"learner_stats": {}}
        if timestep <= self.stop_training_after_n_epi:
            # Add episodes in replay buffer
            # for i in range(self.batch_size):
            i = 0
            multiagent_batch = self.multi_agent_batch_builders[
                i
            ].build_and_reset()
            self.local_replay_buffer.add_batch(multiagent_batch)

            # update lr in scheduler & in optimizer
            self.dqn_policy.on_global_var_update({"timestep": timestep})
            self.dqn_policy.optimizer()
            if hasattr(self.dqn_policy, "cur_lr"):
                for opt in self.dqn_policy._optimizers:
                    for p in opt.param_groups:
                        p["lr"] = self.dqn_policy.cur_lr
            # Generate training batch and train
            for _ in range(self.train_exploiter_n_times_per_epi):
                replay_batch = self.local_replay_buffer.replay()
                if (
                    replay_batch is not None
                ):  # is None when there is not enough step in the data buffer
                    stats = self.dqn_policy.learn_on_batch(
                        replay_batch.policy_batches["player_blue"]
                    )

        stats["learner_stats"]["exploiter_lr_cur"] = self.dqn_policy.cur_lr
        for j, opt in enumerate(self.dqn_policy._optimizers):
            stats["learner_stats"]["exploiter_lr_from_params"] = [
                p["lr"] for p in opt.param_groups
            ][0]
        return stats
