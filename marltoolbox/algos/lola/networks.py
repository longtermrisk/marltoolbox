"""
Policy and value networks used in LOLA experiments.
"""
import copy
import tensorflow as tf
import tensorflow.contrib.layers as layers
from tensorflow.python.ops import math_ops
from functools import partial
from marltoolbox.algos.lola.utils import *


class Pnetwork:
    """
    Recurrent policy network used in Coin Game experiments.
    """
    def __init__(self, myScope, h_size, agent, env, trace_length, batch_size,
                 reuse=None, step=False, changed_config= False, ac_lr=1.0, use_MAE=False,
                 use_toolbox_env=False, clip_loss_norm=False, sess=None, entropy_coeff=1.0,
                 weigth_decay=0.01, use_critic=False):
        self.sess = sess

        if use_toolbox_env:
            ob_space_shape = list(env.OBSERVATION_SPACE.shape)
        else:
            ob_space_shape = env.ob_space_shape
        print("ob_space_shape", ob_space_shape)

        if step:
            trace_length = 1
        else:
            trace_length = trace_length
        with tf.variable_scope(myScope, reuse=reuse):
            self.batch_size = batch_size
            zero_state = tf.zeros((batch_size, h_size * 2), dtype=tf.float32)
            self.gamma_array = tf.placeholder(
                shape=[1, trace_length],
                dtype=tf.float32,
                name='gamma_array')
            self.gamma_array_inverse = tf.placeholder(
                shape=[1, trace_length],
                dtype=tf.float32,
                name='gamma_array_inv')

            self.lstm_state = tf.placeholder(
                shape=[batch_size, h_size*2], dtype=tf.float32,
                name='lstm_state')

            if step:
                self.state_input =  tf.placeholder(
                    shape=[self.batch_size] + ob_space_shape,
                    dtype=tf.float32,
                    name='state_input')
                lstm_state = self.lstm_state
            else:
                self.state_input =  tf.placeholder(
                    shape=[batch_size * trace_length] + ob_space_shape,
                    dtype=tf.float32,
                    name='state_input')
                lstm_state = zero_state

            self.sample_return = tf.placeholder(
                shape=[None, trace_length],
                dtype=tf.float32,
                name='sample_return')
            self.sample_reward = tf.placeholder(
                shape=[None, trace_length],
                dtype=tf.float32,
                name='sample_reward')
            self.sample_reward_bis = tf.placeholder(
                shape=[None, trace_length],
                dtype=tf.float32,
                name='sample_reward')

            self.j = tf.placeholder(
                shape=[None],
                dtype=tf.float32,
                name='j')

            self.loss_multiplier = tf.placeholder(
                shape=[1],
                dtype=tf.float32,
                name='loss_multiplier')

            self.is_training = tf.placeholder(tf.bool, name='is_training')

            with tf.variable_scope('input_proc', reuse=reuse):
                # if not changed_config:
                output = layers.convolution2d(self.state_input,
                    stride=1, kernel_size=3, num_outputs=20,
                    normalizer_fn=layers.batch_norm, activation_fn=tf.nn.relu)
                output = layers.convolution2d(output,
                    stride=1, kernel_size=3, num_outputs=20,
                    normalizer_fn=layers.batch_norm, activation_fn=tf.nn.relu)
                # output = layers.convolution2d(self.state_input,
                #     stride=1, kernel_size=3, num_outputs=20,
                #     normalizer_fn=partial(layers.batch_norm,
                #                           is_training=self.is_training,
                #                           # scale=True,
                #                           # updates_collections=None,
                #                           variables_collections=["batch_norm_non_trainable_variables_collection"]),
                #                               activation_fn=tf.nn.relu)
                # output = layers.convolution2d(output,
                #     stride=1, kernel_size=3, num_outputs=20,
                #     normalizer_fn=partial(layers.batch_norm,
                #                           is_training=self.is_training,
                #                           # scale=True,
                #                           # updates_collections=None,
                #                           variables_collections=["batch_norm_non_trainable_variables_collection"]),
                #                               activation_fn=tf.nn.relu)
                # output = layers.convolution2d(self.state_input,
                #     stride=1, kernel_size=3, num_outputs=20,
                #                               normalizer_fn = None, activation_fn=None)
                #     # normalizer_fn=partial(tf.layers.batch_normalization, training=self.is_training),
                #     #                           activation_fn=tf.nn.relu)
                # output = tf.layers.batch_normalization(output, training=self.is_training)
                # output = tf.nn.relu(output)
                # output = layers.convolution2d(output,
                #     stride=1, kernel_size=3, num_outputs=20,
                #                               normalizer_fn=None, activation_fn=None)
                # # normalizer_fn=partial(tf.layers.batch_normalization, training=self.is_training),
                # #                               activation_fn=tf.nn.relu)
                # output = tf.layers.batch_normalization(output, training=self.is_training)
                # output = tf.nn.relu(output)

                output = layers.flatten(output)

                # if step:
                #     position_in_epi = self.j
                # else:
                #     position_in_epi = tf.range(start=0, limit=trace_length, delta=1)
                # position_in_epi = tf.cast(position_in_epi, tf.float32)
                # position_in_epi = tf.reshape(position_in_epi, [-1, 1])
                # position_in_epi = tf.concat([position_in_epi for i in range(self.batch_size)], axis=0)
                # print_op_23 = tf.print("position_in_epi", tf.shape(position_in_epi))
                # print_op_24 = tf.print("position_in_epi", position_in_epi)
                # with tf.control_dependencies(
                #         [print_op_23, print_op_24]):
                #     output_for_value = tf.concat([output, position_in_epi], 1)
                # print_op_23 = tf.print("output_for_value", tf.shape(output_for_value))
                # print_op_24 = tf.print("output_for_value", output_for_value)
                # with tf.control_dependencies(
                #         [print_op_23, print_op_24]):
                #     interm = layers.fully_connected(tf.nn.relu(output_for_value), 1, activation_fn=None)
                # print("interm", interm.shape)
                # interm_splits = tf.split(interm, [ self.batch_size for i in range(trace_length)], 0)
                # print_op_23 = tf.print("interm_splits", tf.shape(interm_splits))
                # print_op_24 = tf.print("interm_splits", interm_splits)
                # with tf.control_dependencies(
                #         [print_op_23, print_op_24]):
                #     self.value = tf.reshape(tf.stack(interm_splits, axis=1), [-1, trace_length])
                # print("self.value", self.value.shape)

                # output_temp = tf.stop_gradient(output)

                # With a corrected Critic
                # self.value = tf.reshape(layers.fully_connected(
                #     tf.nn.relu(output), 1, activation_fn=None), [-1, trace_length])

                # Without a Critic
                if use_critic:
                    self.value = tf.reshape(layers.fully_connected(
                        tf.nn.relu(output), 1), [-1, trace_length])
                else:
                    self.value = tf.reshape(layers.fully_connected(
                        tf.nn.relu(output), 1), [-1, trace_length]) * 0.0

                # Original:
                # self.value = tf.reshape(layers.fully_connected(
                #     tf.nn.relu(output), 1), [-1, trace_length]) * 0.0

            if step:
                output_seq = batch_to_seq(output, self.batch_size, 1)
            else:
                output_seq = batch_to_seq(output, self.batch_size, trace_length)
            output_seq, state_output = lstm(output_seq, lstm_state,
                                            scope='rnn', nh=h_size)
            output = seq_to_batch(output_seq)

            output = layers.fully_connected(output,
                                            num_outputs=env.NUM_ACTIONS,
                                            activation_fn=None)
            self.log_pi = tf.nn.log_softmax(output)
            self.lstm_state_output = state_output

            entropy_temp = self.log_pi * tf.exp(self.log_pi)
            self.entropy = tf.reduce_mean(entropy_temp) * 4

            self.actions = tf.placeholder(
                shape=[None], dtype=tf.int32, name='actions')
            self.actions_onehot = tf.one_hot(
                self.actions, env.NUM_ACTIONS, dtype=tf.float32)

            predict = tf.multinomial(self.log_pi, 1)
            self.predict = tf.squeeze(predict)

            self.next_value = tf.placeholder(
                shape=[None,1], dtype=tf.float32, name='next_value')
            self.next_v = tf.matmul(self.next_value, self.gamma_array_inverse)
            if use_critic:
                self.target = self.sample_reward_bis + self.next_v
            else:
                self.target = self.sample_return + self.next_v
            # self.target = self.sample_return #+ self.next_v
            # self.target = self.sample_reward_bis + self.next_v

            if not use_MAE:
                self.td_error = tf.square(self.target-self.value) / 2
            else:
                self.td_error = tf.abs(self.target-self.value) / 2

            if use_critic:
                self.loss = tf.reduce_mean(self.td_error)
            else:
                self.loss = tf.reduce_mean(self.td_error) * 0.0

        self.parameters = []
        self.value_params = []
        for i in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                   scope=myScope):
            if not ('value_params' in i.name):
                self.parameters.append(i)  # i.name if you want just a name
            if 'input_proc' in i.name:
                self.value_params.append(i)
        print("myScope", myScope)
        print("self.parameters", self.parameters)
        print("self.value_params", self.value_params)
        self.parameters_norm = tf.reduce_sum([tf.reduce_sum(p*p) for p in self.parameters])
        self.value_params_norm = tf.reduce_sum([tf.reduce_sum(p*p) for p in self.value_params])

        if self.sess is not None:
            self.getparams = GetFlatWtSess(self.parameters, self.sess)
            self.setparams = SetFromFlatWtSess(self.parameters, self.sess)
        else:
            self.getparams = GetFlat(self.parameters)
            self.setparams = SetFromFlat(self.parameters)

        if not step:
            self.log_pi_action = tf.reduce_mean(tf.multiply(
                self.log_pi, self.actions_onehot), reduction_indices=1)
            self.log_pi_action_bs = tf.reduce_sum(tf.reshape(
                self.log_pi_action, [-1, trace_length]),1)
            self.log_pi_action_bs_t = tf.reshape(
                self.log_pi_action, [self.batch_size, trace_length])
            # self.trainer = tf.train.GradientDescentOptimizer(learning_rate=1)
            self.trainer = tf.train.GradientDescentOptimizer(learning_rate=ac_lr)

            weigths_norm = tf.concat(axis=0, values=[tf.reshape(v, [numel(v)]) for v in self.parameters])
            weigths_norm = math_ops.reduce_sum(weigths_norm * weigths_norm, None, keepdims=True)
            weigths_norm = tf.reduce_sum(weigths_norm)
            self.weigths_norm = tf.sqrt(weigths_norm)


            if clip_loss_norm:
                l2sum = math_ops.reduce_sum(self.loss * self.loss, None, keepdims=True)
                print_op_1 = tf.print("loss l2sum", l2sum)
                with tf.control_dependencies([print_op_1]):
                    self.loss = tf.clip_by_norm(self.loss, clip_loss_norm, axes=None, name=None)
            if use_critic:
                total_loss = ((self.loss + entropy_coeff*self.entropy + weigth_decay*self.weigths_norm) *
                             self.loss_multiplier)
            else:
                total_loss = (entropy_coeff * self.entropy + weigth_decay * self.weigths_norm) * self.loss_multiplier
            self.updateModel = self.trainer.minimize(
                total_loss, var_list=self.value_params)
        # self.setparams= SetFromFlat(self.parameters)
        # self.getparams= GetFlat(self.parameters)

        self.param_len = len(self.parameters)

        for var in self.parameters:
            print(var.name, var.get_shape())


class Qnetwork:
    """
    Simple Q-network used in IPD experiments.
    """
    def __init__(self, myScope, agent, env, batch_size, gamma, trace_length, hidden, simple_net, lr, sess):

        self.sess = sess


        #The network recieves a frame from the game, flattened into an array.
        #It then resizes it and processes it through four convolutional layers.
        with tf.variable_scope(myScope):
            self.scalarInput =  tf.placeholder(shape=[None, env.NUM_STATES],dtype=tf.float32)
            self.gamma_array = tf.placeholder(shape=[1, trace_length], dtype=tf.float32, name='gamma_array')
            self.gamma_array_inverse = tf.placeholder(shape=[1, trace_length], dtype=tf.float32, name='gamma_array')

            if simple_net:
                self.logit_vals = tf.Variable(tf.random_normal([5,1]))
                self.temp = tf.matmul(self.scalarInput, self.logit_vals)
                temp_concat = tf.concat([self.temp, self.temp * 0], 1)
                self.log_pi = tf.nn.log_softmax(temp_concat)
            else:
                act = tf.nn.leaky_relu(layers.fully_connected(self.scalarInput, num_outputs=hidden, activation_fn=None))
                self.log_pi = tf.nn.log_softmax(layers.fully_connected(act, num_outputs=2, activation_fn=None))
            self.values = tf.Variable(tf.random_normal([5,1]), name='value_params')
            self.value = tf.reshape(tf.matmul(self.scalarInput, self.values), [batch_size, -1])
            self.sample_return = tf.placeholder(shape=[None, trace_length],dtype=tf.float32, name='sample_return')
            self.sample_reward = tf.placeholder(shape=[None, trace_length], dtype=tf.float32, name='sample_reward_new')

            self.next_value = tf.placeholder(
                shape=[None, 1], dtype=tf.float32, name='next_value')
            self.next_v = tf.matmul(self.next_value, self.gamma_array_inverse)
            self.target = self.sample_return + self.next_v
            self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
            self.actions_onehot = tf.one_hot(
                self.actions, env.NUM_ACTIONS, dtype=tf.float32)

            self.predict = tf.multinomial(self.log_pi ,1)
            self.predict = tf.squeeze(self.predict)
            self.log_pi_action = tf.reduce_mean(
                tf.multiply(self.log_pi, self.actions_onehot),
                reduction_indices=1)

            self.td_error = tf.square(self.target - self.value) / 2
            self.loss = tf.reduce_mean(self.td_error)

        self.parameters = []
        self.value_params = []
        for i in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                   scope=myScope):
            if not ('value_params' in i.name):
                self.parameters.append(i)   # i.name if you want just a name
            else:
                self.value_params.append(i)

        self.trainer = tf.train.GradientDescentOptimizer(learning_rate=1)# / arglist.bs)
        self.updateModel = self.trainer.minimize(self.loss, var_list=self.value_params)

        self.log_pi_action_bs = tf.reduce_sum(tf.reshape(self.log_pi_action, [-1, trace_length]),1)
        self.log_pi_action_bs_t = tf.reshape(self.log_pi_action, [batch_size, trace_length])
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

    def add(self,experience):
        if len(self.buffer) + 1 >= self.buffer_size:
            self.buffer[0:(1+len(self.buffer))-self.buffer_size] = []
        self.buffer.append(experience)

    def sample(self,batch_size,trace_length):
        sampled_episodes = self.buffer
        sampledTraces = []
        for episode in sampled_episodes:
            this_episode = list(copy.deepcopy(episode))
            point = np.random.randint(0,len(this_episode)+1-trace_length)
            sampledTraces.append(this_episode[point:point+trace_length])
        sampledTraces = np.array(sampledTraces)
        return np.reshape(sampledTraces,[batch_size*trace_length,6])
