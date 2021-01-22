##########
# Code modified from: https://github.com/alshedivat/lola
##########

import os
import json
import functools
from ray import tune

import tensorflow as tf
import numpy as np
from marltoolbox.algos.lola_dice.rpg import build_graph, get_update, rollout, gen_trace_batches
from marltoolbox.algos.lola_dice.policy import SimplePolicy, MLPPolicy, ConvPolicy
import marltoolbox.algos.lola_dice.envs as lola_dice_envs
import marltoolbox.algos.lola_dice.utils as utils


def make_simple_policy(ob_size, num_actions, prev=None, root=None, batch_size=None):
    return SimplePolicy(ob_size, num_actions, prev=prev)


def make_mlp_policy(ob_size, num_actions, hidden_sizes, prev=None, batch_size=64):
    return MLPPolicy(ob_size, num_actions, hidden_sizes=hidden_sizes, prev=prev, batch_size=batch_size)


def make_conv_policy(ob_size, num_actions, hidden_sizes, prev=None, batch_size=64):
    return ConvPolicy(ob_size, num_actions, hidden_sizes=hidden_sizes, prev=prev, batch_size=batch_size)


def make_adam_optimizer(lr):
    return tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False,
                                       name='Adam')

def make_sgd_optimizer(lr):
    return tf.train.GradientDescentOptimizer(learning_rate=lr)


class LOLADICE(tune.Trainable):

    def _init_lola(self, *, env, make_policy,
          make_optimizer,
          epochs,
           batch_size, trace_length, grid_size,
          gamma,
          lr_inner,
          lr_outer,
          lr_value,
          lr_om,
          inner_asymm,
          n_agents,
          n_inner_steps,
          value_batch_size,
          value_epochs,
          om_batch_size,
          om_epochs,
          use_baseline,
          use_dice,
          use_opp_modeling,
          seed,
           **kwargs):

        print("args not used:",kwargs)

        # Instantiate the environment
        if env == "IPD":
            self.env = lola_dice_envs.IPD(max_steps=trace_length, batch_size=batch_size)
        elif env == "AsymBoS":
            self.env = lola_dice_envs.AsymBoS(max_steps=trace_length, batch_size=batch_size)
        elif env == "IMP":
            self.env = lola_dice_envs.IMP(trace_length)
        elif env == "CoinGame":
            self.env = lola_dice_envs.CG(trace_length, batch_size, grid_size)
            self.env.seed(int(seed))
        elif env == "AsymCoinGame":
            self.env = lola_dice_envs.AsymCG(trace_length, batch_size, grid_size)
            self.env.seed(int(seed))
        else:
            raise ValueError(f"env: {env}")

        self.gamma = gamma
        self.lr_inner = lr_inner
        self.lr_outer = lr_outer
        self.lr_value = lr_value
        self.lr_om = lr_om
        self.inner_asymm = inner_asymm
        self.n_agents = n_agents
        self.n_inner_steps = n_inner_steps
        self.value_batch_size = value_batch_size
        self.value_epochs = value_epochs
        self.om_batch_size = om_batch_size
        self.om_epochs = om_epochs
        self.use_baseline = use_baseline
        self.use_dice = use_dice
        self.use_opp_modeling = use_opp_modeling
        self.timestep = 0

        if make_policy[0] == "make_simple_policy":
            make_policy = functools.partial(make_simple_policy, **make_policy[1])
        elif make_policy[0] == "make_conv_policy":
            make_policy = functools.partial(make_conv_policy,**make_policy[1])
        elif make_policy[0] == "make_mlp_policy":
            make_policy = functools.partial(make_mlp_policy,**make_policy[1])
        else:
            NotImplementedError()

        if make_optimizer[0] == "make_adam_optimizer":
            make_optimizer = functools.partial(make_adam_optimizer,**make_optimizer[1])
        elif make_optimizer[0] == "make_sgd_optimizer":
            make_optimizer = functools.partial(make_sgd_optimizer,**make_optimizer[1])
        else:
            NotImplementedError()

        # Build.
        graph = tf.Graph()
        with graph.as_default() as g:

            (self.policies, self.rollout_policies, pol_losses, val_losses, om_losses,
             update_pol_ops, update_val_ops, update_om_ops) = build_graph(
                self.env, make_policy, make_optimizer,
                lr_inner=lr_inner, lr_outer=lr_outer, lr_value=lr_value, lr_om=lr_om,
                n_agents=self.n_agents, n_inner_steps=n_inner_steps,
                use_baseline=use_baseline, use_dice=use_dice,
                use_opp_modeling=self.use_opp_modeling, inner_asymm=inner_asymm)

            # Train.
            self.acs_all = []
            self.rets_all = []
            self.params_all = []
            self.params_om_all = []
            self.times_all = []
            self.pick_speed_all = []

            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())

            # Construct update functions.
            self.update_funcs = {
                'policy': [
                    get_update(
                        [self.policies[k]] + self.policies[k].opponents,
                        pol_losses[k], update_pol_ops[k], self.sess,
                        gamma=self.gamma)
                    for k in range(self.n_agents)],
                'value': [
                    get_update(
                        [self.policies[k]],
                        val_losses[k], update_val_ops[k], self.sess,
                        gamma=self.gamma)
                    for k in range(self.n_agents)],
                'opp': [
                    get_update(
                        self.policies[k].root.opponents,
                        om_losses[k], update_om_ops[k], self.sess,
                        gamma=self.gamma)
                    for k in range(self.n_agents)
                ] if om_losses else None,
            }

            self.root_policies = [pi.root for pi in self.policies]

            self.saver = tf.train.Saver(max_to_keep=5)

    def setup(self, config):
        print("_init_lola", config)
        self._init_lola(**config)

    def step(self):
        self.timestep += 1

        times = []

        # Model opponents.
        if self.use_opp_modeling:
            with utils.elapsed_timer() as om_timer:
                # Fit opponent models for several epochs.
                om_losses = np.zeros((self.n_agents, self.n_agents - 1))
                for om_ep in range(self.om_epochs):
                    traces, _ = rollout(
                        self.env, self.root_policies, self.rollout_policies, self.sess,
                        gamma=self.gamma, parent_traces=[])
                    om_traces = [
                        [tr for j, tr in enumerate(traces) if j != k]
                        for k in range(self.n_agents)]
                    for k in range(self.n_agents):
                        update_om = self.update_funcs['opp'][k]
                        for trace_batch in gen_trace_batches(
                                om_traces[k], batch_size=self.om_batch_size):
                            update_om(trace_batch)
                        loss = update_om(om_traces[k])
                        om_losses[k] += np.asarray(loss)
                om_losses /= self.om_epochs
            times.append(om_timer())
        else:
            om_losses = np.array([])

        print("start Fit function")
        # Fit value functions.
        with utils.elapsed_timer() as val_timer:
            # Fit value functions for several epochs.
            value_losses = np.zeros(self.n_agents)
            for v_ep in range(self.value_epochs):
                traces, _ = rollout(
                    self.env, self.root_policies, self.rollout_policies, self.sess,
                    gamma=self.gamma, parent_traces=[])
                for k in range(self.n_agents):
                    update_val = self.update_funcs['value'][k]
                    for trace_batch in gen_trace_batches(
                            [traces[k]], batch_size=self.value_batch_size):
                        update_val(trace_batch)
                    loss = update_val([traces[k]])
                    value_losses[k] += loss[0]
                value_losses /= self.value_epochs
        times.append(val_timer())

        # # Save parameters of the agents (for debug purposes).
        # params = self.sess.run([
        #     tf.squeeze(pi.root.parameters[0])
        #     for pi in self.policies])
        # params_all.append(params)
        #
        # # Save parameters of the opponent models (for debug purposes).
        # params = [
        #     self.sess.run([
        #         tf.squeeze(opp.root.parameters[0])
        #         for opp in pi.opponents])
        #     for pi in self.policies]
        # params_om_all.append(params)

        print("start Inner loops")
        # Inner loop rollouts (lookahead steps).
        inner_all_to_log = []
        with utils.elapsed_timer() as inner_timer:
            inner_traces = []
            for k in range(self.n_agents):
                parent_traces = []
                to_log = []
                for m in range(self.n_inner_steps):
                    policies_k = [self.policies[k].parents[m]] + [
                        opp.parents[m] for opp in self.policies[k].opponents]
                    traces, sub_to_log = rollout(
                        self.env, policies_k, self.rollout_policies, self.sess,
                        gamma=self.gamma, parent_traces=parent_traces)
                    parent_traces.append(traces)
                    to_log.append(sub_to_log)
                inner_traces.append(parent_traces)
                inner_all_to_log.append(to_log)
        times.append(inner_timer())

        print("start Outer loops")
        # Outer loop rollouts (each agent plays against updated opponents).
        outer_all_to_log = []
        with utils.elapsed_timer() as outer_timer:
            outer_traces = []
            for k in range(self.n_agents):
                parent_traces = inner_traces[k]
                policies_k = [self.policies[k]] + self.policies[k].opponents
                traces, to_log = rollout(
                    self.env, policies_k, self.rollout_policies, self.sess,
                    gamma=self.gamma, parent_traces=parent_traces)
                outer_traces.append(traces)
                outer_all_to_log.append([to_log])
        times.append(outer_timer())

        # Updates.
        update_time = 0
        policy_losses = []
        for k in range(self.n_agents):
            # Policy
            with utils.elapsed_timer() as pol_upd_timer:
                parent_traces = inner_traces[k]
                update_pol = self.update_funcs['policy'][k]
                loss = update_pol(
                    outer_traces[k], parent_traces=parent_traces)
                policy_losses.append(loss)
            update_time += pol_upd_timer()

        to_report = {"episodes_total": self.timestep}
        for ag_idx in range(self.n_agents):
            print("== For ag_idx", ag_idx, "==")
            # Logging.
            if self.n_inner_steps > 0:
                # obs, acs, rets, vals, infos = list(zip(*inner_traces[0][ag_idx]))
                obs, acs, rets, vals, infos = list(zip(*inner_traces[ag_idx][0]))
                all_to_log = inner_all_to_log
            else:
                obs, acs, rets, vals, infos = list(zip(*outer_traces[ag_idx]))
                all_to_log = outer_all_to_log
            all_to_log = [per_agent_to_log[0] for per_agent_to_log in all_to_log][ag_idx]
            policy_loss = policy_losses[ag_idx]

            self.times_all.append(times)
            self.acs_all.append([ac.mean() for ac in acs])

            generate_rate_trace = [all_to_log[i].pop('generate_rate') for i in range(len(all_to_log))
                                   if "generate_rate" in all_to_log[i].keys()]
            self.pick_speed_all.append(sum(generate_rate_trace) / len(generate_rate_trace)
                                  if len(generate_rate_trace) > 0 else -1)

            self.rets_all.append([r.sum(axis=0).mean() * (1 - self.gamma) for r in rets])
            # rets_all.append([r.sum(axis=0).mean() for r in rets])
            # print("Epoch:", e + 1, '-' * 60)
            # print("Policy losses:", list(map(sum, policy_losses)))
            print("Value losses:", value_losses.tolist())
            print("OM losses:", om_losses.tolist())
            print("Returns:", self.rets_all[-1])
            print("Defection rate:", self.acs_all[-1])
            print("Pick speed:", self.pick_speed_all[-1])

            # # Save stuff
            # np.save(save_dir + '/acs.npy', acs_all)
            # np.save(save_dir + '/rets.npy', rets_all)
            # np.save(save_dir + '/params.npy', params_all)
            # np.save(save_dir + '/params_om.npy', params_om_all)
            # np.save(save_dir + '/times.npy', times_all)
            # np.save(save_dir + '/pick_speed.npy', pick_speed_all)

            info_from_env = {}
            # Only keep the last info
            for to_log in all_to_log:
                info_from_env.update(to_log)

            initial_info = {
                "returns_player_1": self.rets_all[-1][0],
                "returns_player_2": self.rets_all[-1][1],
                "defection_rate_player_1": self.acs_all[-1][0],
                "defection_rate_player_2": self.acs_all[-1][1],
                "pick_speed_global": self.pick_speed_all[-1],
                "policy_loss": policy_loss,
            }
            for k, v in info_from_env.items():
                to_report[f"ag_{ag_idx}_{k}"] = v
            for k, v in initial_info.items():
                to_report[f"ag_{ag_idx}_{k}"] = v

        return to_report

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
        print('Loading Model...',checkpoint_path)
        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)

        ckpt = tf.train.get_checkpoint_state(checkpoint["tf_checkpoint_dir"],
                                             latest_filename=f'{checkpoint["tf_checkpoint_filename"]}')
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)

        # TODO need to set in eval like batchnorm fixed

    def cleanup(self):
        self.sess.close()
        super().cleanup()

    def _get_agent_to_use(self, policy_id):
        if policy_id == "player_red":
            agent_n = 0
            available_actions = np.ones(shape=[1,4])
        elif policy_id == "player_blue":
            agent_n = 1
            available_actions = np.ones(shape=[1,4])
        elif policy_id == "player_row":
            agent_n = 0
            available_actions = np.ones(shape=[1,2])
        elif policy_id == "player_col":
            agent_n = 1
            available_actions = np.ones(shape=[1,2])
        else:
            raise ValueError(f"policy_id {policy_id}")
        info = {"available_actions": available_actions}
        return agent_n, info

    def _preprocess_obs(self, single_obs):
        single_obs = single_obs[None, ...]  # add batch dim
        return single_obs

    def _post_process_action(self, action):
        return action[None, ...]  # add batch dim


    def compute_actions(self, policy_id:str, obs_batch:list):
        # because of the LSTM
        assert len(obs_batch) == 1

        for single_obs in obs_batch:
            agent_to_use, info = self._get_agent_to_use(policy_id)
            obs = self._preprocess_obs(single_obs)

            a = self.policies[agent_to_use].act(obs, info, sess=self.sess)
        action = self._post_process_action(a)

        state_out = []
        extra_fetches = {}
        return action, state_out, extra_fetches

