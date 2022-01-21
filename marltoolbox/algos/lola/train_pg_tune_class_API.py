##########
# Code modified from: https://github.com/alshedivat/lola
##########
import json
import os

from ray import tune

from marltoolbox.algos.lola.corrections import *
from marltoolbox.algos.lola.networks import *
from marltoolbox.algos.lola.utils import *


def update(mainQN, lr, final_delta_1_v, final_delta_2_v):
    update_theta_1 = mainQN[0].setparams(
        mainQN[0].getparams() + lr * np.squeeze(final_delta_1_v)
    )
    update_theta_2 = mainQN[1].setparams(
        mainQN[1].getparams() + lr * np.squeeze(final_delta_2_v)
    )


class LOLAPGMatrice(tune.Trainable):
    def _init_lola(
        self,
        env_class,
        seed,
        num_episodes,
        trace_length,
        batch_size,
        lola_update,
        gamma,
        hidden,
        lr,
        mem_efficient=True,
        summary_len=20,
        use_toolbox_env=False,
        clip_lola_update_norm=False,
        lola_correction_multiplier=1.0,
        clip_lola_correction_norm=False,
        clip_lola_actor_norm=False,
        set_zero=0,
        simple_net=True,
        **kwargs,
    ):

        print("args not used:", kwargs)

        corrections = lola_update

        # Instantiate the environment
        self.env = env_class(
            config={
                "max_steps": trace_length,
                "get_additional_info": True,
            }
        )
        self.env.seed(seed)

        # observation_space = self.env.NUM_STATES
        self.y = gamma
        self.gamma = gamma
        self.load_model = False  # Whether to load a saved model.
        self.n_agents = self.env.NUM_AGENTS
        self.total_n_agents = self.n_agents
        self.max_epLength = (
            trace_length + 1
        )  # The max allowed length of our episode.
        self.summaryLength = (
            summary_len  # Number of epidoes to periodically save for analysis
        )
        self.batch_size = batch_size
        self.use_toolbox_env = use_toolbox_env
        self.simple_net = simple_net
        self.trace_length = trace_length
        self.lr = lr
        self.num_episodes = num_episodes

        # tf.reset_default_graph()
        graph = tf.Graph()

        with graph.as_default() as g:
            self.sess = tf.Session()

            self.mainQN = []

            self.agent_list = np.arange(self.total_n_agents)
            for agent in range(self.total_n_agents):
                self.mainQN.append(
                    Qnetwork(
                        "main" + str(agent),
                        agent,
                        self.env,
                        lr=lr,
                        gamma=gamma,
                        batch_size=batch_size,
                        trace_length=self.trace_length,
                        hidden=hidden,
                        simple_net=simple_net,
                        sess=self.sess,
                    )
                )

            if not mem_efficient:
                self.cube, self.cube_ops = make_cube(self.trace_length)
            else:
                self.cube, self.cube_ops = None, None

            corrections_func(
                self.mainQN,
                batch_size=batch_size,
                trace_length=self.trace_length,
                corrections=corrections,
                cube=self.cube,
            )

            init = tf.global_variables_initializer()
            self.trainables = tf.trainable_variables()

            self.buffers = []
            for i in range(self.total_n_agents):
                self.buffers.append(ExperienceBuffer(batch_size))

            # create lists to contain total rewards and steps per episode
            self.jList = []
            self.rList = []
            self.aList = []

            self.total_steps = 0

            self.episodes_run = np.zeros(self.total_n_agents)
            self.episodes_run_counter = np.zeros(self.total_n_agents)
            self.episodes_reward = np.zeros(self.total_n_agents)
            self.episodes_actions = np.zeros(
                (self.total_n_agents, self.env.NUM_ACTIONS)
            )
            # need to multiple with
            pow_series = np.arange(self.trace_length)
            self.discount = np.array([pow(gamma, item) for item in pow_series])
            self.discount_array = gamma ** self.trace_length / self.discount
            # print('discount_array',discount_array.shape)
            self.discount = np.expand_dims(self.discount, 0)
            self.discount_array = np.reshape(self.discount_array, [1, -1])

            array = np.eye(self.env.NUM_STATES)
            self.feed_dict_log_pi = {
                self.mainQN[0].scalarInput: array,
                self.mainQN[1].scalarInput: array,
            }

            # with tf.Session() as sess:
            self.sess.run(init)
            if self.cube_ops is not None:
                self.sess.run(self.cube_ops)

            if set_zero == 1:
                for i in range(2):
                    self.mainQN[i].setparams(np.zeros((5)))
                    theta_2_vals = self.mainQN[i].getparams()

            # sP = env.reset()
            self.updated = True
            self.timestep = 0
            self.saver = tf.train.Saver(max_to_keep=5)

            # self.reset_compute_actions_state()

    def setup(self, config):
        print("_init_lola", config)
        self._init_lola(**config)

    def step(self):
        self.timestep += 1

        last_info = {}
        log_items = {}

        episodeBuffer = []
        for ii in range(self.n_agents):
            episodeBuffer.append([])
        np.random.shuffle(self.agent_list)
        if self.n_agents == self.total_n_agents:
            these_agents = range(self.n_agents)
        else:
            these_agents = sorted(self.agent_list[0 : self.n_agents])

        for epi_n in range(self.batch_size):
            # Reset environment and get first new observation
            if not self.use_toolbox_env:
                raise NotImplementedError()
                # using coin game from lola.envs
                # Reset environment and get first new observation
                # sP = env.reset()
                # using coin game from lola_dice.envs
                sP = self.env.reset()
                s = sP
                state = []
            else:
                sP = self.env.reset()
                state = np.zeros(self.env.NUM_STATES)
                state[sP["player_row"]] = 1
                sP = [state, state]
                s = sP
                state = []

            d = False
            rAll = np.zeros((self.n_agents))
            aAll = np.zeros((self.env.NUM_STATES))
            j = 0

            for agent in these_agents:
                self.episodes_run[agent] += 1
                self.episodes_run_counter[agent] += 1
            a_all_old = [0, 0]

            # Rollout the episode
            # ac with self.mainQN[agent].predict andenv.step
            # The Q-Network
            while j < self.max_epLength:
                j += 1
                a_all = []
                for agent_role, agent in enumerate(these_agents):
                    a = self.sess.run(
                        [self.mainQN[agent].predict],
                        feed_dict={
                            self.mainQN[agent].scalarInput: [s[agent_role]]
                        },
                    )
                    a_all.append(a[0])

                if not self.use_toolbox_env:
                    raise NotImplementedError()
                    a_all_old = a_all
                    if a_all[0] > 1 or a_all[1] > 1:
                        print("warning!!!", a_all, "s", s)
                    s1P, r, d = self.env.step(a_all)
                    s1 = s1P
                else:
                    a_all_old = a_all
                    actions = {"player_row": a_all[0], "player_col": a_all[1]}
                    obs, r, d, info = self.env.step(actions)
                    d = d["__all__"]
                    state = np.zeros(self.env.NUM_STATES)
                    state[obs["player_row"]] = 1
                    s1P = [state, state]
                    s1 = s1P
                    if "player_row" in info.keys():
                        last_info.update(
                            {
                                f"player_row_{k}": v
                                for k, v in info["player_row"].items()
                            }
                        )
                    if "player_col" in info.keys():
                        last_info.update(
                            {
                                f"player_col_{k}": v
                                for k, v in info["player_col"].items()
                            }
                        )
                    r = [r["player_row"], r["player_col"]]

                self.total_steps += 1
                for agent_role, agent in enumerate(these_agents):
                    episodeBuffer[agent_role].append(
                        [
                            s[0],
                            a_all[agent_role],
                            r[agent_role],
                            s1[0],
                            d,
                            these_agents[agent_role],
                        ]
                    )
                    self.episodes_reward[agent] += r[agent_role]
                rAll += [r[ii] * self.gamma ** (j - 1) for ii in range(2)]

                aAll[a_all[0]] += 1
                aAll[a_all[1] + 2] += 1
                s_old = s
                s = s1
                sP = s1P
                if d == True:
                    break

            # Add the episode to the experience buffer
            for agent_role, agent in enumerate(these_agents):
                self.buffers[agent].add(np.array(episodeBuffer[agent_role]))

            self.jList.append(j)
            self.rList.append(rAll)
            self.aList.append(aAll)

        # training every batch_size episodes
        if (
            self.episodes_run[agent] % self.batch_size == 0
            and self.episodes_run[agent] > 0
        ):

            # Sample from data buffer
            trainBatch0 = self.buffers[0].sample(
                self.batch_size, self.trace_length
            )  # Get a random batch of experiences.
            trainBatch1 = self.buffers[1].sample(
                self.batch_size, self.trace_length
            )

            sample_return0 = np.reshape(
                get_monte_carlo(
                    trainBatch0[:, 2],
                    self.y,
                    self.trace_length,
                    self.batch_size,
                ),
                [self.batch_size, -1],
            )
            sample_return1 = np.reshape(
                get_monte_carlo(
                    trainBatch1[:, 2],
                    self.y,
                    self.trace_length,
                    self.batch_size,
                ),
                [self.batch_size, -1],
            )

            sample_reward0 = (
                np.reshape(
                    trainBatch0[:, 2] - np.mean(trainBatch0[:, 2]),
                    [-1, self.trace_length],
                )
                * self.discount
            )
            sample_reward1 = (
                np.reshape(
                    trainBatch1[:, 2] - np.mean(trainBatch1[:, 2]),
                    [-1, self.trace_length],
                )
                * self.discount
            )

            last_state = np.reshape(
                np.vstack(trainBatch0[:, 3]),
                [-1, self.trace_length, self.env.NUM_STATES],
            )[:, -1, :]

            value_0_next, value_1_next = self.sess.run(
                [self.mainQN[0].value, self.mainQN[1].value],
                feed_dict={
                    self.mainQN[0].scalarInput: last_state,
                    self.mainQN[1].scalarInput: last_state,
                },
            )

            fetches = [
                self.mainQN[0].values,
                self.mainQN[0].updateModel,
                self.mainQN[1].updateModel,
                self.mainQN[0].delta,
                self.mainQN[1].delta,
                self.mainQN[0].grad,
                self.mainQN[1].grad,
                self.mainQN[0].v_0_grad_01,
                self.mainQN[1].v_1_grad_10,
            ]
            feed_dict = {
                self.mainQN[0].scalarInput: np.vstack(trainBatch0[:, 0]),
                self.mainQN[0].sample_return: sample_return0,
                self.mainQN[0].actions: trainBatch0[:, 1],
                self.mainQN[1].scalarInput: np.vstack(trainBatch1[:, 0]),
                self.mainQN[1].sample_return: sample_return1,
                self.mainQN[1].actions: trainBatch1[:, 1],
                self.mainQN[0].sample_reward: sample_reward0,
                self.mainQN[1].sample_reward: sample_reward1,
                self.mainQN[0].next_value: value_0_next,
                self.mainQN[1].next_value: value_1_next,
                self.mainQN[0].gamma_array: self.discount,
                self.mainQN[1].gamma_array: self.discount,
                self.mainQN[0].gamma_array_inverse: self.discount_array,
                self.mainQN[1].gamma_array_inverse: self.discount_array,
            }
            # Condition not needed?
            if (
                self.episodes_run[agent] % self.batch_size == 0
                and self.episodes_run[agent] > 0
            ):
                (
                    values,
                    _,
                    _,
                    update1,
                    update2,
                    grad_1,
                    grad_2,
                    v0_grad_01,
                    v1_grad_10,
                ) = self.sess.run(fetches, feed_dict=feed_dict)

            # Condition not needed?
            if (
                self.episodes_run[agent] % self.batch_size == 0
                and self.episodes_run[agent] > 0
            ):
                update(self.mainQN, self.lr, update1, update2)
                self.updated = True
                print("update params")
                print("grad_1", grad_1)
                print("grad_2", grad_2)
                print("v0_grad_01", v0_grad_01)
                print("v1_grad_10", v1_grad_10)
                print("values", values)
            self.episodes_run_counter[agent] = (
                self.episodes_run_counter[agent] * 0
            )
            self.episodes_actions[agent] = self.episodes_actions[agent] * 0
            self.episodes_reward[agent] = self.episodes_reward[agent] * 0

        # Log info every summaryLength episodes
        if (
            len(self.rList) % self.summaryLength == 0
            and len(self.rList) != 0
            and self.updated == True
        ):
            self.updated = False
            gamma_discount = 1 / (1 - self.gamma)
            print(
                self.total_steps,
                "reward",
                np.mean(self.rList[-self.summaryLength :], 0) / gamma_discount,
                "action",
                (
                    np.mean(self.aList[-self.summaryLength :], 0)
                    * 2.0
                    / np.sum(np.mean(self.aList[-self.summaryLength :], 0))
                )
                * 100
                // 1,
            )

            action_prob = (
                np.mean(self.aList[-self.summaryLength :], 0)
                * 2.0
                / np.sum(np.mean(self.aList[-self.summaryLength :], 0))
            )
            log_items["reward_agent0"] = np.mean(
                self.rList[-self.summaryLength :], 0
            )[0]
            log_items["reward_agent1"] = np.mean(
                self.rList[-self.summaryLength :], 0
            )[1]
            log_items["agent0_C"] = action_prob[0]
            log_items["agent0_D"] = action_prob[1]
            log_items["agent1_C"] = action_prob[2]
            log_items["agent1_D"] = action_prob[3]
            if self.simple_net:
                theta_1_vals = self.mainQN[0].getparams()
                theta_2_vals = self.mainQN[1].getparams()
                print("theta_1_vals", theta_1_vals)
                print("theta_2_vals", theta_2_vals)

                log_items["theta_1_0"] = theta_1_vals[0]
                log_items["theta_1_1"] = theta_1_vals[1]
                log_items["theta_1_2"] = theta_1_vals[2]
                log_items["theta_1_3"] = theta_1_vals[3]
                log_items["theta_1_4"] = theta_1_vals[4]
                log_items["theta_2_0"] = theta_2_vals[0]
                log_items["theta_2_1"] = theta_2_vals[1]
                log_items["theta_2_2"] = theta_2_vals[2]
                log_items["theta_2_3"] = theta_2_vals[3]
                log_items["theta_2_4"] = theta_2_vals[4]
            else:
                log_pi0, log_pi1 = self.sess.run(
                    [self.mainQN[0].log_pi, self.mainQN[1].log_pi],
                    feed_dict=self.feed_dict_log_pi,
                )
                print("pi 0", np.exp(log_pi0))
                print("pi 1", np.exp(log_pi1))

                log_items["pi_1_0"] = np.exp(log_pi0[0][0])
                log_items["pi_1_1"] = np.exp(log_pi0[1][0])
                log_items["pi_1_2"] = np.exp(log_pi0[2][0])
                log_items["pi_1_3"] = np.exp(log_pi0[3][0])
                log_items["pi_1_4"] = np.exp(log_pi0[4][0])

                log_items["pi_2_0"] = np.exp(log_pi1[0][0])
                log_items["pi_2_1"] = np.exp(log_pi1[1][0])
                log_items["pi_2_2"] = np.exp(log_pi1[2][0])
                log_items["pi_2_3"] = np.exp(log_pi1[3][0])
                log_items["pi_2_4"] = np.exp(log_pi1[4][0])

        last_info.pop("available_actions", None)
        log_items.update(last_info)
        log_items["episodes_total"] = self.timestep
        log_items["finished"] = (
            False if self.timestep < self.num_episodes else True
        )
        return log_items

    def save_checkpoint(self, checkpoint_dir):
        path = os.path.join(checkpoint_dir, "checkpoint.json")
        tf_checkpoint_path = os.path.join(checkpoint_dir, "checkpoint")
        tf_checkpoint_dir, tf_checkpoint_filename = os.path.split(
            tf_checkpoint_path
        )
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
        print("Loading Model...", checkpoint_path)
        with open(checkpoint_path, "r") as f:
            checkpoint = json.load(f)

        ckpt = tf.train.get_checkpoint_state(
            checkpoint["tf_checkpoint_dir"],
            latest_filename=f'{checkpoint["tf_checkpoint_filename"]}',
        )
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)

    def cleanup(self):
        self.sess.close()
        super().cleanup()

    def _get_agent_to_use(self, policy_id):
        if policy_id == "player_row":
            agent_n = 0
        elif policy_id == "player_col":
            agent_n = 1
        else:
            raise ValueError(f"policy_id {policy_id}")
        return agent_n

    def _preprocess_obs(self, single_obs):
        return single_obs

    def _post_process_action(self, action):
        print("action", action)
        action = np.array(action)
        return action

    def compute_actions(self, policy_id: str, obs_batch: list):
        assert len(obs_batch) == 1

        for single_obs in obs_batch:
            agent_to_use = self._get_agent_to_use(policy_id)
            obs = self._preprocess_obs(single_obs)

            obs = np.expand_dims(obs, axis=0)
            a = self.sess.run(
                [self.mainQN[agent_to_use].predict],
                # feed_dict={self.mainQN[agent_to_use].scalarInput: [obs]},
                feed_dict={self.mainQN[agent_to_use].scalarInput: obs},
            )
        action = self._post_process_action(a)

        state_out = []
        extra_fetches = {}
        return action, state_out, extra_fetches
