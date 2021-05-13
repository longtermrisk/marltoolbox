#############################################
# Code modified from ray.rllib.rollout.py
# Code modified from ray.rllib.agent.trainer.py
#############################################

import collections
import copy
import logging
from typing import List

from gym import wrappers as gym_wrappers
from ray.rllib.env import MultiAgentEnv
from ray.rllib.env.base_env import _DUMMY_AGENT_ID
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.rollout import (
    DefaultMapping,
    default_policy_agent_mapping,
    RolloutSaver,
)
from ray.rllib.utils.framework import TensorStructType
from ray.rllib.utils.spaces.space_utils import flatten_to_single_ndarray
from ray.rllib.utils.typing import EnvInfoDict, PolicyID

logger = logging.getLogger(__name__)


class RolloutManager(RolloutSaver):
    """
    Modified version of the utility class for storing rollouts.
    Modified to allow to store data even if no output file is provided.
    """

    def end_rollout(self):
        if self._use_shelve:
            # Save this episode as a new entry in the shelf database,
            # using the episode number as the key.
            self._shelf[str(self._num_episodes)] = self._current_rollout
        else:
            # Append this rollout to our list, to save laer.
            self._rollouts.append(self._current_rollout)
        # Even if the episode is not completely finished
        self._num_episodes += 1
        if self._update_file:
            self._update_file.seek(0)
            self._update_file.write(self._get_progress() + "\n")
            self._update_file.flush()

    def append_step(self, obs, action, next_obs, reward, done, info):
        """Add a step to the current rollout, if we are saving them"""
        if self._save_info:
            self._current_rollout.append(
                [obs, action, next_obs, reward, done, info]
            )
        else:
            self._current_rollout.append([obs, action, next_obs, reward, done])
        self._total_steps += 1


def internal_rollout(
    worker,
    num_steps,
    policy_map=None,
    policy_agent_mapping=None,
    reset_env_before=True,
    num_episodes=0,
    last_obs=None,
    saver=None,
    no_render=True,
    video_dir=None,
    seed=None,
    explore=None,
    last_rnn_states=None,
    base_env=None,
):
    """
    Can perform rollouts on the environment from inside a worker_rollout or
    from a policy. Can perform rollouts during the evaluation rollouts ran
    from an RLLib Trainer.

    :param worker: worker from an RLLib Trainer.
    The interal rollouts will be run inside this worker, using its policies.
    :param num_steps: number of maximum steps to perform in total
    :param policy_map: (optional) by default the policy_map of the provided
    worker will be used
    :param policy_agent_mapping: (optional) by default the policy_mapping_fn
    of the provided worker will be used
    :param reset_env_before: (optional) reset the environment from the worker
    before first using it
    :param num_episodes: (optional) number of maximum episodes to perform
    :param last_obs: (optional) if reset_env_before is False then you must
    provide the last observation
    :param saver: (optional) an instance of a RolloutManager
    :param no_render: (optional) option to call env.render()
    :param video_dir: (optional)
    :param seed: (optional) random seed to set for the environment by calling
    env.seed(seed)
    :param last_rnn_states: map of policy_id to rnn_states
    :return: an instance of a RolloutManager, which contains the data about
    the rollouts performed
    """

    assert num_steps is not None or num_episodes is not None
    assert reset_env_before or last_obs is not None

    if saver is None:
        saver = RolloutManager()

    if base_env is None:
        env = worker.env
    else:
        env = base_env.get_unwrapped()[0]

    # if hasattr(env, "seed") and callable(env.seed):
    #     env.seed(seed)

    env = copy.deepcopy(env)
    multiagent = isinstance(env, MultiAgentEnv)
    if policy_agent_mapping is None:
        if worker.multiagent:
            policy_agent_mapping = worker.policy_config["multiagent"][
                "policy_mapping_fn"
            ]
        else:
            policy_agent_mapping = default_policy_agent_mapping

    if policy_map is None:
        policy_map = worker.policy_map
    state_init = {p: m.get_initial_state() for p, m in policy_map.items()}
    use_lstm = {p: len(s) > 0 for p, s in state_init.items()}
    action_init = {
        p: flatten_to_single_ndarray(m.action_space.sample())
        for p, m in policy_map.items()
    }

    # If monitoring has been requested, manually wrap our environment with a
    # gym monitor, which is set to record every episode.
    if video_dir:
        env = gym_wrappers.Monitor(
            env=env,
            directory=video_dir,
            video_callable=lambda x: True,
            force=True,
        )

    random_policy_id = list(policy_map.keys())[0]
    virtual_global_timestep = worker.get_policy(
        random_policy_id
    ).global_timestep

    steps = 0
    episodes = 0
    while _keep_going(steps, num_steps, episodes, num_episodes):
        # logger.info(f"Starting epsiode {episodes} in rollout")
        # print(f"Starting epsiode {episodes} in rollout")
        mapping_cache = {}  # in case policy_agent_mapping is stochastic
        saver.begin_rollout()
        obs, agent_states = _get_first_obs(
            env,
            reset_env_before,
            episodes,
            last_obs,
            mapping_cache,
            state_init,
            last_rnn_states,
        )
        prev_actions = DefaultMapping(
            lambda agent_id_: action_init[mapping_cache[agent_id_]]
        )
        prev_rewards = collections.defaultdict(lambda: 0.0)
        done = False
        reward_total = 0.0
        while not done and _keep_going(
            steps, num_steps, episodes, num_episodes
        ):
            multi_obs = obs if multiagent else {_DUMMY_AGENT_ID: obs}
            action_dict = {}
            virtual_global_timestep += 1
            for agent_id, a_obs in multi_obs.items():
                if a_obs is not None:
                    policy_id = mapping_cache.setdefault(
                        agent_id, policy_agent_mapping(agent_id)
                    )
                    p_use_lstm = use_lstm[policy_id]
                    # print("p_use_lstm", p_use_lstm)
                    # print(
                    #     agent_id,
                    #     "agent_states[agent_id]",
                    #     agent_states[agent_id],
                    # )
                    if p_use_lstm:
                        a_action, p_state, _ = _worker_compute_action(
                            worker,
                            timestep=virtual_global_timestep,
                            observation=a_obs,
                            state=agent_states[agent_id],
                            prev_action=prev_actions[agent_id],
                            prev_reward=prev_rewards[agent_id],
                            policy_id=policy_id,
                            explore=explore,
                        )
                        # print(
                        #     "after rollout _worker_compute_action p_state",
                        #     p_state,
                        # )
                        agent_states[agent_id] = p_state
                    else:
                        a_action = _worker_compute_action(
                            worker,
                            virtual_global_timestep,
                            observation=a_obs,
                            prev_action=prev_actions[agent_id],
                            prev_reward=prev_rewards[agent_id],
                            policy_id=policy_id,
                            explore=explore,
                        )
                    a_action = flatten_to_single_ndarray(a_action)
                    action_dict[agent_id] = a_action
                    prev_actions[agent_id] = a_action

            action = action_dict

            action = action if multiagent else action[_DUMMY_AGENT_ID]
            next_obs, reward, done, info = env.step(action)
            if multiagent:
                for agent_id, r in reward.items():
                    prev_rewards[agent_id] = r
            else:
                prev_rewards[_DUMMY_AGENT_ID] = reward

            if multiagent:
                done = done["__all__"]
                reward_total += sum(
                    r for r in reward.values() if r is not None
                )
            else:
                reward_total += reward
            if not no_render:
                env.render()
            saver.append_step(obs, action, next_obs, reward, done, info)
            steps += 1
            obs = next_obs
        saver.end_rollout()
        if done:
            episodes += 1
    return saver


def _keep_going(steps, num_steps, episodes, num_episodes):
    """
    Modified version.
    Determine whether we have collected enough data
    """

    if num_episodes and num_steps:
        return episodes < num_episodes and steps < num_steps
    # if num_episodes is set, this overrides num_steps
    if num_episodes:
        return episodes < num_episodes
    # if num_steps is set, continue until we reach the limit
    if num_steps:
        return steps < num_steps
    # otherwise keep going forever
    return True


def _get_first_obs(
    env,
    reset_env_before,
    episodes,
    last_obs,
    mapping_cache,
    state_init,
    last_rnn_states,
):
    if reset_env_before or episodes > 0:
        obs = env.reset()
        agent_states = DefaultMapping(
            lambda agent_id_: state_init[mapping_cache[agent_id_]]
        )
    else:
        obs = last_obs
        if last_rnn_states is not None:
            agent_states = DefaultMapping(
                lambda agent_id_: last_rnn_states[mapping_cache[agent_id_]]
            )
        else:
            agent_states = DefaultMapping(
                lambda agent_id_: state_init[mapping_cache[agent_id_]]
            )
    return obs, agent_states


def _worker_compute_action(
    worker,
    timestep,
    observation: TensorStructType,
    state: List[TensorStructType] = None,
    prev_action: TensorStructType = None,
    prev_reward: float = None,
    info: EnvInfoDict = None,
    policy_id: PolicyID = DEFAULT_POLICY_ID,
    full_fetch: bool = False,
    explore: bool = None,
) -> TensorStructType:
    """
    Modified version of the Trainer compute_action method
    """
    if state is None:
        state = []
    # Check the preprocessor and preprocess, if necessary.
    pp = worker.preprocessors[policy_id]
    if type(pp).__name__ != "NoPreprocessor":
        observation = pp.transform(observation)
    filtered_obs = worker.filters[policy_id](observation, update=False)
    result = worker.get_policy(policy_id).compute_single_action(
        filtered_obs,
        state,
        prev_action,
        prev_reward,
        info,
        clip_actions=worker.policy_config["clip_actions"],
        explore=explore,
        timestep=timestep,
    )

    if state or full_fetch:
        return result
    else:
        return result[0]  # backwards compatibility
