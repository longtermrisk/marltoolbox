import json
import logging
import os

import numpy as np
from ray.rllib.evaluation import MultiAgentEpisode
from ray.tune.logger import SafeFallbackEncoder

logger = logging.getLogger(__name__)


class FullEpisodeLogger:
    """
    Helper to log the entire history of one episode as txt
    """

    def __init__(
        self, logdir: str, log_interval: int, convert_one_hot_obs_to_idx: bool
    ):
        """

        :param logdir: dir where to save the log file with the full episode
        :param log_interval: interval (in number of episode) between the log
            of two episodes
        :param convert_one_hot_obs_to_idx: bool flag to chose to convert the
            observation to their idx
            (indented to bue used when dealing with one hot observations)
        """
        self.log_interval = log_interval
        self.log_ful_epi_one_hot_obs = convert_one_hot_obs_to_idx

        file_path = os.path.join(logdir, "full_episodes_logs.json")
        self.file_path = os.path.expanduser(file_path)
        logger.info(f"FullEpisodeLogger: using as file_path: {self.file_path}")

        self._init_logging_new_full_episode()
        self.internal_episode_counter = -1
        self.step_counter = 0
        self.episode_finised = True

        self.json_logger = JsonSimpleLogger(self.file_path)

    def on_episode_start(self):
        if self.episode_finised:
            self.episode_finised = False
            self.internal_episode_counter += 1

        if self.internal_episode_counter % self.log_interval == 0:
            self._init_logging_new_full_episode()
            self.json_logger.open()
            self.json_logger.write_json(
                {"status": f"start of episode {self.internal_episode_counter}"}
            )
            self.json_logger.write("\n")

    def _init_logging_new_full_episode(self):
        self._log_current_full_episode = True
        self._log_full_epi_tmp_data = {}

    def on_episode_step(
        self, episode: MultiAgentEpisode = None, step_data: dict = None
    ):
        if not self._log_current_full_episode:
            return None

        assert episode is not None or step_data is not None
        assert episode is None or step_data is None

        if step_data is None:
            step_data = {}
            for agent_id, policy in episode._policies.items():

                if agent_id in self._log_full_epi_tmp_data.keys():
                    obs_before_act = self._log_full_epi_tmp_data[agent_id]
                else:
                    obs_before_act = None
                action = episode.last_action_for(agent_id).tolist()
                epi = episode.episode_id
                rewards = episode._agent_reward_history[agent_id]
                reward = rewards[-1] if len(rewards) > 0 else None
                info = episode.last_info_for(agent_id)
                if hasattr(policy, "to_log"):
                    info.update(policy.to_log)
                else:
                    logger.info(
                        f"policy {policy} doesn't have attrib "
                        "to_log. hasattr(policy, 'to_log'): "
                        f"{hasattr(policy, 'to_log')}"
                    )
                # Episode provide the last action with the given last
                # observation produced by this action. But we need the
                # observation that cause the agent to play this action
                # thus the observation n-1
                obs_after_act = episode.last_observation_for(agent_id)
                self._log_full_epi_tmp_data[agent_id] = obs_after_act

                if self.log_ful_epi_one_hot_obs:
                    obs_before_act = np.argwhere(obs_before_act)
                    obs_after_act = np.argwhere(obs_after_act)

                step_data[agent_id] = {
                    "obs_before_act": obs_before_act,
                    "obs_after_act": obs_after_act,
                    "action": action,
                    "reward": reward,
                    "info": info,
                    "epi": epi,
                }

        self.json_logger.write_json(step_data)
        self.json_logger.write("\n")
        self.step_counter += 1

    def on_episode_end(self, base_env=None):
        if self._log_current_full_episode:
            if base_env is not None:
                env = base_env.get_unwrapped()[0]
                if hasattr(env, "max_steps"):
                    assert self.step_counter == env.max_steps, (
                        "The number of steps written to full episode "
                        "log file must be equal to the number of step in an "
                        f"episode self.step_counter {self.step_counter} "
                        f"must equal env.max_steps {env.max_steps}. "
                        "Otherwise there are some issue with the "
                        "state of the callback object, maybe being used by "
                        "several experiments at the same time."
                    )
            self.json_logger.write_json(
                {"status": f"end of episode {self.internal_episode_counter}"}
            )
            self.json_logger.write("\n")
            self.json_logger.write("\n")
            self.json_logger.flush()
            self.json_logger.close()
            self._log_current_full_episode = False
            self.step_counter = 0
        self.episode_finised = True


class JsonSimpleLogger:
    """
    Simple logger in json format
    """

    def __init__(self, file_path):
        """

        :param file_path: file path to the file to save to
        """
        self.local_file = file_path

    def write_json(self, json_data):
        json.dump(json_data, self, cls=SafeFallbackEncoder)

    def write(self, b):
        self.local_out.write(b)

    def flush(self):
        self.local_out.flush()

    def open(self):
        self.local_out = open(self.local_file, "a")

    def close(self):
        self.local_out.close()
