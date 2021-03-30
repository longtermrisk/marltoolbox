import copy

import numpy as np
import torch
from ray.rllib.models import MODEL_DEFAULTS
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.torch_action_dist import TorchCategorical
from ray.rllib.utils.schedules import PiecewiseSchedule

from marltoolbox.envs.coin_game import \
    CoinGame
from marltoolbox.envs.matrix_sequential_social_dilemma import \
    IteratedPrisonersDilemma
from marltoolbox.utils import exploration

ROUNDING_ERROR = 1e-3


def assert_equal_wt_some_epsilon(v1, v2):
    delta = torch.abs(v1 - v2)
    assert torch.all(delta < ROUNDING_ERROR)


def test_clusterize_by_distance():
    output = exploration.clusterize_by_distance(
        torch.Tensor([0.0, 0.4, 1.0, 1.4, 1.8, 3.0]), 0.5)
    assert_equal_wt_some_epsilon(
        output,
        torch.Tensor([0.2000, 0.2000, 1.4000, 1.4000, 1.4000, 3.0000]))

    output = exploration.clusterize_by_distance(
        torch.Tensor([0.0, 0.5, 1.0, 1.4, 1.8, 3.0]), 0.5)
    assert_equal_wt_some_epsilon(
        output,
        torch.Tensor([0.0000, 0.5000, 1.4000, 1.4000, 1.4000, 3.0000]))

    output = exploration.clusterize_by_distance(
        torch.Tensor([-10.0, -9.8, 1.0, 1.4, 1.8, 3.0]), 0.5)
    assert_equal_wt_some_epsilon(
        output,
        torch.Tensor([-9.9000, -9.9000, 1.4000, 1.4000, 1.4000, 3.0000]))

    output = exploration.clusterize_by_distance(
        torch.Tensor([-1.0, -0.51, -0.1, 0.0, 0.1, 0.51, 1.0]), 0.5)
    assert_equal_wt_some_epsilon(
        output,
        torch.Tensor([0., 0., 0., 0., 0., 0., 0.]))


class TestSoftQSchedule:

    def set_class_to_test(self):
        self.class_to_test = exploration.SoftQSchedule

    def test__set_temperature_wt_explore(self):
        self.set_class_to_test()
        self.arrange_for_simple_ipd()

        self.softqschedule._set_temperature(
            explore=True, timestep=0)
        assert self.softqschedule.temperature == self.initial_temperature

        self.softqschedule._set_temperature(
            explore=True, timestep=self.temperature_timesteps)
        assert self.softqschedule.temperature == self.final_temperature

        self.softqschedule._set_temperature(
            explore=True, timestep=self.temperature_timesteps // 2)
        assert abs(self.softqschedule.temperature -
                   (self.initial_temperature - self.final_temperature) / 2) < \
               ROUNDING_ERROR

    def test__set_temperature_wtout_explore(self):
        self.set_class_to_test()
        self.arrange_for_simple_ipd()

        self.softqschedule._set_temperature(
            explore=False, timestep=0)
        assert self.softqschedule.temperature == 1.0

        self.softqschedule._set_temperature(
            explore=False, timestep=self.temperature_timesteps)
        assert self.softqschedule.temperature == 1.0

        self.softqschedule._set_temperature(
            explore=False, timestep=self.temperature_timesteps // 2)
        assert self.softqschedule.temperature == 1.0

    def test__set_temperature_wt_explore_wt_multi_steps_schedule(self):
        self.class_to_test = exploration.SoftQSchedule
        self.arrange_for_multi_step_wt_coin_game()

        self.softqschedule._set_temperature(
            explore=True, timestep=0)
        assert self.softqschedule.temperature == 2.0

        self.softqschedule._set_temperature(
            explore=True, timestep=2000)
        assert self.softqschedule.temperature == 0.1

        self.softqschedule._set_temperature(
            explore=True, timestep=3000)
        assert self.softqschedule.temperature == 0.1

        self.softqschedule._set_temperature(
            explore=True, timestep=500)
        assert abs(self.softqschedule.temperature - 1.25) < ROUNDING_ERROR

        self.softqschedule._set_temperature(
            explore=True, timestep=1500)
        assert abs(self.softqschedule.temperature - 0.3) < ROUNDING_ERROR

    def arrange_for_simple_ipd(self):
        self.initial_temperature = 1.0
        self.final_temperature = 1e-6
        self.temperature_timesteps = int(1e5)
        self.temperature_schedule = None
        self.init_ipd_scheduler()

    def arrange_for_multi_step_wt_coin_game(self):
        self.initial_temperature = 0.0
        self.final_temperature = 0.0
        self.temperature_timesteps = 0.0
        self.temperature_schedule = PiecewiseSchedule(
            endpoints=[
                (0, 2.0),
                (1000, 0.5),
                (2000, 0.1)],
            outside_value=0.1,
            framework="torch")
        self.init_coin_game_scheduler()

    def init_ipd_scheduler(self):
        self.softqschedule = self.init_scheduler(
            IteratedPrisonersDilemma.ACTION_SPACE,
            IteratedPrisonersDilemma.OBSERVATION_SPACE
        )

    def init_coin_game_scheduler(self):
        self.softqschedule = self.init_scheduler(
            CoinGame.ACTION_SPACE,
            CoinGame({}).OBSERVATION_SPACE
        )

    def init_scheduler(self, action_space, obs_space):
        return self.class_to_test(
            action_space=action_space,
            framework="torch",
            initial_temperature=self.initial_temperature,
            final_temperature=self.final_temperature,
            temperature_timesteps=self.temperature_timesteps,
            temperature_schedule=self.temperature_schedule,
            policy_config={},
            num_workers=0,
            worker_index=0,
            model=FullyConnectedNetwork(
                obs_space=obs_space,
                action_space=action_space,
                num_outputs=action_space.n,
                name="fc",
                model_config=MODEL_DEFAULTS
            )
        )

    def test__apply_temperature(self):
        self.set_class_to_test()
        self.arrange_for_multi_step_wt_coin_game()

        for _ in range(10):
            self.apply_and_assert_apply_temperature(
                temperature=self.random_temperature(),
                inputs=self.random_inputs()[0],
            )

    def apply_and_assert_apply_temperature(self, temperature, inputs):
        action_distribution, action_dist_class = \
            self.set_temperature_and_get_args(temperature=temperature,
                                              inputs=inputs)

        new_action_distribution = self.softqschedule._apply_temperature(
            copy.deepcopy(action_distribution), action_dist_class)

        assert all(
            abs(n_v - v / self.softqschedule.temperature) < ROUNDING_ERROR
            for v, n_v in zip(action_distribution.inputs,
                              new_action_distribution.inputs))

    def set_temperature_and_get_args(self, temperature, inputs):
        action_dist_class = TorchCategorical
        action_distribution = TorchCategorical(
            inputs, self.softqschedule.model, temperature=1.0)
        self.softqschedule.temperature = temperature
        return action_distribution, action_dist_class

    def test_get_exploration_action_wtout_explore(self):
        self.helper_test_get_exploration_action_wt_explore(explore=False)

    def random_inputs(self):
        return np.random.random(
            size=(1, np.random.randint(1, 50, size=1)[0]))

    def random_timestep(self):
        return np.random.randint(0, 10000, size=1)[0]

    def random_temperature(self):
        return np.random.random(size=1)[0] * 10 + 1e-9

    def apply_and_assert_get_exploration_action(
            self, inputs, explore, timestep):

        initial_action_distribution, _ = \
            self.set_temperature_and_get_args(temperature=1.0,
                                              inputs=inputs)
        action_distribution = copy.deepcopy(initial_action_distribution)

        _ = self.softqschedule.get_exploration_action(
            action_distribution,
            timestep=timestep,
            explore=explore
        )

        temperature = self.softqschedule.temperature if explore else 1.0
        errors = [abs(n_v - v / temperature)
                  for v, n_v in zip(initial_action_distribution.inputs[0],
                                    action_distribution.inputs[0])]
        assert all(err < ROUNDING_ERROR for err in errors), f"errors: {errors}"

    def test_get_exploration_action_wt_explore(self):
        self.helper_test_get_exploration_action_wt_explore(explore=True)

    def helper_test_get_exploration_action_wt_explore(self, explore):
        self.set_class_to_test()
        self.arrange_for_multi_step_wt_coin_game()

        for _ in range(10):
            self.apply_and_assert_get_exploration_action(
                inputs=self.random_inputs(),
                explore=explore,
                timestep=self.random_timestep())


class TestSoftQScheduleWtClustering(TestSoftQSchedule):

    def set_class_to_test(self):
        self.class_to_test = exploration.SoftQScheduleWtClustering

    def helper_test_get_exploration_action_wt_explore(self, explore):
        self.set_class_to_test()
        self.arrange_for_multi_step_wt_coin_game()

        for inputs in self.get_inputs_list():
            self.apply_and_assert_get_exploration_action(
                inputs=inputs,
                explore=explore,
                timestep=self.random_timestep())

    def get_inputs_list(self):
        return [
            [[1.0, 0.0]],
            [[5.0, -1.0]],
            [[1.0, 1.6]],
            [[101, -2.3]],
            [[65, 98, 13, 56, 123, 156, 84]],
        ]
