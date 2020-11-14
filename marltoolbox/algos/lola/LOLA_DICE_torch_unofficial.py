##########
# Code modified from: https://github.com/alexis-jacq/LOLA_DiCE/blob/master/ipd_DiCE.py
# WORK IN PROGRESS!! This implementation is not working properly.
##########

from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np

import torch

from ray.rllib.agents import Trainer
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import TensorType
from ray.rllib.utils.torch_ops import convert_to_torch_tensor, convert_to_non_torch_type
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID

def magic_box(x):
    return torch.exp(x - x.detach())

class Memory():
    def __init__(self, use_baseline,gamma):
        self.self_logprobs = []
        self.other_logprobs = []
        self.values = []
        self.rewards = []
        self.USE_BASELINE = use_baseline
        self.GAMMA = gamma

    def add(self, lp, other_lp, v, r):
        self.self_logprobs.append(lp)
        self.other_logprobs.append(other_lp)
        # TODO add values
        self.values.append(v)
        self.rewards.append(r)

    def dice_objective(self):
        self_logprobs = torch.stack(self.self_logprobs, dim=1)
        other_logprobs = torch.stack(self.other_logprobs, dim=1)
        # TODO add values
        values = torch.stack(self.values, dim=1)
        rewards = torch.stack(self.rewards, dim=1)

        # apply discount:
        cum_discount = torch.cumprod(self.GAMMA * torch.ones(*rewards.size()), dim=1)/self.GAMMA
        discounted_rewards = rewards * cum_discount
        # TODO add values
        discounted_values = values * cum_discount

        # stochastics nodes involved in rewards dependencies:
        dependencies = torch.cumsum(self_logprobs + other_logprobs, dim=1)

        # logprob of each stochastic nodes:
        stochastic_nodes = self_logprobs + other_logprobs

        # dice objective:
        dice_objective = torch.mean(torch.sum(magic_box(dependencies) * discounted_rewards, dim=1))

        # TODO add values
        if self.USE_BASELINE:
            # variance_reduction:
            baseline_term = torch.mean(torch.sum((1 - magic_box(stochastic_nodes)) * discounted_values, dim=1))
            dice_objective = dice_objective + baseline_term

        return -dice_objective # want to minimize -objective

    def value_loss(self):
        values = torch.stack(self.values, dim=1)
        rewards = torch.stack(self.rewards, dim=1)
        return torch.mean((rewards - values)**2)


class LolaDiceTorchTrainerMixin:
    # TODO get these hyper-parameters from a config dict
    N_AGENTS = 2
    N_INNER_STEPS = 2
    GAMMA = 0.96
    BASE_LR = 0.001
    LR_OUTER = 0.2 * BASE_LR
    LR_INNER = 0.3 * BASE_LR
    LR_VALUE = 0.1 * BASE_LR * 100.0
    LR_OM = 0.1 * BASE_LR
    USE_BASELINE = True
    USE_DICE = True
    USE_OPP_MODELING = False
    LOLA_BATCH_SIZE = 128
    LEN_ROLLOUT = 150

    def lola_init(self):
        # TODO assert 2 agents/policies
        # Do not support remote workers
        assert len(self.workers._remote_workers) == 0

        self.lola_learning_worker = self.workers._local_worker
        self.lola_env = self.lola_learning_worker.env

        self.lola_outer_optimizers = {policy_id: torch.optim.SGD(policy.model.parameters(), lr=self.LR_OUTER)
                            for policy_id, policy in self.lola_learning_worker.policy_map.items()
                                if isinstance(policy, LolaDiceTorchPolicyMixin)}
        # self.lola_inner_optimizers = {policy_id: torch.optim.SGD(policy.model.parameters(), lr=self.LR_INNER)
        #                     for policy_id, policy in self.lola_learning_worker.policy_map.items()
        #                         if isinstance(policy, LolaDiceTorchPolicyMixin)}
        self.lola_value_optimizers = {policy_id: torch.optim.SGD(policy.model.parameters(), lr=self.LR_VALUE)
                            for policy_id, policy in self.lola_learning_worker.policy_map.items()
                                if isinstance(policy, LolaDiceTorchPolicyMixin)}

        # Create a dict in each policies to store stuff to log
        for policy_id, policy in self.lola_learning_worker.policy_map.items():
            policy.__setattr__("to_log", {})

    def lola_update(self):

        print("on_train_result")
        # TODO remove this? This should be useless
        [ optim.zero_grad() for optim in self.lola_outer_optimizers.values()]
        # [ optim.zero_grad() for optim in self.lola_inner_optimizers.values()]
        [ optim.zero_grad() for optim in self.lola_value_optimizers.values()]

        policies_initial_weights = {policy_id:policy.get_weights()
                            for policy_id, policy in self.lola_learning_worker.policy_map.items()}

        ag1, ag2 = tuple(self.lola_learning_worker.policy_map.items())
        ag1_id, ag1_policy = ag1
        ag2_id, ag2_policy = ag2

        opp_lola_update_done = False
        for policy_id, policy in self.lola_learning_worker.policy_map.items():
            if isinstance(policy, LolaDiceTorchPolicyMixin):
                # TODO can I simplify this?
                if policy_id == ag1_id:
                    opponent_id, opponent_policy = ag2
                elif policy_id == ag2_id:
                    opponent_id, opponent_policy = ag1
                else:
                    raise ValueError(f"policy_id {policy_id} must be one of {[ag1_id,ag2_id]}")

                if opp_lola_update_done:
                    opponent_weights_after_lola_update = opponent_policy.get_weights()
                    opponent_policy.set_weights(policies_initial_weights[opponent_id])

                for k in range(self.N_INNER_STEPS):
                    # estimate other's gradients from in_lookahead:
                    opponent_policy = self.in_lookahead(own_policy=policy, own_policy_id=policy_id,
                                              opponent_policy=opponent_policy, opponent_policy_id=opponent_id)
                # update parameters from out_lookahead:
                policy = self.out_lookahead(own_policy=policy, own_policy_id=policy_id,
                                           opponent_policy=opponent_policy, opponent_policy_id=opponent_id)

                if not opp_lola_update_done:
                    opponent_policy.set_weights(policies_initial_weights[opponent_id])
                if opp_lola_update_done:
                    opponent_policy.set_weights(opponent_weights_after_lola_update)

                # TODO thus LOLA only supports rollout workers working on full episodes => assert this
                self.lola_env.reset()
                opp_lola_update_done = True

        # TODO need to flush some more stuff? or restore some previous values?

        print("lola_update done")

    def in_lookahead(self,own_policy, own_policy_id,
                     opponent_policy, opponent_policy_id):

        observations_batch = [self.lola_env.reset() for i in range(self.LOLA_BATCH_SIZE)]
        dice_memory = Memory(use_baseline=self.USE_BASELINE,gamma=self.GAMMA)
        # TODO compute batch + pass grad in log_prob_act
        # TODO change LEN_ROLLOUT to env max steps
        for t in range(self.LEN_ROLLOUT):
            obs_batch = {f"{own_policy_id}_{i}":obs[own_policy_id] for i, obs in enumerate(observations_batch)}
            obs_batch_opp = {f"{opponent_policy_id}_{i}":obs[opponent_policy_id] for i, obs in enumerate(observations_batch)}

            action, _, infos = self.compute_actions_wt_grad(
                observations=obs_batch,
                policy_id=own_policy_id,
                full_fetch=True)
            action_opp, _, infos_opp = self.compute_actions_wt_grad(
                observations=obs_batch_opp,
                policy_id=opponent_policy_id,
                full_fetch=True)
            opp_values = opponent_policy.model.value_function()

            actions = action
            actions.update(action_opp)

            reward_opp_batch = []
            observations_batch = []
            for i in range(self.LOLA_BATCH_SIZE):
                actions_i = {own_policy_id:actions[f"{own_policy_id}_{i}"],
                             opponent_policy_id:actions[f"{opponent_policy_id}_{i}"]}
                observations, rewards, _, _ = self.lola_env.step(actions_i)
                reward, reward_opp = rewards[own_policy_id], rewards[opponent_policy_id]
                reward_opp_batch.append(reward_opp)
                observations_batch.append(observations)

            log_prob_act = infos[SampleBatch.ACTION_LOGP]
            log_prob_act_opp = infos_opp[SampleBatch.ACTION_LOGP]

            dice_memory.add(log_prob_act_opp, log_prob_act,
                            opp_values, torch.tensor(reward_opp_batch).float())

        opponent_dice_loss = dice_memory.dice_objective()
        own_policy.to_log["inner_loss"] = own_policy.to_log.get("inner_loss",0) + opponent_dice_loss
        self._update_policy_wt_higher_order_grad(agent_policy=opponent_policy,
                                                 loss=opponent_dice_loss)

        # useless but more readable
        return opponent_policy

    def out_lookahead(self,own_policy, own_policy_id, opponent_policy, opponent_policy_id):

        observations_batch = [self.lola_env.reset() for i in range(self.LOLA_BATCH_SIZE)]
        dice_memory = Memory(use_baseline=self.USE_BASELINE,gamma=self.GAMMA)
        # TODO change LEN_ROLLOUT to env max steps
        for t in range(self.LEN_ROLLOUT):
            obs_batch = {f"{own_policy_id}_{i}": obs[own_policy_id] for i, obs in enumerate(observations_batch)}
            obs_batch_opp = {f"{opponent_policy_id}_{i}": obs[opponent_policy_id] for i, obs in enumerate(observations_batch)}

            action, _, infos = self.compute_actions_wt_grad(
                observations=obs_batch,
                policy_id=own_policy_id,
                full_fetch=True)
            action_opp, _, infos_opp = self.compute_actions_wt_grad(
                observations=obs_batch_opp,
                policy_id=opponent_policy_id,
                full_fetch=True)
            actions = action
            actions.update(action_opp)
            own_values = own_policy.model.value_function()

            reward_batch = []
            observations_batch = []
            for i in range(self.LOLA_BATCH_SIZE):
                actions_i = {own_policy_id:actions[f"{own_policy_id}_{i}"],
                             opponent_policy_id:actions[f"{opponent_policy_id}_{i}"]}
                observations, rewards, _, _ = self.lola_env.step(actions_i)
                reward, reward_opp = rewards[own_policy_id], rewards[opponent_policy_id]
                reward_batch.append(reward)
                observations_batch.append(observations)

            log_prob_act = infos[SampleBatch.ACTION_LOGP]
            log_prob_act_opp = infos_opp[SampleBatch.ACTION_LOGP]

            dice_memory.add(log_prob_act, log_prob_act_opp,
                            own_values, torch.tensor(reward_batch).float())

        # update self theta
        own_dice_loss = dice_memory.dice_objective()
        own_policy.to_log["outer_loss"] = own_policy.to_log.get("outer_loss",0) + own_dice_loss
        self._update_policy(agent_id=own_policy_id, agent_policy=own_policy,
                            loss=own_dice_loss, optimizers=self.lola_outer_optimizers,
                            retain_graph=True, optim_step=False)

        # update self value:
        v_loss = dice_memory.value_loss()
        own_policy.to_log["value_loss"] = own_policy.to_log.get("value_loss",0) + v_loss
        self._update_policy(agent_id=own_policy_id, agent_policy=own_policy,
                            loss=v_loss, optimizers=self.lola_value_optimizers,
                            zero_grad=False)
        return own_policy

    def _update_policy_wt_higher_order_grad(self, agent_policy, loss):
        grads = torch.autograd.grad(loss, agent_policy.model.parameters(),
                                    create_graph=True, retain_graph=True, allow_unused=True)
        for p, p_grad in zip(agent_policy.model.parameters(), grads):
            p.data.sub_(self.LR_INNER * p_grad)

    def _update_policy(self, agent_id, agent_policy, loss, optimizers,
                       retain_graph=False, optim_step=True, zero_grad=True):
        policy_optimizer = optimizers[agent_id]
        if zero_grad:
            policy_optimizer.zero_grad()
        loss.backward(retain_graph=retain_graph)
        if optim_step:
            policy_optimizer.step()

    # Clone of the Trainer.compute_actions method but modified to support gradients
    # TODO is there a cleaner way to do this (preprocessing + forward + postprocessing) with gradients?
    def compute_actions_wt_grad(self,
                        observations,
                        state=None,
                        prev_action=None,
                        prev_reward=None,
                        info=None,
                        policy_id=DEFAULT_POLICY_ID,
                        full_fetch=False,
                        explore=None):
        """Computes an action for the specified policy on the local Worker.

        Note that you can also access the policy object through
        self.get_policy(policy_id) and call compute_actions() on it directly.

        Arguments:
            observation (obj): observation from the environment.
            state (dict): RNN hidden state, if any. If state is not None,
                then all of compute_single_action(...) is returned
                (computed action, rnn state(s), logits dictionary).
                Otherwise compute_single_action(...)[0] is returned
                (computed action).
            prev_action (obj): previous action value, if any
            prev_reward (int): previous reward, if any
            info (dict): info object, if any
            policy_id (str): Policy to query (only applies to multi-agent).
            full_fetch (bool): Whether to return extra action fetch results.
                This is always set to True if RNN state is specified.
            explore (bool): Whether to pick an exploitation or exploration
                action (default: None -> use self.config["explore"]).

        Returns:
            any: The computed action if full_fetch=False, or
            tuple: The full output of policy.compute_actions() if
                full_fetch=True or we have an RNN-based Policy.
        """
        # Preprocess obs and states
        stateDefined = state is not None
        policy = self.get_policy(policy_id)
        filtered_obs, filtered_state = [], []
        for agent_id, ob in observations.items():
            worker = self.workers.local_worker()
            preprocessed = worker.preprocessors[policy_id].transform(ob)
            filtered = worker.filters[policy_id](preprocessed, update=False)
            filtered_obs.append(filtered)
            if state is None:
                continue
            elif agent_id in state:
                filtered_state.append(state[agent_id])
            else:
                filtered_state.append(policy.get_initial_state())

        # Batch obs and states
        obs_batch = np.stack(filtered_obs)
        if state is None:
            state = []
        else:
            state = list(zip(*filtered_state))
            state = [np.stack(s) for s in state]

        # Figure out the current (sample) time step and pass it into Policy.
        # 1) change made here: commenting
        # self.global_vars["timestep"] += 1

        # Batch compute actions
        # 2) change made here: using compute_actions_wt_grad instead of compute_actions
        actions, states, infos = policy.compute_actions_wt_grad(
            obs_batch,
            state,
            prev_action,
            prev_reward,
            info,
            clip_actions=self.config["clip_actions"],
            explore=explore,
            timestep=self.global_vars["timestep"])

        # 3) change made here: remove the slow space_utils.unbatch call
        # # Unbatch actions for the environment
        # atns, actions = space_utils.unbatch(actions), {}
        # for key, atn in zip(observations, atns):
        #     actions[key] = atn
        final_actions = {}
        for key, atn in zip(observations, actions):
            final_actions[key] = atn
        actions = final_actions

        # Unbatch states into a dict
        # unbatched_states = {}
        # for idx, agent_id in enumerate(observations):
        #     unbatched_states[agent_id] = [s[idx] for s in states]

        # Return only actions or full tuple
        if stateDefined or full_fetch:
            # return actions, unbatched_states, infos
            return actions, None, infos
        else:
            return actions


# TODO Split this in two classes: one Mixin for all policies played
#  and one Mixin to select which policy need LOLA updates
class LolaDiceTorchPolicyMixin:

    # Clone of the TorchPolicy.compute_actions method but modified to support gradients
    # TODO is there a cleaner way to do this (preprocessing + forward + postprocessing) with gradients?
    def compute_actions_wt_grad(
            self,
            obs_batch: Union[List[TensorType], TensorType],
            state_batches: Optional[List[TensorType]] = None,
            prev_action_batch: Union[List[TensorType], TensorType] = None,
            prev_reward_batch: Union[List[TensorType], TensorType] = None,
            info_batch: Optional[Dict[str, list]] = None,
            episodes: Optional[List["MultiAgentEpisode"]] = None,
            explore: Optional[bool] = None,
            timestep: Optional[int] = None,
            **kwargs) -> \
            Tuple[TensorType, List[TensorType], Dict[str, TensorType]]:
        explore = explore if explore is not None else self.config["explore"]
        timestep = timestep if timestep is not None else self.global_timestep

        # 1) change made here: commenting
        # with torch.no_grad():
        seq_lens = torch.ones(len(obs_batch), dtype=torch.int32)
        input_dict = self._lazy_tensor_dict({
            SampleBatch.CUR_OBS: np.asarray(obs_batch),
            # TODO need to switch to training?
            "is_training": False,
        })
        if prev_action_batch is not None:
            input_dict[SampleBatch.PREV_ACTIONS] = \
                np.asarray(prev_action_batch)
        if prev_reward_batch is not None:
            input_dict[SampleBatch.PREV_REWARDS] = \
                np.asarray(prev_reward_batch)
        state_batches = [
            convert_to_torch_tensor(s, self.device)
            for s in (state_batches or [])
        ]
        actions, state_out, extra_fetches, logp = \
            self._compute_action_helper(
                input_dict, state_batches, seq_lens, explore, timestep)
        # Action-logp and action-prob.
        if logp is not None:
            logp = logp
            extra_fetches[SampleBatch.ACTION_PROB] = torch.exp(logp)
            extra_fetches[SampleBatch.ACTION_LOGP] = logp

        # 2) change made here: keep torch.Tensor for extra_fetches
        # return return convert_to_non_torch_type((actions, state_out, extra_fetches))
        return (convert_to_non_torch_type(actions), state_out, extra_fetches)



class LOLACallbacks:
    def on_train_result(self, *, trainer, result: dict, **kwargs):
        """Called at the end of Trainable.train().

        Args:
            trainer (Trainer): Current trainer instance.
            result (dict): Dict of results returned from trainer.train() call.
                You can mutate this object to add additional metrics.
            kwargs: Forward compatibility placeholder.
        """
        for policy_id, policy in trainer.lola_learning_worker.policy_map.items():
            print("before lola", policy_id, dict(policy.model.named_parameters()))
        to_log = trainer.lola_update()
        for policy_id, policy in trainer.lola_learning_worker.policy_map.items():
            print("after lola", policy_id, dict(policy.model.named_parameters()))
        # TODO envoyer params aux rollout workers!!

        return result

def init_lola(trainer: Trainer) -> None:
    trainer.lola_init()
