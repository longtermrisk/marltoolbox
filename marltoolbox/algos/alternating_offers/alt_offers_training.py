##########
# Part of the code modified from:
# https://github.com/asappresearch/emergent-comms-negotiation
##########

import json
import time
import argparse
import os
import datetime
from os import path
import numpy as np
import pickle
import torch
from torch import autograd, optim, nn
from collections import defaultdict
from torch.autograd import Variable
import torch.nn.functional as F

import marltoolbox.algos.alternating_offers.nets as nets
import marltoolbox.algos.alternating_offers.logging_utils as logging_utils
from marltoolbox.algos.alternating_offers.envs.alt_offers_env import AltOffersEnv, AltOffersEnvMemory, calc_rewards
from marltoolbox.algos.alternating_offers.envs.alt_offers_env_sieve import SievePlayback
from marltoolbox.examples.tune_class_api.alternating_offers.cond_params import cond_params_iter, round_cond_params

from ray import tune


class AltOffersTraining(tune.Trainable):
    def setup(self, args):
        '''One-time setup in the beginning of training
        '''
#         torch.autograd.set_detect_anomaly(True)  # for autograd debug
        self.episodes_completed = 0
        self.start_time = time.time()
        self.args = args
    
        self.type_constr = torch.cuda if args['enable_cuda'] else torch
        if args['train_seed'] is not None:
            np.random.seed(args['train_seed'])  # not sure it would init ALL the seeds, check if use
            torch.manual_seed(args['train_seed'])
            self.train_random_state = np.random.RandomState(args['train_seed'])
        else:
            self.train_random_state = np.random
#         self.test_random_state = np.random.RandomState(args['test_seed']) if args['test_seed'] is not None else np.random
    #     test_batches = alt_offers_env_sampling.generate_test_batches(batch_size=args['batch_size'], num_batches=5, random_state=test_random_state, utility_type=args['utility_type'])
    #     test_hashes = alt_offers_env_sampling.hash_batches(test_batches)
    
    # This part of the code is for loading a frozen model for response training and is usually disabled
        self.fixed_agent_i = None
#         restore_weights_dir = '/home/alexander/ray_results/cross_play_20K_episodes_binding/2021_03_23/12_37_37'
#         checkpoint_dir_name = 'checkpoint_312'
# #         self.fixed_agent_i = 0
#         self.fixed_agent_i = 1
#         trial_info = []
#         for i, trial_name in enumerate(os.listdir(restore_weights_dir)):
#             trial_path = os.path.join(restore_weights_dir, trial_name)
#             if os.path.isdir(trial_path):
#                 cur_trial_info = {}
#                 cur_trial_info['params'] = pickle.load(open(os.path.join(trial_path, 'params.pkl'), 'rb'))
#                 model_path = os.path.join(trial_path, checkpoint_dir_name, 'model.pth')
#                 with open(model_path, 'rb') as f:
#                     state = torch.load(f)
#                 cur_trial_info['agents'] = []
#                 for agent_i in range(2):
#                     cur_trial_info['agents'].append(state['agent%s' % agent_i]['model_state'])  # either a single model or a list of population models
#                 trial_info.append(cur_trial_info)
                
#         matches = {}
#         all_training_modes = list(set([trial['params']['match_mode'] for trial in trial_info]))
#         for mode in all_training_modes:
#             matches[mode] = [trial for trial in trial_info if trial['params']['match_mode'] == mode]
#         match_indices = defaultdict(list)
#         for match_regime, match_trials in matches.items():
#             eq_list = [(i, trial) for (i, trial) in enumerate(match_trials) if round_cond_params(trial['params']['cond_params']) == self.args['cond_params']]
#             indices = [eq_trial[0] for eq_trial in eq_list]
#             match_indices[match_regime].append(indices)
#         agent_0_match_regime, agent_1_match_regime = ('custom_vs_custom_1', 'custom_vs_custom_2')
#         cond_params_list = list(cond_params_iter())
        
#         trial_0_indices = match_indices[agent_0_match_regime][0]
#         trial_1_indices = match_indices[agent_1_match_regime][0]
#         eval_trials_list = [(trial_0_i, trial_1_i) for trial_0_i in trial_0_indices for trial_1_i in trial_1_indices]
#         cur_opponent_index = self.args['repetition_i']
#         if self.fixed_agent_i == 0:
#             trial_0_i = trial_0_indices[cur_opponent_index]
#             trial_0 = matches[agent_0_match_regime][trial_0_i]
#             print('Loading agent 0')
#             self.fixed_agent_weight = matches[agent_0_match_regime][trial_0_i]['agents'][0]
#         else:
#             trial_1_i = trial_1_indices[cur_opponent_index]
#             trial_1 = matches[agent_1_match_regime][trial_1_i]
#             print('Loading agent 1')
#             self.fixed_agent_weight = matches[agent_1_match_regime][trial_1_i]['agents'][1]

        self.agent_models = []
        self.agent_opts = []
        for agent_i in range(2):
            if isinstance(args['hidden_embedding_sizes'][agent_i], list):
#                 print(f"is list, {args['hidden_embedding_sizes'][agent_i]}")
                cur_pop = [
                    nets.AgentModel(
                    enable_binding_comm=args['enable_binding_comm'],
                    enable_cheap_comm=args['enable_cheap_comm'],
                    response_entropy_reg=args['response_entropy_reg'],
                    utterance_entropy_reg=args['utterance_entropy_reg'],
                    proposal_entropy_reg=args['proposal_entropy_reg'],
                    hidden_embedding_size=args['hidden_embedding_sizes'][agent_i][pop_i],)
                    for pop_i in range(len(args['hidden_embedding_sizes']))
                ]
                if args['enable_cuda']:
                    for model in cur_pop:
                        model.cuda()
                if args['agents_sgd']:
                    cur_pop_opts = [optim.SGD(model.parameters(), lr=1e-3, weight_decay=1e-4)
                                    for model in cur_pop]
                else:
                    cur_pop_opts = [optim.Adam(params=model.parameters())
                                    for model in cur_pop]
                self.agent_models.append(cur_pop)
                self.agent_opts.append(cur_pop_opts)
            else:
                model = nets.AgentModel(
                    enable_binding_comm=args['enable_binding_comm'],
                    enable_cheap_comm=args['enable_cheap_comm'],
                    response_entropy_reg=args['response_entropy_reg'],
                    utterance_entropy_reg=args['utterance_entropy_reg'],
                    proposal_entropy_reg=args['proposal_entropy_reg'],
                    hidden_embedding_size=args['hidden_embedding_sizes'][agent_i],
                )
                if args['enable_cuda']:
                    model.cuda()
                if self.fixed_agent_i is not None and agent_i == self.fixed_agent_i:
                    model.load_state_dict(self.fixed_agent_weight)
                self.agent_models.append(model)
                if self.fixed_agent_i is not None and agent_i != self.fixed_agent_i:
                    if args['agents_sgd']:
                        self.agent_opts.append(optim.SGD(self.agent_models[agent_i].parameters(), lr=1e-3, weight_decay=1e-4))
                    else:
                        self.agent_opts.append(optim.Adam(params=self.agent_models[agent_i].parameters()))
                else:
                    self.agent_opts.append(None)
                
        if isinstance(self.agent_models[0], list):
            assert len(self.agent_models[0]) == len(self.agent_models[1])

        if args['enable_arbitrator']:
            assert len(args['hidden_embedding_sizes']) > 2  # need an embedding size for the arbitrator too then
            self.arbitrator_model = nets.ArbitratorModel(entropy_reg=args['arbitrator_entropy_reg'], share_utilities=args['share_utilities'],
                                                         hidden_embedding_size=args['hidden_embedding_sizes'][2])
            if args['enable_cuda']:
                self.arbitrator_model.cuda()
            if args['arbitrator_sgd']:
                self.arbitrator_opt = optim.SGD(self.arbitrator_model.parameters(), lr=1e-3, weight_decay=1e-4)
            else:
                self.arbitrator_opt = optim.Adam(params=self.arbitrator_model.parameters())

        self.last_print = time.time()
        self.baseline = None
    
    def step(self):
        '''Run episodes_per_step episodes
        '''
        self.agent_forward_stats = [{}, {}]
        self.arb_forward_stats = {}
        
        self.num_turns_accum = 0
        self.processed_games_sum = 0
        self.rewards_sum = None
        
        self.main_loss_by_agent_accum = [0, 0]
        self.entropy_loss_by_agent_accum = [0, 0]
        self.arb_main_loss_accum = 0
        self.arb_entropy_loss_accum = 0
        
        for cur_episode_i in range(self.args['episodes_per_step']):
            if isinstance(self.agent_models[0], list):
                batches_num = len(self.agent_models[0])
            else:
                batches_num = 1
            for cur_batch_i in range(batches_num):
                index_0 = None
                index_1 = None
                if isinstance(self.agent_models[0], list):
                    index_0 = np.random.randint(0, len(self.agent_models[0]))
                    index_1 = np.random.randint(0, len(self.agent_models[0]))
                    cur_batch_models = [self.agent_models[0][index_0], self.agent_models[1][index_1]]
                    cur_batch_opts = [self.agent_opts[0][index_0], self.agent_opts[1][index_1]]
                    embedding_sizes = [self.args['hidden_embedding_sizes'][0][index_0], self.args['hidden_embedding_sizes'][1][index_1]]
                else:
                    cur_batch_models = self.agent_models
                    cur_batch_opts = self.agent_opts
                    embedding_sizes = self.args['hidden_embedding_sizes']
                    
                if index_0 is None:
                    env_state = AltOffersEnv(batch_size=self.args['batch_size'], random_state=self.train_random_state, utility_types=self.args['utility_types'])
                else:
                    env_state = AltOffersEnv(batch_size=self.args['batch_size'], random_state=self.train_random_state, utility_types=[self.args['utility_types'][0][index_0], self.args['utility_types'][1][index_1]])

                action_logprobs, entropy_loss_by_agent, self.arb_logprobs_accum, arb_entropy_loss, rewards, num_turns, alive_masks, \
                self.new_agent_forward_stats, self.new_arb_forward_stats = run_episode(
                    args=self.args,
                    env_state=env_state,
                    agent_models=cur_batch_models,
                    arbitrator_model=self.arbitrator_model if self.args['enable_arbitrator'] else None,
                    render_flag=(cur_episode_i == self.args['episodes_per_step']-1) and not self.args['suppress_output'],
                    deterministic=False,
                    embedding_sizes=embedding_sizes,
                )

                for agent in [0, 1]:
                    for key, val in self.new_agent_forward_stats[agent].items():
                        if key in self.agent_forward_stats[agent]:
                            self.agent_forward_stats[agent][key] += val
                        else:
                            self.agent_forward_stats[agent][key] = val
                for key, val in self.new_arb_forward_stats.items():
                    if key in self.arb_forward_stats:
                        self.arb_forward_stats[key] += val
                    else:
                        self.arb_forward_stats[key] = val

                if self.baseline is None:
                    self.baseline = {name: self.type_constr.FloatTensor(self.args['batch_size']).fill_(0) for name in rewards}
                if self.rewards_sum is None:
                    self.rewards_sum = {name: 0.0 for name in rewards}

                baselined_rewards = {name: rewards[name] - self.baseline[name] for name in rewards}

                sgd_decay_episodes = self.args['training_episodes']
                phase = min(float(self.episodes_completed) / sgd_decay_episodes, 1)
                cur_sgd_lr = 1e-3 + phase * (1e-6 - 1e-3)
                if self.args['agents_sgd']:
                    for i in range(2):
                        for group in cur_batch_opts[i].param_groups:
                            group['lr'] = cur_sgd_lr
                if self.args['enable_arbitrator'] and self.args['arbitrator_sgd']:
                    for group in self.arbitrator_opt.param_groups:
                        group['lr'] = cur_sgd_lr

                if self.args['enable_arbitrator']:
                    for i in range(2):
                        if cur_batch_opts[i] is not None:
                            cur_batch_opts[i].zero_grad()

                    orig_welfare_reward_by_agent = [0, 0]
                    welfare_grads = []
                    sieve_playback = SievePlayback(alive_masks, enable_cuda=self.args['enable_cuda'])
                    for t, global_idxes in sieve_playback:  # iterate over which games were active on every step (for each batch elem)
                        agent = t % 2
                        for action in action_logprobs[t]:  # iterate over 7 action_logprobs: 1xresponse, 3xproposal, 3xproposal_repr
                            orig_welfare_reward = baselined_rewards['sum_share_of_max'][global_idxes].float().contiguous().view(
                                sieve_playback.batch_size, 1)
                            orig_welfare_reward_by_agent[agent] += (action * Variable(orig_welfare_reward)).sum()
                    for i in range(2):
                        welfare_grads.append(torch.autograd.grad(orig_welfare_reward_by_agent[i], cur_batch_models[i].parameters(), retain_graph=True))

                    self.arbitrator_opt.zero_grad()
                    for i in range(2):
                        if cur_batch_opts[i] is not None:
                            cur_batch_opts[i].zero_grad()
                    induced_reward_by_agent = [0, 0]
                    induced_grads = []
                    sieve_playback = SievePlayback(alive_masks, enable_cuda=self.args['enable_cuda'])
                    for t, global_idxes in sieve_playback:  # iterate over which games were active on every step (for each batch elem)
                        agent = t % 2
                        for action in action_logprobs[t]:  # iterate over 7 action_logprobs: 1xresponse, 3xproposal, 3xproposal_repr
                            reward_name = 'player0_arb_induced_part' if agent == 0 else 'player1_arb_induced_part'
                            # OK to use baselines for the induced part? Yeah I think this should be an unbiased gradient estimate
                            induced_agent_reward = baselined_rewards[reward_name][global_idxes].float().contiguous().view(sieve_playback.batch_size, 1)
                            cur_arb_logprobs = self.arb_logprobs_accum.sum(1)[global_idxes].view(sieve_playback.batch_size, 1)
                            induced_reward_by_agent[agent] += (action * cur_arb_logprobs * Variable(induced_agent_reward)).sum()
                    for i in range(2):
                        # should account for maxent exploration bonus of each agent here too! but not in the orig_welfare_reward_by_agent.
                        cur_grads = torch.autograd.grad(induced_reward_by_agent[i]-entropy_loss_by_agent[i], cur_batch_models[i].parameters(), retain_graph=True, create_graph=True)
                        induced_grads.append(cur_grads)

                    arb_main_loss = 0
                    for agent_i in range(2):
                        for grad_i in range(len(welfare_grads)):
                            # sums up along batch dimension too
                            arb_main_loss -= (welfare_grads[agent_i][grad_i] * induced_grads[agent_i][grad_i]).sum()
                    cur_arb_main_loss_coeff = self.args['arbitrator_main_loss_coeff']
                    arb_main_loss *= cur_arb_main_loss_coeff
                    arb_loss = arb_entropy_loss + arb_main_loss
                    arb_loss.backward(retain_graph=True)
                    self.arbitrator_opt.step()

                    self.arb_main_loss_accum += arb_main_loss.item()
                    self.arb_entropy_loss_accum += arb_entropy_loss.item()

                for i in range(2):
                    if cur_batch_opts[i] is not None:
                        cur_batch_opts[i].zero_grad()
                main_loss_by_agent = [0, 0]
                sieve_playback = SievePlayback(alive_masks, enable_cuda=self.args['enable_cuda'])
                for t, global_idxes in sieve_playback:  # iterate over which games were active on every step (for each batch elem)
                    agent = t % 2
                    for action in action_logprobs[t]:  # iterate over 7 action_logprobs: 1xresponse, 3xproposal, 3xproposal_repr
                        if not self.args['enable_arbitrator']:
                            reward_name = f'player{agent}_share_of_max'
                        else:
                            reward_name = f'player{agent}_after_arb'

                        # only correct if scale_before_redist==True, otherwise the welfare changes
                        # because of redistribution and needs to be recomputed in a different way
                        
                        if index_0 is not None:
                            pop_i = index_0 if agent == 0 else index_1
                            cur_prosociality_level = self.args['prosociality_levels'][agent][pop_i]
                            cur_fairness_coeff = self.args['fairness_coeffs'][agent][pop_i]
                        else:
                            cur_prosociality_level = self.args['prosociality_levels'][agent]
                            cur_fairness_coeff = self.args['fairness_coeffs'][agent]
                        assert self.args['scale_before_redist'] or cur_prosociality_level==0
                        # to use fairness_coeff with arbitrator, need to compute how fairness penalties affect
                        # the before-redistribution welfare in mediator optimization code, this is not implemented
                        assert not self.args['enable_arbitrator'] or cur_fairness_coeff == 0
                        total_agent_reward = (1-cur_prosociality_level)*baselined_rewards[reward_name][global_idxes] + \
                                              cur_prosociality_level*baselined_rewards['sum_share_of_max'][global_idxes] - \
                                              cur_fairness_coeff*baselined_rewards['difference_in_share_of_max'][global_idxes]

                        total_agent_reward = total_agent_reward.float().contiguous().view(sieve_playback.batch_size, 1)
                        main_loss_by_agent[agent] += -(action * Variable(total_agent_reward)).sum()

                for i in range(2):
                    # rewards can be negative here because of the baseline
                    # entropy_loss is negative bc entropy is positive
                    loss = entropy_loss_by_agent[i] + main_loss_by_agent[i]
                    loss.backward()
                    if cur_batch_opts[i] is not None:
                        cur_batch_opts[i].step()

                    self.main_loss_by_agent_accum[i] += main_loss_by_agent[i].item()
                    self.entropy_loss_by_agent_accum[i] += entropy_loss_by_agent[i].item()


                self.num_turns_accum += num_turns.sum()
                for name in rewards:
                    self.rewards_sum[name] += rewards[name].sum()
                    self.baseline[name] = 0.7*self.baseline[name] + 0.3*rewards[name].mean()
                self.processed_games_sum += self.args['batch_size']
                self.episodes_completed += 1

        log_data = logging_utils.get_step_log(self)
        if not self.args['suppress_output']:
            logging_utils.print_step_log(self, log_data)
        return log_data

    def save_checkpoint(self, tmp_checkpoint_dir):
        state = {}
        for i in range(2):
            state['agent%s' % i] = {}
            if isinstance(self.agent_models[i], list):
                state['agent%s' % i]['model_state'] = [model.state_dict() for model in self.agent_models[i]]
                state['agent%s' % i]['opt_state'] = [opt.state_dict() if opt is not None else None for opt in self.agent_opts[i]]
            else:
                state['agent%s' % i]['model_state'] = self.agent_models[i].state_dict()
                state['agent%s' % i]['opt_state'] = self.agent_opts[i].state_dict() if self.agent_opts[i] is not None else None
        state['episodes'] = self.episodes_completed
#         state['elapsed_time'] = time.time() - self.start_time

        checkpoint_path = os.path.join(tmp_checkpoint_dir, 'model.pth')
        with open(checkpoint_path, 'wb') as f:
            torch.save(state, f)
        return tmp_checkpoint_dir

    def load_checkpoint(self, tmp_checkpoint_dir):
        checkpoint_path = os.path.join(tmp_checkpoint_dir, 'model.pth')
        with open(checkpoint_path, 'rb') as f:
            state = torch.load(f)
            
        for agent_i in range(2):
            if isinstance(state['agent%s' % i]['model_state'], list):
                for pop_i in range(len(self.agent_models[i])):
                    self.agent_models[agent_i][pop_i].load_state_dict(state['agent%s' % agent_i]['model_state'][pop_i])
                    if state['agent%s' % agent_i]['opt_state'][pop_i] is not None:
                        self.agent_opts[agent_i][pop_i].load_state_dict(state['agent%s' % agent_i]['opt_state'][pop_i])
            else:
                self.agent_models[agent_i].load_state_dict(state['agent%s' % agent_i]['model_state'])
                if state['agent%s' % agent_i]['opt_state'] is not None:
                    self.agent_opts[agent_i].load_state_dict(state['agent%s' % agent_i]['opt_state'])
        self.episodes_completed = state['episodes']
#         start_time = time.time() - state['elapsed_time']

def run_episode(args, env_state, agent_models, arbitrator_model, deterministic, render_flag, embedding_sizes):
    env_agents_state = AltOffersEnvMemory(env_state, msg_len=3, hidden_embedding_sizes=embedding_sizes, enable_arbitrator=(arbitrator_model is not None), enable_cuda=args['enable_cuda'], enable_binding_comm=args['enable_binding_comm'], enable_cheap_comm=args['enable_cheap_comm'])
    type_constr = torch.cuda if args['enable_cuda'] else torch
    max_batch_size = env_state.N.size()[0]

    logprobs_by_timestep = []
    alive_masks = []  # accumulator of indices of alive games on every iteration

    rewards = None
    if arbitrator_model is None:
        arb_logprobs_accum = None
        arb_entropy_loss_accum = None
    else:
        arb_entropy_loss_accum = type_constr.FloatTensor(1).fill_(0)
        arb_logprobs_accum = type_constr.FloatTensor(max_batch_size, 2).fill_(0)  # 2 decisions
        
    agent_forward_stats = [{}, {}]
    arb_forward_stats = {}
    
    entropy_loss_by_agent = [
        Variable(type_constr.FloatTensor(1).fill_(0)),
        Variable(type_constr.FloatTensor(1).fill_(0))
    ]
    
    for env_state.t in range(10):
        agent = env_state.get_agent()
        agent_model = agent_models[agent]
        
        own_message, opponent_message = env_agents_state.get_messages()
        own_proposal, opponent_proposal = env_agents_state.get_proposals()
        hidden_state, cell_state = env_agents_state.get_agent_states()
        
        action_logprobs, response, new_message, new_proposal, entropy_loss, new_hidden_state, new_cell_state, cur_forward_stats = agent_model(
            pool=Variable(env_state.pool),
            utility=Variable(env_state.utilities[:, agent]),
            own_proposal=own_proposal,
            own_message=own_message,
            opponent_proposal=opponent_proposal,
            opponent_message=opponent_message,
            hidden_state=hidden_state,
            cell_state=cell_state,
            deterministic=deterministic,
            timestep=env_state.t,
        )
        
        for key, val in cur_forward_stats.items():
            if key in agent_forward_stats[agent]:
                agent_forward_stats[agent][key] += val
            else:
                agent_forward_stats[agent][key] = val
                
        env_agents_state.update_agent_states(new_hidden_state, new_cell_state)
            
        entropy_loss_by_agent[agent] += entropy_loss
        # entries of action_logprobs correspond to decisions made for every element of the batch
        logprobs_by_timestep.append(action_logprobs)  # all action log-likelihoods
        
        if render_flag and env_agents_state.sieve.out_idxes[0] == 0:  # as long as the game with index 0 is still alive, render it
            logging_utils.render_action(
                s=env_state,
                response=response,
                prop=new_proposal,
                msg=new_message,
                enable_binding_comm=args['enable_binding_comm'],
                enable_cheap_comm=args['enable_cheap_comm'],
            )
        
        # backprop through time is accomplished via cell and hidden states, not via env states
        env_agents_state.update_message_proposal(new_message.detach(), new_proposal.detach())
        
        enable_decision = None
        favor_decision = None
        if arbitrator_model is not None:
            # use old opponent_proposal here
            arb_logprobs, enable_decision, favor_decision, arb_entropy_loss,\
            env_agents_state.arb_hidden_state, env_agents_state.arb_cell_state, cur_forward_stats = \
                arbitrator_model(env_state.pool, env_state.utilities[:, 0], env_state.utilities[:, 1], opponent_proposal, new_message,
                                 response, env_state.t, deterministic, env_agents_state.arb_hidden_state, env_agents_state.arb_cell_state)
            arb_logprobs_accum[env_agents_state.sieve.out_idxes] = torch.cat(arb_logprobs, 1)
            arb_entropy_loss_accum += arb_entropy_loss
            for key, val in cur_forward_stats.items():
                if key in arb_forward_stats:
                    arb_forward_stats[key] += val
                else:
                    arb_forward_stats[key] = val
    
        new_rewards = calc_rewards(
            env_agents_state=env_agents_state,
            response=response,
            enable_decision=enable_decision,
            favor_decision=favor_decision,
            enable_overflow=args['enable_overflow'],
            scale_before_redist=args['scale_before_redist'],
        )
        # rewards to all current games (0 if hasn't ended yet)
        # rewards vector is not sieved / reduced so we update it using out_idxes
        if rewards is None:
            rewards = {name: type_constr.FloatTensor(max_batch_size).fill_(0) for name in new_rewards}
        for name, val in new_rewards.items():
            rewards[name][env_agents_state.sieve.out_idxes] = new_rewards[name]
        
        cur_alive_mask, all_dead, num_turns = env_agents_state.update_sieve(response)
        # only num_turns from the last loop iteration with real number of turns for every game gets returned from run_episode()
        alive_masks.append(cur_alive_mask)  # cur_alive_masks all go into a list for playback during gradient computation
        if all_dead:
            break
        
    if render_flag:  # render game with index 0 in the full, unsieved, batch
        logging_utils.render_rewards(rewards)

    return logprobs_by_timestep, entropy_loss_by_agent, arb_logprobs_accum, arb_entropy_loss_accum, \
        rewards, num_turns, alive_masks, agent_forward_stats, arb_forward_stats
