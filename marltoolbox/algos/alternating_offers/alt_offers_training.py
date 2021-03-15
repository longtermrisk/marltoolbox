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
import torch
from torch import autograd, optim, nn
from collections import defaultdict
from torch.autograd import Variable
import torch.nn.functional as F

import marltoolbox.algos.alternating_offers.nets as nets
import marltoolbox.algos.alternating_offers.logging_utils as logging_utils
from marltoolbox.algos.alternating_offers.envs.alt_offers_env import AltOffersEnv, AltOffersEnvMemory, calc_rewards
from marltoolbox.algos.alternating_offers.envs.alt_offers_env_sieve import SievePlayback

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

        self.agent_models = []
        self.agent_opts = []
        for i in range(2):
            model = nets.AgentModel(
                enable_binding_comm=args['enable_binding_comm'],
                enable_cheap_comm=args['enable_cheap_comm'],
                response_entropy_reg=args['response_entropy_reg'],
                utterance_entropy_reg=args['utterance_entropy_reg'],
                proposal_entropy_reg=args['proposal_entropy_reg'],
                hidden_embedding_size=args['hidden_embedding_size'],
            )
            if args['enable_cuda']:
                model.cuda()
            self.agent_models.append(model)
            if args['agents_sgd']:
                self.agent_opts.append(torch.optim.SGD(self.agent_models[i].parameters(), lr=1e-3, weight_decay=1e-4))
            else:
                self.agent_opts.append(optim.Adam(params=self.agent_models[i].parameters()))

        if args['enable_arbitrator']:
            self.arbitrator_model = nets.ArbitratorModel(entropy_reg=args['arbitrator_entropy_reg'], share_utilities=args['share_utilities'],
                                                     hidden_embedding_size=args['hidden_embedding_size'])
            if args['enable_cuda']:
                self.arbitrator_model.cuda()
            if args['arbitrator_sgd']:
                self.arbitrator_opt = torch.optim.SGD(self.arbitrator_model.parameters(), lr=1e-3, weight_decay=1e-4)
            else:
                self.arbitrator_opt = optim.Adam(params=self.arbitrator_model.parameters())

        self.last_print = time.time()
        for d in ['logs', 'model_saves']:
            if not path.isdir(d):
                os.makedirs(d)

        self.baseline = None
    
    def step(self):
        '''Run 1 episode (1 batch of games) and then update the parameters
        '''
#         by guidelines, should be more than a few seconds but also short enough to allow reporting
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
            env_state = AltOffersEnv(batch_size=self.args['batch_size'], random_state=self.train_random_state, utility_type=self.args['utility_type'])

            action_logprobs, entropy_loss_by_agent, self.arb_logprobs_accum, arb_entropy_loss, rewards, num_turns, alive_masks, \
            self.new_agent_forward_stats, self.new_arb_forward_stats = run_episode(
                args=self.args,
                env_state=env_state,
                agent_models=self.agent_models,
                arbitrator_model=self.arbitrator_model if self.args['enable_arbitrator'] else None,
                render_flag=(cur_episode_i == self.args['episodes_per_step']-1),
                deterministic=False,
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
                    for group in self.agent_opts[i].param_groups:
                        group['lr'] = cur_sgd_lr
            if self.args['enable_arbitrator'] and self.args['arbitrator_sgd']:
                for group in self.arbitrator_opt.param_groups:
                    group['lr'] = cur_sgd_lr

            if self.args['enable_arbitrator']:
                for i in range(2):
                    self.agent_opts[i].zero_grad()

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
                    welfare_grads.append(torch.autograd.grad(orig_welfare_reward_by_agent[i], self.agent_models[i].parameters(), retain_graph=True))

                self.arbitrator_opt.zero_grad()
                for i in range(2):
                    self.agent_opts[i].zero_grad()
                induced_reward_by_agent = [0, 0]
                induced_grads = []
                sieve_playback = SievePlayback(alive_masks, enable_cuda=self.args['enable_cuda'])
                for t, global_idxes in sieve_playback:  # iterate over which games were active on every step (for each batch elem)
                    agent = t % 2
                    for action in action_logprobs[t]:  # iterate over 7 action_logprobs: 1xresponse, 3xproposal, 3xproposal_repr
                        reward_name = 'player0_arb_induced_part' if agent == 0 else 'player1_arb_induced_part'
                        induced_agent_reward = baselined_rewards[reward_name][global_idxes].float().contiguous().view(sieve_playback.batch_size, 1)
                        cur_arb_logprobs = self.arb_logprobs_accum.sum(1)[global_idxes].view(sieve_playback.batch_size, 1)
                        induced_reward_by_agent[agent] += (action * cur_arb_logprobs * Variable(induced_agent_reward)).sum()
                for i in range(2):
                    # should account for maxent exploration bonus of each agent here too! but not in the orig_welfare_reward_by_agent.
                    cur_grads = torch.autograd.grad(induced_reward_by_agent[i]-entropy_loss_by_agent[i], self.agent_models[i].parameters(), retain_graph=True, create_graph=True)
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
                self.agent_opts[i].zero_grad()
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
                    assert self.args['scale_before_redist'] or self.args['prosociality_level']==0
                    total_agent_reward = (1-self.args['prosociality_level'])*baselined_rewards[reward_name][global_idxes] + \
                                          self.args['prosociality_level']*baselined_rewards['sum_share_of_max'][global_idxes]

                    total_agent_reward = total_agent_reward.float().contiguous().view(sieve_playback.batch_size, 1)
                    main_loss_by_agent[agent] += -(action * Variable(total_agent_reward)).sum()

            for i in range(2):
                # rewards can be negative here because of the baseline
                # entropy_loss is negative bc entropy is positive
                loss = entropy_loss_by_agent[i] + main_loss_by_agent[i]
                loss.backward()
                self.agent_opts[i].step()

                self.main_loss_by_agent_accum[i] += main_loss_by_agent[i].item()
                self.entropy_loss_by_agent_accum[i] += entropy_loss_by_agent[i].item()


            self.num_turns_accum += num_turns.sum()
            for name in rewards:
                self.rewards_sum[name] += rewards[name].sum()
                self.baseline[name] = 0.7*self.baseline[name] + 0.3*rewards[name].mean()
            self.processed_games_sum += self.args['batch_size']
            self.episodes_completed += 1

        log_data = logging_utils.get_step_log(self)
        logging_utils.print_step_log(self, log_data)
        return log_data

    def save_checkpoint(self, tmp_checkpoint_dir):
        state = {}
        for i in range(2):
            state['agent%s' % i] = {}
            state['agent%s' % i]['model_state'] = self.agent_models[i].state_dict()
            state['agent%s' % i]['opt_state'] = self.agent_opts[i].state_dict()
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
            
        for i in range(2):
            self.agent_models[i].load_state_dict(state['agent%s' % i]['model_state'])
            self.agent_opts[i].load_state_dict(state['agent%s' % i]['opt_state'])
        self.episodes_completed = state['episodes']
#         start_time = time.time() - state['elapsed_time']
        pass

def run_episode(args, env_state, agent_models, arbitrator_model, deterministic, render_flag):
    env_agents_state = AltOffersEnvMemory(env_state, msg_len=3, hidden_embedding_size=args['hidden_embedding_size'], enable_arbitrator=(arbitrator_model is not None), enable_cuda=args['enable_cuda'], enable_binding_comm=args['enable_binding_comm'], enable_cheap_comm=args['enable_cheap_comm'])
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
