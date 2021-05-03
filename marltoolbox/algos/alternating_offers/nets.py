##########
# Part of the code modified from:
# https://github.com/asappresearch/emergent-comms-negotiation
##########

import torch
from torch import nn, autograd
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np


eps = 1e-6

class NumberSequenceEncoder(nn.Module):
    def __init__(self, num_values, embedding_size=100):
        super().__init__()
        self.embedding_size = embedding_size
        self.num_values = num_values
        self.embedding = nn.Embedding(num_values, embedding_size)  # learnable lookup table for lookup of each of num_values elements
        self.lstm = nn.LSTMCell(input_size=embedding_size, hidden_size=embedding_size)

    def forward(self, x):
        batch_size = x.size()[0]
        seq_len = x.size()[1]
        x = x.transpose(0, 1)  # now 0 is sequence dim, 1 is batch dim - they've been transposed
        x = self.embedding(x)
        type_constr = torch.cuda if x.is_cuda else torch
        state = (
                Variable(type_constr.FloatTensor(batch_size, self.embedding_size).fill_(0)),
                Variable(type_constr.FloatTensor(batch_size, self.embedding_size).fill_(0))
            )  # hidden state, cell state - initialized with zeros to form a clean-slate LSTM
        for s in range(seq_len):
            state = self.lstm(x[s], state)  # takes the whole batch but only at embedded timestep s
        return state[0]  # hidden state of size (batch_size, embedding_size)


class TermPolicy(nn.Module):
    def __init__(self, embedding_size=100):
        super().__init__()
        self.h1 = nn.Linear(embedding_size, 1)

    def forward(self, thoughtvector, testing):
        logits = self.h1(thoughtvector)
        response_probs = torch.clamp(F.sigmoid(logits), eps, 1-eps)  # acceptance probabilities
        stochastic_draws = 0

        res_greedy = (response_probs.data >= 0.5).view(-1, 1).float()

        log_g = None
        if not testing:
            a = torch.bernoulli(response_probs)  # sample a decision for the batch elems
            g = a.detach() * response_probs + (1 - a.detach()) * (1 - response_probs)
            log_g = g.log()
            a = a.data
        else:
            a = res_greedy
            
        stochastic_draws += thoughtvector.size()[0]

        matches_greedy = (res_greedy == a)
        matches_greedy_count = matches_greedy.int().sum()
#         response_probs = response_probs + eps
        entropy = - (response_probs * response_probs.log() + (1-response_probs)*(1-response_probs).log()).sum()
        # for the batch:
        # probs of acceptance, probs of sampled decisions, sampled decisions, bernoulli entropies, how many decisions are argmaxes in batch
        return response_probs, log_g, a.byte(), entropy, matches_greedy_count, stochastic_draws


class ProposalPolicy(nn.Module):
    def __init__(self, embedding_size=100, num_counts=6, num_items=3):
        super().__init__()
        self.num_counts = num_counts
        self.num_items = num_items
        self.embedding_size = embedding_size
        self.fcs = []
        for i in range(num_items):
            fc = nn.Linear(embedding_size, num_counts)  # linear for each item
            self.fcs.append(fc)
            self.__setattr__('h1_%s' % i, fc)

    def forward(self, x, testing):
        batch_size = x.size()[0]
        nodes = []
        entropy = 0
        matches_argmax_count = 0
        type_constr = torch.cuda if x.is_cuda else torch
        stochastic_draws = 0
        proposal = type_constr.LongTensor(batch_size, self.num_items).fill_(0)
        for i in range(self.num_items):
            logits = self.fcs[i](x)
            probs = torch.clamp(F.softmax(logits), eps, 1-eps)

            _, res_greedy = probs.data.max(1)
            res_greedy = res_greedy.view(-1, 1).long()

            log_g = None
            if not testing:
                a = torch.multinomial(probs, num_samples=1)
                g = torch.gather(probs, 1, Variable(a.data))  # place probs on indices specified in samples
                log_g = g.log()
                a = a.data
            else:
                a = res_greedy

            matches_argmax = res_greedy == a
            matches_argmax_count += matches_argmax.int().sum()
            stochastic_draws += batch_size

            if log_g is not None:
                nodes.append(log_g)
#             probs = probs + eps
            entropy += (- probs * probs.log()).sum()  # probs are from softmax so there's the required sum from the entropy formula
            proposal[:, i] = a[:, 0]

        return nodes, proposal, entropy, matches_argmax_count, stochastic_draws
    
class ProposalReprPolicy(nn.Module):
    def __init__(self, embedding_size=100, num_counts=6, num_items=3):
        super().__init__()
        self.num_counts = num_counts
        self.num_items = num_items
        self.embedding_size = embedding_size
        self.fcs = []
        for i in range(num_items):
            # linear for each item. Takes the same embedding (size 100) plus onehot of the same item in the hidden proposal
            fc = nn.Linear(embedding_size+num_counts, num_counts)
            self.fcs.append(fc)
            self.__setattr__('h1_%s' % i, fc)

    def forward(self, x, hidden_proposal, testing):
        batch_size = x.size()[0]
        nodes = []
        entropy = 0
        matches_argmax_count = 0
        type_constr = torch.cuda if x.is_cuda else torch
        stochastic_draws = 0
        hidden_proposal_onehot = torch.zeros(batch_size, self.num_items, self.num_counts)
        if x.is_cuda:
            hidden_proposal_onehot = hidden_proposal_onehot.cuda()
        hidden_proposal_onehot.scatter_(2, hidden_proposal.unsqueeze(2), 1)  # dim, index, src val
        proposal = type_constr.LongTensor(batch_size, self.num_items).fill_(0)  # new public proposal
        
        for i in range(self.num_items):
            cur_item_hidden_proposal = hidden_proposal_onehot[:, i, :]
            logits = self.fcs[i](torch.cat([x, cur_item_hidden_proposal], dim=1))
            probs = torch.clamp(F.softmax(logits), eps, 1-eps)

            _, res_greedy = probs.data.max(1)
            res_greedy = res_greedy.view(-1, 1).long()

            log_g = None
            if not testing:
                a = torch.multinomial(probs, num_samples=1)
                g = torch.gather(probs, 1, Variable(a.data))
                log_g = g.log()
                a = a.data
            else:
                a = res_greedy

            matches_argmax = (res_greedy == a)
            matches_argmax_count += matches_argmax.int().sum()
            stochastic_draws += batch_size

            if log_g is not None:
                nodes.append(log_g)
#             probs = probs + eps
            entropy += (- probs * probs.log()).sum()  # probs are from softmax so there's the required sum from the entropy formula
            proposal[:, i] = a[:, 0]

        return nodes, proposal, entropy, matches_argmax_count, stochastic_draws
    
class RedistributionPolicy(nn.Module):
    def __init__(self, embedding_size=100):
        super().__init__()
        self.enable_redist = nn.Linear(embedding_size, 1)
        self.favor_first_player = nn.Linear(embedding_size, 1)

    def forward(self, thoughtvector, testing, mid_move_indices):
        enable_probs = torch.clamp(F.sigmoid(self.enable_redist(thoughtvector)), eps, 1-eps)
        favor_probs = torch.clamp(F.sigmoid(self.favor_first_player(thoughtvector)), eps, 1-eps)
        

        enable_decision = torch.bernoulli(enable_probs).long()  # sample a decision for the batch elems
        enable_argmax = (enable_decision == (enable_probs > 0.5).long())
        enable_logprob = (enable_probs * enable_decision.float().detach() + (1 - enable_probs) * (1 - enable_decision.float().detach())).log()
        enable_entropy = - enable_probs*enable_probs.log() - (1-enable_probs)*(1-enable_probs).log()
        
        enable_decision[mid_move_indices] = 0
        enable_argmax[mid_move_indices] = 0
        enable_logprob[mid_move_indices] = 0
        enable_entropy[mid_move_indices] = 0
        # to save probs but also avoid backward problems from in-place computation
        enable_probs_clone = enable_probs.clone()
        enable_probs_clone[mid_move_indices] = 0
        enable_argmax_count = enable_argmax.sum().item()
        enable_entropy = enable_entropy.sum()
        
        
        favor_decision = torch.bernoulli(favor_probs).long()
        favor_argmax = (favor_decision == (favor_probs > 0.5).long())
        favor_logprob = (favor_probs * favor_decision.float().detach() + (1 - favor_probs) * (1 - favor_decision.float().detach())).log()
        favor_entropy = - favor_probs*favor_probs.log() - (1-favor_probs)*(1-favor_probs).log()

        favor_decision[enable_decision == 0] = 0
        favor_argmax[enable_decision == 0] = 0
        favor_logprob[enable_decision == 0] = 0
        favor_entropy[enable_decision == 0] = 0
        # to save probs but also avoid backward problems from in-place computation
        favor_probs_clone = favor_probs.clone()
        favor_probs_clone[enable_decision == 0] = 0
        favor_argmax_count = favor_argmax.sum().item()
        favor_entropy = favor_entropy.sum()
        
        return enable_logprob, enable_probs_clone, enable_decision.data.byte(), enable_entropy, favor_logprob, favor_probs_clone, favor_decision.data.byte(), favor_entropy, enable_argmax_count, favor_argmax_count, thoughtvector.size(0)-len(mid_move_indices), enable_decision.sum().item()


class AgentModel(nn.Module):
    def __init__(
            self, enable_binding_comm, enable_cheap_comm,
            response_entropy_reg,
            utterance_entropy_reg,
            proposal_entropy_reg,
            hidden_embedding_size=30,):
        super().__init__()
        self.response_entropy_reg = response_entropy_reg
        self.utterance_entropy_reg = utterance_entropy_reg
        self.proposal_entropy_reg = proposal_entropy_reg
        self.hidden_embedding_size = hidden_embedding_size
        self.enable_binding_comm = enable_binding_comm  # ignored here, the proposal is predicted but then blocked in ecn.py
        self.enable_cheap_comm = enable_cheap_comm
        
        self.lstm = nn.LSTMCell(
            input_size=hidden_embedding_size,
            hidden_size=hidden_embedding_size)
        
        self.combined_net = nn.Sequential(nn.Linear(19, hidden_embedding_size), nn.ReLU())
#         self.combined_net = nn.Sequential(nn.Linear(19, hidden_embedding_size), nn.ReLU(), nn.Linear(hidden_embedding_size, hidden_embedding_size), nn.ReLU())

        self.response_policy = TermPolicy(embedding_size=hidden_embedding_size)
        self.proposal_policy = ProposalPolicy(embedding_size=hidden_embedding_size)
        if self.enable_cheap_comm:
            self.proposal_repr_policy = ProposalReprPolicy(embedding_size=hidden_embedding_size)

    def forward(self, pool, utility, own_proposal, own_message,
                opponent_proposal, opponent_message, hidden_state, cell_state, deterministic, timestep):
        if deterministic:
            raise NotImplementedError  # disabled this for the time being because evaluating models with stochastic actions makes a bit more sense
        forward_stats = {}
        
        batch_size = pool.size()[0]
        type_constr = torch.cuda if pool.is_cuda else torch
        
        timestep_formatted = np.reshape(np.repeat(np.array([timestep]), batch_size), (batch_size, 1))
        timestep_formatted = torch.from_numpy(timestep_formatted).float()
        if pool.is_cuda:
            timestep_formatted.cuda()

        h_t = torch.cat([ten.float() for ten in [pool, utility, own_proposal, own_message, opponent_proposal, opponent_message, timestep_formatted]], -1)  # (static game context, utterance, proposal)
        h_t = self.combined_net(h_t)  # act on (static game context, utterance, proposal) with linear and relu
        
        hidden_state, cell_state = self.lstm(h_t, (hidden_state, cell_state))  # takes the whole batch but only at embedded timestep s
        h_t = hidden_state
        
        entropy_loss = 0
        nodes = []

        # probs of acceptance, probs of sampled decisions, sampled decisions, bernoulli entropies, how many decisions are argmaxes in batch, num of decisions
        response_probs, response_node, response, response_entropy, response_matches_argmax_count, forward_stats['response_stochastic_draws'] = self.response_policy(h_t, testing=deterministic)
        forward_stats['response_prob'] = response_probs.sum().item()
        forward_stats['response_entropy'] = response_entropy.item()
        forward_stats['response_matches_argmax_count'] = response_matches_argmax_count.sum().item()
        nodes.append(response_node)
        entropy_loss -= self.response_entropy_reg * response_entropy  # maximize entropy so minimize loss ~ (-entropy)

        proposal_nodes, proposal, prop_entropy, prop_matches_argmax_count, forward_stats['prop_stochastic_draws'] = self.proposal_policy(h_t, testing=deterministic)
        forward_stats['prop_entropy'] = prop_entropy.item()
        forward_stats['prop_matches_argmax_count'] = prop_matches_argmax_count.sum().item()
        nodes += proposal_nodes
        entropy_loss -= self.proposal_entropy_reg * prop_entropy

        utterance = None
        if self.enable_cheap_comm:
            utterance_nodes, utterance, utt_entropy, utt_matches_argmax_count, forward_stats['utt_stochastic_draws'] = self.proposal_repr_policy(h_t, proposal, testing=deterministic)
            forward_stats['utt_entropy'] = utt_entropy.item()
            forward_stats['utt_matches_argmax_count'] = utt_matches_argmax_count.sum().item()
            nodes += utterance_nodes
            entropy_loss -= self.utterance_entropy_reg * utt_entropy
        else:
            forward_stats['utt_entropy'] = 0
            forward_stats['utt_matches_argmax_count'] = 0
            forward_stats['utt_stochastic_draws'] = 0
            utterance = type_constr.LongTensor(batch_size, 3).zero_()

        return nodes, response, utterance, proposal, entropy_loss, hidden_state, cell_state, forward_stats


class ArbitratorModel(nn.Module):
    def __init__(self,
                 entropy_reg,
                 share_utilities,
                 hidden_embedding_size=30,):
        super().__init__()
        
        self.entropy_reg = entropy_reg
        self.share_utilities = share_utilities
        self.hidden_embedding_size = hidden_embedding_size

        input_size = 17 if self.share_utilities else 11
        print(input_size)
        self.combined_net = nn.Sequential(nn.Linear(input_size, hidden_embedding_size), nn.ReLU())
#         self.combined_net = nn.Sequential(nn.Linear(input_size, hidden_embedding_size), nn.ReLU(), nn.Linear(hidden_embedding_size, hidden_embedding_size), nn.ReLU())

        self.lstm = nn.LSTMCell(
            input_size=hidden_embedding_size,
            hidden_size=hidden_embedding_size)
    
        self.redist_policy = RedistributionPolicy(hidden_embedding_size)

    def forward(self, pool, utilities0, utilities1, proposal, message, game_finished, timestep, deterministic, hidden_state, cell_state):
        if deterministic:
            raise NotImplementedError  # disabled this for the time being because evaluating models with stochastic actions makes a bit more sense
        forward_stats = {}
        
        batch_size = pool.size()[0]
        final_move_indices = game_finished[:, 0].nonzero()
        mid_move_indices = (1-game_finished[:, 0]).nonzero()
        
        type_constr = torch.cuda if pool.is_cuda else torch
        msg_encoded = message.clone() # encoding the prev message
        prop_encoded = proposal.clone()  # encoding final proposal with same net if it's non-zero
        
        timestep_formatted = np.reshape(np.repeat(np.array([timestep]), batch_size), (batch_size, 1))
        timestep_formatted = torch.from_numpy(timestep_formatted)
        if pool.is_cuda:
            timestep_formatted.cuda()
        
        prop_encoded[mid_move_indices, :] = 0  # no access to the proposal if game hasn't ended
        msg_encoded[final_move_indices, :] = 0  # the final message doesn't get through
        
        if self.share_utilities:
            input_tens = [pool, utilities0, utilities1, prop_encoded, msg_encoded, game_finished, timestep_formatted]
        else:
            input_tens = [pool, prop_encoded, msg_encoded, game_finished, timestep_formatted]
        h_t = torch.cat([ten.float() for ten in input_tens], -1)  # (static game context, utterance, proposal)
        h_t = self.combined_net(h_t)  # act on (static game context, utterance, proposal) with linear and relu
        
        hidden_state, cell_state = self.lstm(h_t, (hidden_state, cell_state))  # takes the whole batch but only at embedded timestep s
        h_t = hidden_state

        enable_logprob, forward_stats['enable_probs'], enable_decision, enable_entropy, \
            favor_logprob, forward_stats['favor_probs'], favor_decision, favor_entropy, \
            forward_stats['enable_argmax_count'], forward_stats['favor_argmax_count'], \
            forward_stats['enable_draws'], forward_stats['favor_draws'] = \
            self.redist_policy(h_t, testing=deterministic, mid_move_indices=mid_move_indices)
        forward_stats['enable_probs'] = forward_stats['enable_probs'].sum().item()
        forward_stats['enable_decision'] = enable_decision.sum().item()
        forward_stats['enable_entropy'] = enable_entropy.item()
        forward_stats['favor_probs'] = forward_stats['favor_probs'].sum().item()
        forward_stats['favor_decision'] = favor_decision.sum().item()
        forward_stats['favor_entropy'] = favor_entropy.item()
        nodes = [enable_logprob, favor_logprob]
        entropy_loss = -(enable_entropy + favor_entropy) * self.entropy_reg

        return nodes, enable_decision, favor_decision, entropy_loss, hidden_state, cell_state, forward_stats
