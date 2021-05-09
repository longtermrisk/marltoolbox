##########
# Part of the code modified from:
# https://github.com/asappresearch/emergent-comms-negotiation
##########

import torch
from marltoolbox.algos.alternating_offers.envs.alt_offers_env_sampling import generate_batch
from marltoolbox.algos.alternating_offers.envs.alt_offers_env_sieve import AliveSieve


class AltOffersEnv(object):
    '''State of the environment (max lengths of games, numbers of pool items, reward values for different items)
    '''
    def __init__(self, batch_size, random_state, utility_types):
        # sample game parameters for a new batch
        batch_properties = generate_batch(batch_size=batch_size, random_state=random_state, utility_types=utility_types)
        
        self.pool = batch_properties['pool']
        self.N = batch_properties['N']
        self.max_batch_size = self.N.size()[0]  # actual batch size will be smaller on some steps because games end early
        self.utilities = torch.zeros(self.max_batch_size, 2, 3).long()
        self.utilities[:, 0, :] = batch_properties['utilities'][0]  # dimensions: batch size, player_i, item_i
        self.utilities[:, 1, :] = batch_properties['utilities'][1]
        
        self.t = 0  # timestep
        # here we store the final number of turns for each game which is 10 by default but then decreases dynamically upon games finishing
        self.cur_num_turns = torch.LongTensor(self.max_batch_size).fill_(10)

    def cuda(self):
        self.N = self.N.cuda()
        self.pool = self.pool.cuda()
        self.utilities = self.utilities.cuda()
        self.cur_num_turns = self.cur_num_turns.cuda()

    def apply_sieve(self, still_alive_idxes):
        '''Reduce the batch to only contain games active on current timestep
        '''
        self.N = self.N[still_alive_idxes]
        self.pool = self.pool[still_alive_idxes]
        self.utilities = self.utilities[still_alive_idxes]
        
    def get_agent(self):
        '''Get an index of the player whose turn it is to act on a given timestep (actions of different players are formally sequential)
        '''
        return self.t % 2


class AltOffersEnvMemory(object):
    '''In this class, the states of the players and the arbitrator are stored
    '''
    def __init__(self, alt_offers_env, msg_len, hidden_embedding_sizes, enable_arbitrator, enable_cuda,
                 enable_binding_comm, enable_cheap_comm):
        self.env_state = alt_offers_env
        max_batch_size = self.env_state.max_batch_size
        
        self.proposal_0 = torch.zeros(max_batch_size, 3).long()  # binding proposal
        self.proposal_1 = torch.zeros(max_batch_size, 3).long()
        self.message_0 = torch.zeros(max_batch_size, msg_len).long()  # cheap utterance
        self.message_1 = torch.zeros(max_batch_size, msg_len).long()
        self.hidden_state_0 = torch.zeros(max_batch_size, hidden_embedding_sizes[0])
        self.hidden_state_1 = torch.zeros(max_batch_size, hidden_embedding_sizes[1])
        self.cell_state_0 = torch.zeros(max_batch_size, hidden_embedding_sizes[0])
        self.cell_state_1 = torch.zeros(max_batch_size, hidden_embedding_sizes[1])
        
        self.enable_arbitrator = enable_arbitrator
        if self.enable_arbitrator:
            self.arb_hidden_state = torch.zeros(max_batch_size, hidden_embedding_sizes[2])
            self.arb_cell_state = torch.zeros(max_batch_size, hidden_embedding_sizes[2])
            
        self.sieve = AliveSieve(batch_size=max_batch_size, enable_cuda=enable_cuda)
        
        self.enable_binding_comm = enable_binding_comm
        self.enable_cheap_comm = enable_cheap_comm
        
        if enable_cuda:
            self.cuda()
            self.type_constr = torch.cuda
        else:
            self.type_constr = torch
            
    def get_messages(self):
        if self.enable_cheap_comm:
            agent = self.env_state.get_agent()
            own_message = self.message_0 if agent == 0 else self.message_1
            opponent_message = self.message_1 if agent == 0 else self.message_0
        else:
            own_message = self.type_constr.LongTensor(self.sieve.batch_size, 3).fill_(0)  # zero fillers
            opponent_message = self.type_constr.LongTensor(self.sieve.batch_size, 3).fill_(0)
        return own_message, opponent_message
    
    def get_proposals(self):
        '''Returns publicly known proposals for a given timestep and for a given batch
        Not suitable for calculating rewards using privately known proposals
        '''
        if self.enable_binding_comm:
            agent = self.env_state.get_agent()
            own_proposal = self.proposal_0 if agent == 0 else self.proposal_1
            opponent_proposal = self.proposal_1 if agent == 0 else self.proposal_0
        else:
            own_proposal = self.type_constr.LongTensor(self.sieve.batch_size, 3).fill_(0)  # zero fillers
            opponent_proposal = self.type_constr.LongTensor(self.sieve.batch_size, 3).fill_(0)
        return own_proposal, opponent_proposal
    
    def get_agent_states(self):
        agent = self.env_state.get_agent()
        hidden_state = self.hidden_state_0 if agent == 0 else self.hidden_state_1
        cell_state = self.cell_state_0 if agent == 0 else self.cell_state_1
        return hidden_state, cell_state
    
    def update_agent_states(self, new_hidden_state, new_cell_state):
        agent = self.env_state.get_agent()
        if agent == 0:
            self.hidden_state_0 = new_hidden_state
            self.cell_state_0 = new_cell_state
        else:
            self.hidden_state_1 = new_hidden_state
            self.cell_state_1 = new_cell_state
            
    def update_message_proposal(self, new_message, new_proposal):
        agent = self.env_state.get_agent()
        if agent == 0:
            self.message_0 = new_message 
            self.proposal_0 = new_proposal
        else:
            self.message_1 = new_message
            self.proposal_1 = new_proposal
    
    def cuda(self):
        self.env_state = self.env_state.cuda()
        
        self.proposal_0 = self.proposal_0.cuda()
        self.proposal_1 = self.proposal_1.cuda()
        self.message_0 = self.message_0.cuda()
        self.message_1 = self.message_1.cuda()
        
        self.hidden_state_0 = self.hidden_state_0.cuda()
        self.hidden_state_1 = self.hidden_state_1.cuda()
        self.cell_state_0 = self.cell_state_0.cuda()
        self.cell_state_1 = self.cell_state_1.cuda()
        
        if self.enable_arbitrator:
            self.arb_hidden_state = self.arb_hidden_state.cuda()
            self.arb_cell_state = self.arb_cell_state.cuda()
            
    def update_sieve(self, response):
        '''Tracks whether some games have ended early and removes games that have ended from the batch
        Inputs 
        
        Returns:
        1) cur_alive_mask - indices of games that are active for the current timestep, used for playback during gradient computation
        2) all_dead - binary, if all games have ended or not; used to end the episode rollout if True
        3) cur_num_turns - num of turns of each game, real if ended and maximum if in progress
        '''
        t = self.env_state.t  # timestep
        self.sieve.mark_dead(response)  # if a proposal was accepted, remove the game
        self.sieve.mark_dead(t + 1 >= self.env_state.N)  # if the time is out for a game, also remove the game
        # remember which games are active for current timestep, used for playback during gradient computation
        cur_alive_mask = self.sieve.alive_mask.clone()
        self.sieve.set_dead_global(self.env_state.cur_num_turns, t + 1)  # assign a length of t+1 to those games that have just ended
        all_dead = self.sieve.all_dead()  # binary, if all games have ended or not; used to stop the episode rollout
        
        self.apply_sieve()  # remove dead games from the state
        self.sieve.remove_dead()  # remove dead games from sieve which tracks which games are alive
        
        return cur_alive_mask, all_dead, self.env_state.cur_num_turns
            
    def apply_sieve(self):
        still_alive_idxes = self.sieve.alive_idxes
        
        self.env_state.apply_sieve(still_alive_idxes)
        
        self.proposal_0 = self.proposal_0[still_alive_idxes]
        self.proposal_1 = self.proposal_1[still_alive_idxes]
        self.message_0 = self.message_0[still_alive_idxes]
        self.message_1 = self.message_1[still_alive_idxes]
        
        self.hidden_state_0 = self.hidden_state_0[still_alive_idxes]
        self.hidden_state_1 = self.hidden_state_1[still_alive_idxes]
        self.cell_state_0 = self.cell_state_0[still_alive_idxes]
        self.cell_state_1 = self.cell_state_1[still_alive_idxes]
        
        if self.enable_arbitrator:
            self.arb_hidden_state = self.arb_hidden_state[still_alive_idxes]
            self.arb_cell_state = self.arb_cell_state[still_alive_idxes]

def calc_rewards(env_agents_state, response, enable_overflow, scale_before_redist, enable_decision=None, favor_decision=None):
    '''Calculate rewards for a batch of games on a given timestep
    
    Parameters:
    env_agents_state - current states of the environment and the agents for each game in the batch
    response - decisions of the current player about whether to accept the proposal for each game in the batch
    enable_overflow - binary, True if game if modified to destroy items of a particular type if an agreement is not reached for that type
    scale_before_redist - binary, True if normalization of the rewards should take place before their redistribution by the arbitrator (not relevant if an arbitrator is not present)
    enable_decision - whether redistribution happens on a given timestep for each game in the batch (not relevant if an arbitrator is not present)
    favor_decision - True if redistributing rewards to 0th agent for each game in the batch (not relevant if an arbitrator is not present)
    
    Returns:
    rewards_final - a dict {str reward_type : FloatTensor rewards for the batch of games}
    '''
    s = env_agents_state.env_state
    t = s.t  # timestep
    agent = s.get_agent()  # agent whose turn it is to act on a given timestep

    batch_size = response.size()[0]
    utility = s.utilities[:, agent]
    type_constr = torch.cuda if s.pool.is_cuda else torch

    rewards_final = {}
    # player0_share_of_max means the ratios of rewards that player 0 obtains to maximum rewards it could theoretically obtain on the current batch
    # sum_raw is utilitarian welfare (before redistribution)
    # sum_share_of_max is utilitarian welfare after normalizing (before redistribution)
    # player0_after_arb means redistributed rewards
    # player0_arb_induced_part means the difference between the redistributed rewards and the original rewards
    reward_names = ['player0_raw', 'player1_raw', 'sum_raw', 'player0_share_of_max', 'player1_share_of_max', 'sum_share_of_max',
                    'difference_in_share_of_max']
    if enable_decision is not None:
        reward_names += ['player0_after_arb', 'player1_after_arb', 'player0_arb_induced_part', 'player1_arb_induced_part']
    for name in reward_names:
        rewards_final[name] = type_constr.FloatTensor(batch_size).fill_(0)
        
    if t == 0:
        # return zero rewards on the first timestep, because there are no proposals to accept
        # non-zero rewards are only assigned if a proposal is accepted
        return rewards_final

    reward_eligible_mask = response.view(batch_size).clone().byte()
    if reward_eligible_mask.max() == 0:
        # if none of the games have accepted proposals on a given timestep, return zero rewards everywhere
        return rewards_final

    cur_proposal = env_agents_state.proposal_0 if agent==0 else env_agents_state.proposal_1
    prev_proposal = env_agents_state.proposal_0 if agent==1 else env_agents_state.proposal_1  # on previous turn
    
    # checking if the proposed distribution of items falls outside the range permissible by the pool (then return zero rewards)
    exceeded_pool, _ = ((prev_proposal - s.pool) > 0).max(1)
    if exceeded_pool.max() > 0:
        reward_eligible_mask[exceeded_pool.nonzero().long().view(-1)] = 0  # set the finishing flags on these games
        if reward_eligible_mask.max() == 0:
            # in all games agents have violated the pool
            return rewards_final  # return zeros

    proposer = 1 - agent  # agent on the previous turn
    acceptor = agent
    proposal = torch.zeros(batch_size, 2, 3).long()
    proposal[:, proposer] = prev_proposal  # proposal is for the player who proposed
    if not enable_overflow:
        proposal[:, acceptor] = s.pool - prev_proposal  # (pool-proposal) is for the other player
    else:
        proposal[:, acceptor] = cur_proposal
    max_utility, _ = s.utilities.max(1)  # maximum possible utility for each item, take max over players

    reward_eligible_idxes = reward_eligible_mask.nonzero().long().view(-1)
    for b in reward_eligible_idxes:  # iterate over indices of finished games
        raw_rewards = torch.FloatTensor(2).fill_(0)
        scaled_rewards = torch.FloatTensor(2).fill_(0)
        max_theoretical_rewards = []
        # max_agreement_rewards is like max_theoretical_rewards but with deduction of item types where an agreement was not reached
        max_agreement_rewards = []  # only used if the enable_overflow variant flag is True
        
        for player_i in range(2):
            max_agreement_r = 0.  # the maximum an agent can obtain given that an agreement was not reached on some items
            if not enable_overflow:
                raw_rewards[player_i] = s.utilities[b, player_i].cpu().dot(proposal[b, player_i].cpu())
            else:
                for item_i in range(3):
                    if cur_proposal[b, item_i] + prev_proposal[b, item_i] <= s.pool[b, item_i]:
                        raw_rewards[player_i] += proposal[b, player_i, item_i].item() * s.utilities[b, player_i, item_i].item()
                        # only count items where an agreement was reached in the maximum theoretical rewards sum
                        max_agreement_r += s.pool[b, item_i].item() * s.utilities[b, player_i, item_i].item()
                        
            max_theoretical_r = s.utilities[b, player_i].cpu().dot(s.pool[b].cpu())  # reward for player_i assuming that it got all of the pool
            max_theoretical_rewards.append(max_theoretical_r.item())  # suppose agent i has all items from the pool
            max_agreement_rewards.append(max_agreement_r)

        for i in range(2):
            if max_theoretical_rewards[i] != 0:
                scaled_rewards[i] = raw_rewards[i] / max_theoretical_rewards[i]  # reduces training instability
                
        rewards_final['player0_raw'][b] = raw_rewards[0]
        rewards_final['player1_raw'][b] = raw_rewards[1]
        rewards_final['player0_share_of_max'][b] = scaled_rewards[0]
        rewards_final['player1_share_of_max'][b] = scaled_rewards[1]
            
        rewards_final['difference_in_share_of_max'][b] = (rewards_final['player0_share_of_max'][b] - rewards_final['player1_share_of_max'][b]).abs()
                
        rewards_final['sum_raw'][b] = raw_rewards.sum()
        max_prosocial = max_utility[b].cpu().dot(s.pool[b].cpu())
        if max_prosocial != 0:
            # difference between actual and maximum potential utilitarian welfare
            rewards_final['sum_share_of_max'][b] = rewards_final['sum_raw'][b] / max_prosocial
                
        # since prosociality levels of both agents are always equal, we can disregard them when redistributing value
        # and instead work with the original raw rewards or rescaled rewards when redistributing
        if enable_decision is not None:
            # initialize redistributed rewards with shares of maximum possible rewards for each agent
            rewards_final['player0_after_arb'][b] = rewards_final['player0_share_of_max'][b]
            rewards_final['player1_after_arb'][b] = rewards_final['player1_share_of_max'][b]
            
            if enable_decision[b] == 1:
                # if we first rescale rewards to shares of maximum possible rewards for players and then redistribute based on rescaled rewards:
                if scale_before_redist: 
                    if favor_decision[b] == 1:  # if favoring player 0
                        redist_amount = rewards_final['player1_after_arb'][b]  # take away what agent 1 has
                        rewards_final['player0_arb_induced_part'][b] = redist_amount
                        rewards_final['player1_arb_induced_part'][b] = -redist_amount
                    elif favor_decision[b] == 0:
                        redist_amount = rewards_final['player0_after_arb'][b]
                        rewards_final['player0_arb_induced_part'][b] = -redist_amount
                        rewards_final['player1_arb_induced_part'][b] = redist_amount
                    else:
                        raise Exception
                # if we first redistribute raw rewards and then then normalize them:
                # (=> 0/1 rewards after redistribution)
                else:
                    favored_i = 1-favor_decision[b]  # index of the player who to redistribute the rewards to
                    if max_theoretical_rewards[favored_i] == 0:  # if dot product of utilities and pool item numbers is 0:
                        max_r = 0  # the final reward of favored_i is zero
                    elif not enable_overflow:
                        max_r = 1  # normally favored_i gets 100% of the pool -> (final reward) / (its theoretical maximum) is 1
                    else:
                        max_r = max_agreement_rewards[favored_i] / max_theoretical_rewards[favored_i]
                    if favored_i == 0:  # if redistributing to player 0
                        rewards_final['player0_arb_induced_part'][b] = max_r-rewards_final['player0_share_of_max'][b]  # => =max_r
                        rewards_final['player1_arb_induced_part'][b] = -rewards_final['player1_share_of_max'][b]  # => =0
                    elif favored_i == 1:
                        rewards_final['player0_arb_induced_part'][b] = -rewards_final['player0_share_of_max'][b]  # => =0
                        rewards_final['player1_arb_induced_part'][b] = max_r-rewards_final['player1_share_of_max'][b]  # => =max_r
                    else:
                        raise Exception

                rewards_final['player0_after_arb'][b] += rewards_final['player0_arb_induced_part'][b]
                rewards_final['player1_after_arb'][b] += rewards_final['player1_arb_induced_part'][b]

    return rewards_final
    
