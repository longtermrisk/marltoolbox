##########
# Part of the code modified from:
# https://github.com/asappresearch/emergent-comms-negotiation
##########

import json
import time

def render_action(s, prop, msg, response, enable_binding_comm, enable_cheap_comm):
    '''Params: timestep, state object, proposal with long type, acc/rej
    acc/rej response is of size (batch_size, 1)
    proposal is of size (batch_size, 3)
    Prints one action from the game corresponding to the first batch element
    '''
    t = s.t
    if t == 0:  # print utilities first
        print('  ')
        print('  ', end='')
        # [batch elem, agent_i][item_i]
        print(f' utilities {s.utilities[0, 0][0].item()}{s.utilities[0, 0][1].item()}{s.utilities[0, 0][2].item()}')
        print('                                   ', end='')
        print(f' utilities {s.utilities[0, 1][0].item()}{s.utilities[0, 1][1].item()}{s.utilities[0, 1][2].item()}')
        
    agent = t % 2
    speaker = 'A' if agent == 0 else 'B'
    print('  ', end='')
    if speaker == 'B':
        print('                                   ', end='')
    if response[0][0]:
        print(' ACC')
    else:
        binding_regime = 'public' if enable_binding_comm else 'private'
        cheap_str = f', cheap {msg[0][0].item()}/{s.pool[0][0].item()} {msg[0][1].item()}/{s.pool[0][1].item()} {msg[0][2].item()}/{s.pool[0][2].item()}' if enable_cheap_comm else ''
        print(f' binding {binding_regime} {prop[0][0].item()}/{s.pool[0][0].item()} {prop[0][1].item()}/{s.pool[0][1].item()} {prop[0][2].item()}/{s.pool[0][2].item()}{cheap_str}', end='')
        print('')
        if t + 1 == s.N[0]:
            print('  [out of time]')
            
def render_rewards(rewards):
    print('   player0 %.2f, player1 %.2f, welfare %.2f' % (rewards['player0_share_of_max'][0], rewards['player1_share_of_max'][0], rewards['sum_share_of_max'][0]))
    print('  ')

def get_step_log(training):
    '''Logging over several episodes in the step
    '''
    time_since_last = time.time() - training.last_print  # to compute how many games processed per second
    training.last_print = time.time()

    log_data = {
        'episodes': training.episodes_completed,
        'orig_reward_0': (training.rewards_sum['player0_share_of_max'] / training.processed_games_sum).item(),
        'orig_reward_1': (training.rewards_sum['player1_share_of_max'] / training.processed_games_sum).item(),
        'orig_welfare': (training.rewards_sum['sum_share_of_max'] / training.processed_games_sum).item(),
        'avg_steps': float(training.num_turns_accum) / training.processed_games_sum,
        'games_sec': int(training.processed_games_sum / time_since_last),
        'elapsed': time.time() - training.start_time,

        'argmaxp_response_0': float(training.agent_forward_stats[0]['response_matches_argmax_count']) / training.agent_forward_stats[0]['response_stochastic_draws'],
        'argmaxp_response_1': float(training.agent_forward_stats[1]['response_matches_argmax_count']) / training.agent_forward_stats[1]['response_stochastic_draws'],
        'argmaxp_utt_0': safe_div(float(training.agent_forward_stats[0]['utt_matches_argmax_count']), training.agent_forward_stats[0]['utt_stochastic_draws']),
        'argmaxp_utt_1': safe_div(float(training.agent_forward_stats[1]['utt_matches_argmax_count']), training.agent_forward_stats[1]['utt_stochastic_draws']),
        'argmaxp_prop_0': float(training.agent_forward_stats[0]['prop_matches_argmax_count']) / training.agent_forward_stats[0]['prop_stochastic_draws'],
        'argmaxp_prop_1': float(training.agent_forward_stats[1]['prop_matches_argmax_count']) / training.agent_forward_stats[1]['prop_stochastic_draws'],
        'response_entropy_0': float(training.agent_forward_stats[0]['response_entropy']) / training.agent_forward_stats[0]['response_stochastic_draws'],
        'response_entropy_1': float(training.agent_forward_stats[1]['response_entropy']) / training.agent_forward_stats[1]['response_stochastic_draws'],
        'utt_entropy_0': safe_div(float(training.agent_forward_stats[0]['utt_entropy']), training.agent_forward_stats[0]['utt_stochastic_draws']),
        'utt_entropy_1': safe_div(float(training.agent_forward_stats[1]['utt_entropy']), training.agent_forward_stats[1]['utt_stochastic_draws']),
        'prop_entropy_0': float(training.agent_forward_stats[0]['prop_entropy']) / training.agent_forward_stats[0]['prop_stochastic_draws'],
        'prop_entropy_1': float(training.agent_forward_stats[1]['prop_entropy']) / training.agent_forward_stats[1]['prop_stochastic_draws'],
        'response_prob_0': float(training.agent_forward_stats[0]['response_prob']) / training.agent_forward_stats[0]['response_stochastic_draws'],
        'response_prob_1': float(training.agent_forward_stats[1]['response_prob']) / training.agent_forward_stats[1]['response_stochastic_draws'],
        'entropy_loss_by_agent_0': training.entropy_loss_by_agent_accum[0] / training.processed_games_sum,
        'entropy_loss_by_agent_1': training.entropy_loss_by_agent_accum[1] / training.processed_games_sum,
        'main_loss_by_agent_0': training.main_loss_by_agent_accum[0] / training.processed_games_sum,
        'main_loss_by_agent_1': training.main_loss_by_agent_accum[1] / training.processed_games_sum,
    }
    if training.args['enable_arbitrator']:
        log_data.update({
            'arb_main_loss': training.arb_main_loss_accum / training.processed_games_sum,
            'arb_entropy_loss': training.arb_entropy_loss_accum / training.processed_games_sum,
            'redistributed_reward_0': (training.rewards_sum['player0_after_arb'] / training.processed_games_sum).item(),
            'redistributed_reward_1': (training.rewards_sum['player1_after_arb'] / training.processed_games_sum).item(),
            'induced_reward_0': (training.rewards_sum['player0_arb_induced_part'] / training.processed_games_sum).item(),
            'induced_reward_1': (training.rewards_sum['player1_arb_induced_part'] / training.processed_games_sum).item(),

            'argmaxp_enable': safe_div(float(training.arb_forward_stats['enable_argmax_count']), training.arb_forward_stats['enable_draws']),
            'argmaxp_favor': safe_div(float(training.arb_forward_stats['favor_argmax_count']), training.arb_forward_stats['favor_draws']),
            'enable_entropy': safe_div(float(training.arb_forward_stats['enable_entropy']), training.arb_forward_stats['enable_draws']),
            'favor_entropy': safe_div(float(training.arb_forward_stats['favor_entropy']), training.arb_forward_stats['favor_draws']),
            'enable_decision': safe_div(float(training.arb_forward_stats['enable_decision']), training.arb_forward_stats['enable_draws']),
            'favor_decision': safe_div(float(training.arb_forward_stats['favor_decision']), training.arb_forward_stats['favor_draws']),
            'enable_probs': safe_div(float(training.arb_forward_stats['enable_probs']), training.arb_forward_stats['enable_draws']),
            'favor_probs': safe_div(float(training.arb_forward_stats['favor_probs']), training.arb_forward_stats['favor_draws']),
        })
    return log_data
    
def print_step_log(training, log_data):
    '''Console logging at the end of the episode
    '''
    if training.args['enable_arbitrator']:
        print('orig0 %.2f, orig1 %.2f, welfare %.2f, induced0 %.2f, induced1 %.2f' % (log_data['orig_reward_0'], log_data['orig_reward_1'], log_data['orig_welfare'], log_data['induced_reward_0'], log_data['induced_reward_1']))
    else:
        print('orig0 %.2f, orig1 %.2f, welfare %.2f' % (log_data['orig_reward_0'], log_data['orig_reward_1'], log_data['orig_welfare']))
    print(f"n_episodes={log_data['episodes']}, n_games={(log_data['episodes']+1)*training.args['batch_size']}, batch_size={training.args['batch_size']}, {log_data['games_sec']} games/s")
    print(f"avg_steps {log_data['avg_steps']:.3f}, argmaxp_response_0 {log_data['argmaxp_response_0']:.3f}, argmaxp_utt_0 {log_data['argmaxp_utt_0']:.3f}, argmaxp_prop_0 {log_data['argmaxp_prop_0']:.3f}")
    
def safe_div(a, b):
    return 0 if b == 0 else a / b