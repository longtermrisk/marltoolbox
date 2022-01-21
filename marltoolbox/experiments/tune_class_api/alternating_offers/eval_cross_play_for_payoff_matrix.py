import datetime
from ray import tune
import os
import torch
import pickle
from tqdm import tqdm, trange
from tqdm.contrib.concurrent import process_map
import numpy as np
from collections import defaultdict
import multiprocessing

from marltoolbox.algos.alternating_offers.alt_offers_training import AltOffersTraining, run_episode
from marltoolbox.algos.alternating_offers.envs.alt_offers_env import AltOffersEnv
from marltoolbox.experiments.tune_class_api.alternating_offers.cond_params import cond_params_iter, cond_params_iter_pop_training, default_cond_params, round_cond_params


# eval_episodes = 128
# checkpoint_dir_name = 'checkpoint_312'  # for 20K episodes

# cond_params - changing agent parameters and not trial parameters here
default_cond_params = default_cond_params
cond_params_list = list(cond_params_iter())

# root_path = '/home/alex_grig_lyzhov/ray_results/cross_play_20K_episodes/2021_03_18/22_47_17'
# binding = False

# root_path = '/home/alex_grig_lyzhov/ray_results/cross_play_20K_episodes_binding/2021_03_20/20_17_24'
# binding = True

# root_path = '/home/alexander/ray_results/cross_play_20K_episodes_cheap/2021_03_24/05_30_06'
# binding = False

# root_path = '/home/alexander/ray_results/cross_play_20K_episodes_binding/2021_03_23/12_37_37'
# binding = True

root_path = '/home/alexander/ray_results/cross_play_20K_episodes_binding_prosocial/2021_05_20/01_53_31'
binding = True

# results_filename = os.path.join('cross_play_evals', 'results_' + '_'.join(root_path.split('/')[-3:]) + '.pickle')
results_filename = os.path.join('cross_play_evals', 'results_more_games_' + '_'.join(root_path.split('/')[-3:]) + '.pickle')


# eval_episodes = 64
eval_episodes = 2048
# checkpoint_dir_name = 'checkpoint_1'
checkpoint_dir_name = 'checkpoint_312'  # for 20K episodes
# binding = True
# default_cond_params = default_cond_params

# root_path = '/home/alexander/ray_results/population_experiments/2021_04_20/14_01_26'
# cond_params_list = list(cond_params_iter_pop_training())

# root_path = '/home/alexander/ray_results/cross_play_20K_episodes_best_response_fixed_agent_0/2021_05_03/00_35_50'
# root_path = '/home/alexander/ray_results/cross_play_20K_episodes_best_response_fixed_agent_1/2021_05_03/09_05_24'
# cond_params_list = list(cond_params_iter())


def get_rewards(args):
    agent_match_regimes, cond_params_i, trial_indices, agent_params = args
#     agent_params, agent_weights, eval_episodes, binding = args

    # TODO load all other params from both configs
    config = {
        'name': 'cross_play_eval',
        'batch_size': 128,
        'training_episodes': eval_episodes,

        'enable_arbitrator': False,
        'scale_before_redist': True,

        'response_entropy_reg': 0.05,
        'utterance_entropy_reg': 1e-3,
        'proposal_entropy_reg': 0.05,
        'arbitrator_entropy_reg': None,
        'arbitrator_main_loss_coeff': None,

        'train_seed': None,
        'enable_cuda': False,
        'episodes_per_step': 64,
        'suppress_output': True,

        'agents_sgd': False,
        'arbitrator_sgd': False,
        'share_utilities': False,
        'enable_overflow': False,

        'enable_binding_comm': binding,
        'enable_cheap_comm': not binding,
        'hidden_embedding_sizes': [agent_params[0]['hidden_embedding_size'],
                                   agent_params[1]['hidden_embedding_size']],
        'utility_types': [f"skew_{agent_params[0]['utilities_skew']:.1f}",
                          f"skew_{agent_params[1]['utilities_skew']:.1f}"],
        'prosociality_levels': [agent_params[0]['prosociality_level'],
                                agent_params[1]['prosociality_level']],
        'fairness_coeffs': [agent_params[0]['fairness_coeff'],
                            agent_params[1]['fairness_coeff']],
        
        'cond_params': cond_params_list[cond_params_i],
    }
    alt_offers = AltOffersTraining(config)
    
    if isinstance(trial_indices[0], list):  # population training eval
        trial_is = (trial_indices[0][0], trial_indices[1][0])
        pop_is = (trial_indices[0][1], trial_indices[1][1])  # indices of agents in the population for each selected trial
        agent_weights = [  # have to fetch them here, otherwise if I pass weights in args it slows multiprocessing down
            matches[agent_match_regimes[0]][trial_is[0]]['agents'][0][pop_is[0]],
            matches[agent_match_regimes[1]][trial_is[1]]['agents'][1][pop_is[1]],
        ]
    else:
        agent_weights = [  # have to fetch them here, otherwise if I pass weights in args it slows multiprocessing down
            matches[agent_match_regimes[0]][trial_indices[0]]['agents'][0],
            matches[agent_match_regimes[1]][trial_indices[1]]['agents'][1],
        ]
    alt_offers.agent_models[0].load_state_dict(agent_weights[0])
    alt_offers.agent_models[1].load_state_dict(agent_weights[1])

    all_rewards = {}
    for i in range(eval_episodes):
        env_state = AltOffersEnv(batch_size=alt_offers.args['batch_size'], random_state=alt_offers.train_random_state, utility_types=alt_offers.args['utility_types'])
            
        action_logprobs, entropy_loss_by_agent, arb_logprobs_accum, arb_entropy_loss, rewards, num_turns, alive_masks, \
        new_agent_forward_stats, new_arb_forward_stats = run_episode(
            args=alt_offers.args,
            env_state=env_state,
            agent_models=alt_offers.agent_models,
            arbitrator_model=alt_offers.arbitrator_model if alt_offers.args['enable_arbitrator'] else None,
            render_flag=False,
            deterministic=False,
            embedding_sizes=alt_offers.args['hidden_embedding_sizes'],
        )
        
        items_list = list(rewards.items()) + [('num_turns', num_turns)]
#         print(len(items_list[0]))
        for reward_name, val in items_list:
            if reward_name not in all_rewards:
                all_rewards[reward_name] = val.numpy()
            else:
                all_rewards[reward_name] = np.concatenate([all_rewards[reward_name], val.numpy()])
#         for reward_name, val in ['num_turns']
#         if reward_name not in all_rewards:
#                 all_rewards[reward_name] = rewards[reward_name].numpy()
#             else:
#                 all_rewards[reward_name] = np.concatenate([all_rewards[reward_name], rewards[reward_name].numpy()])
    return all_rewards

default_indices = []
trial_info = []
for i, trial_name in enumerate(os.listdir(root_path)):
    trial_path = os.path.join(root_path, trial_name)
    if os.path.isdir(trial_path):
        cur_trial_info = {}
        cur_trial_info['params'] = pickle.load(open(os.path.join(trial_path, 'params.pkl'), 'rb'))
        model_path = os.path.join(trial_path, checkpoint_dir_name, 'model.pth')
        with open(model_path, 'rb') as f:
            state = torch.load(f)
        cur_trial_info['agents'] = []
        for agent_i in range(2):
            cur_trial_info['agents'].append(state['agent%s' % agent_i]['model_state'])  # either a single model or a list of population models
        trial_info.append(cur_trial_info)
        
        # disable for population training
        if cur_trial_info['params']['cond_params'] == default_cond_params:  # if custom params == default params
            default_indices.append(len(trial_info)-1)  # correspond to default_vs_default plays
            
matches = {}
# matches['custom_vs_custom_1'] = [trial for trial in trial_info if (trial['params']['match_mode'] == 'custom_vs_custom_1')]
# matches['custom_vs_custom_2'] = [trial for trial in trial_info if (trial['params']['match_mode'] == 'custom_vs_custom_2')]
# matches['default_vs_custom'] = [trial for trial in trial_info if (trial['params']['match_mode'] == 'default_vs_custom')]
# matches['custom_vs_default'] = [trial for trial in trial_info if (trial['params']['match_mode'] == 'custom_vs_default')]
# disable for population training
matches['default_vs_default'] = [trial_info[default_i] for default_i in default_indices]
all_training_modes = list(set([trial['params']['match_mode'] for trial in trial_info]))
for mode in all_training_modes:
    matches[mode] = [trial for trial in trial_info if trial['params']['match_mode'] == mode]

# now sort training matches for various regimes in the order prescribed by cond_params_list
match_indices = defaultdict(list)
for match_regime, match_trials in matches.items():
# disable for population training
    if match_regime == 'default_vs_default':
        match_indices['default_vs_default'] = [list(range(len(matches['default_vs_default'])))
                                               for cond_params_i in range(len(cond_params_list))]
    else:
        for cond_params_i, cond_params in enumerate(cond_params_list):
            eq_list = [(i, trial) for (i, trial) in enumerate(match_trials) if round_cond_params(trial['params']['cond_params']) == cond_params]
            indices = [eq_trial[0] for eq_trial in eq_list]
            match_indices[match_regime].append(indices)
            
print(match_indices['custom_vs_custom_1'])
args_list = []
for agent_0_match_regime, agent_1_match_regime in (
    # normal cross-play
#                                                    ('custom_vs_custom_1', 'custom_vs_custom_1'),  # normal eval
#                                                    ('custom_vs_custom_1', 'custom_vs_custom_2'),  # simplest (1)
#                                                    ('default_vs_default', 'default_vs_default'),
                                                   
#                                                    ('custom_vs_custom_1', 'default_vs_default'),  # 2.1
#                                                    ('default_vs_default', 'custom_vs_custom_1'),  # 2.2
                                                   
#                                                    ('default_vs_custom', 'custom_vs_custom_1'),  # 3.1
#                                                    ('custom_vs_custom_1', 'default_vs_custom'),  # 4.1
#                                                    ('default_vs_default', 'custom_vs_default'),
#                                                    ('custom_vs_default', 'default_vs_default'),
    
#                                                    ('default_vs_default', 'default_vs_custom'),  # 3.2
#                                                    ('default_vs_custom', 'default_vs_default'),  # 4.2
#                                                    ('custom_vs_default', 'custom_vs_custom_1'),
#                                                    ('custom_vs_custom_1', 'custom_vs_default'),
    
#                                                    ('custom_vs_custom_1', 'custom_vs_custom_1'),  # normal eval
    
                                                   ('custom_vs_custom_1', 'custom_vs_custom_2'),  # simplest (1)
#                                                    ('default_vs_default', 'default_vs_default'),
                                                   
#                                                    ('custom_vs_custom_1', 'default_vs_default'),  # 2.1
#                                                    ('default_vs_default', 'custom_vs_custom_1'),  # 2.2
                                                   
#                                                    ('default_vs_custom', 'custom_vs_custom_1'),  # 3.1
#     # the part next was not included:
#                                                    ('custom_vs_custom_1', 'default_vs_custom'),  # 4.1
#                                                    ('default_vs_default', 'custom_vs_default'),
#                                                    ('custom_vs_default', 'default_vs_default'),
    
#                                                    ('default_vs_default', 'default_vs_custom'),  # 3.2
#                                                    ('default_vs_custom', 'default_vs_default'),  # 4.2
#                                                    ('custom_vs_default', 'custom_vs_custom_1'),
#                                                    ('custom_vs_custom_1', 'custom_vs_default'),
    
#     #   for ('default_vs_custom_pop_1', 'custom_pop_vs_custom_1') eval - population training
#     ('pop_2_default_constant_vs_pop_2_custom_binary', 'pop_2_custom_binary_vs_pop_2_custom_constant_1'),
#     ('pop_6_default_constant_vs_pop_6_custom_binary', 'pop_6_custom_binary_vs_pop_6_custom_constant_1',),
#     ('pop_6_default_constant_vs_pop_6_custom_uniform', 'pop_6_custom_uniform_vs_pop_6_custom_constant_1',),
#     ('pop_2_default_constant_vs_pop_2_custom_constant', 'pop_2_custom_constant_vs_pop_2_custom_constant_1',),
#     ('pop_6_default_constant_vs_pop_6_custom_constant', 'pop_6_custom_constant_vs_pop_6_custom_constant_1',),

#     #   for ('custom_1_vs_custom_pop', 'custom_pop_vs_custom_2') eval - baseline
#     ('pop_2_custom_constant_vs_pop_2_custom_binary', 'pop_2_custom_binary_vs_pop_2_custom_constant_2'),
#     ('pop_6_custom_constant_vs_pop_6_custom_binary', 'pop_6_custom_binary_vs_pop_6_custom_constant_2'),
#     ('pop_6_custom_constant_vs_pop_6_custom_uniform', 'pop_6_custom_uniform_vs_pop_6_custom_constant_2'),
#     ('pop_2_custom_constant_vs_pop_2_custom_constant_3', 'pop_2_custom_constant_vs_pop_2_custom_constant_2'),
#     ('pop_6_custom_constant_vs_pop_6_custom_constant_3', 'pop_6_custom_constant_vs_pop_6_custom_constant_2'),
):
    for cond_params_i_0 in [0,] + list(range(7, 14)):  # over all types of agents
        for cond_params_i_1 in [0,] + list(range(7, 14)):  # over all types of agents
            trial_0_indices = match_indices[agent_0_match_regime][cond_params_i_0]
            trial_1_indices = match_indices[agent_1_match_regime][cond_params_i_1]
    #         for seed_i in range(min(5, len(trial_0_indices), len(trial_0_indices))):
    #             trial_0_i = np.random.choice(trial_0_indices)
    #             if agent_0_match_regime == agent_1_match_regime:
    #                 trial_1_i = trial_0_i  # to match agents trained together
    #             else:
    #                 trial_1_i = np.random.choice(trial_1_indices)
            if agent_0_match_regime == agent_1_match_regime:
                eval_trials_list = [(trial_0_i, trial_0_i) for trial_0_i in trial_0_indices]  # to match agents trained together
            else:
                eval_trials_list = [(trial_0_i, trial_1_i) for trial_0_i in trial_0_indices for trial_1_i in trial_1_indices]

            for trial_0_i, trial_1_i in eval_trials_list:
                trial_0 = matches[agent_0_match_regime][trial_0_i]
                trial_1 = matches[agent_1_match_regime][trial_1_i]

                if 'default' in agent_0_match_regime.split('_vs_')[0]:
                    agent_0_params = default_cond_params
                else:
                    agent_0_params = trial_0['params']['cond_params']
                if 'default' in agent_1_match_regime.split('_vs_')[1]:
                    agent_1_params = default_cond_params
                else:
                    agent_1_params = trial_1['params']['cond_params']

                if isinstance(trial_0['agents'][0], list):
    #                 for iter_i in range(len(trial_0['agents'][0])):  # let's pick N random matches for 2 populations of size N
    #                     pop_0_i = np.random.randint(0, len(trial_0['agents'][0]))
    #                     pop_1_i = np.random.randint(0, len(trial_0['agents'][0]))
                    for pop_0_i in range(len(trial_0['agents'][0])):
                        for pop_1_i in range(len(trial_1['agents'][1])):
                            args_list.append(((agent_0_match_regime, agent_1_match_regime), cond_params_i,
                                              ([trial_0_i, pop_0_i], [trial_1_i, pop_1_i]), (agent_0_params, agent_1_params),))
                else:
                    args_list.append(((agent_0_match_regime, agent_1_match_regime), 0,
                                      (trial_0_i, trial_1_i), (agent_0_params, agent_1_params),))

# pool = multiprocessing.Pool()
# results = pool.map(get_rewards, args_list)
results = []
# chunk_size = 1024
chunk_size = 64
chunks = [args_list[x:x+chunk_size] for x in range(0, len(args_list), chunk_size)]
print('args_list', len(args_list))
print('chunks', len(chunks))
for chunk in tqdm(chunks):
    cur_results = process_map(get_rewards, chunk)
#     print(cur_results[0]['player0_share_of_max'].mean())
#     print(cur_results[10]['player0_share_of_max'].mean())
#     print(cur_results[20]['player0_share_of_max'].mean())
#     print(cur_results[30]['player0_share_of_max'].mean())
    statistics = [(result['player0_share_of_max'].mean(), result['player0_share_of_max'].std(),
                   result['player1_share_of_max'].mean(), result['player1_share_of_max'].std(),
                   result['sum_share_of_max'].mean(), result['sum_share_of_max'].std()) for result in cur_results]
    del cur_results
    statistics = [chunk[i][:3] + (statistic,) for i, statistic in enumerate(statistics)]
    results.extend(statistics)
    os.makedirs('results', exist_ok=True)
    pickle.dump(results, open(results_filename, 'wb'))

#     results = process_map(get_rewards, args_list)
# results = [args_list[i][:3] + (result,) for i, result in enumerate(results)]
# os.makedirs('results', exist_ok=True)
# pickle.dump(results, open(results_filename, 'wb'))
# if pool is not None:
#     pool.close()

