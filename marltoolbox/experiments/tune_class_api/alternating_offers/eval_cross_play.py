import datetime
from ray import tune
import os
import torch
import pickle
from tqdm import tqdm, trange
import numpy as np
from collections import defaultdict

from marltoolbox.algos.alternating_offers.alt_offers_training import AltOffersTraining, run_episode
from marltoolbox.algos.alternating_offers.envs.alt_offers_env import AltOffersEnv
from marltoolbox.experiments.tune_class_api.alternating_offers import train_cross_play



eval_episodes = 128

root_path = '/home/alex_grig_lyzhov/ray_results/cross_play_20K_episodes/2021_03_18/22_47_17'
checkpoint_dir_name = 'checkpoint_312'  # for 20K episodes
binding = False

# root_path = '/home/alex_grig_lyzhov/ray_results/cross_play_20K_episodes_binding/2021_03_20/20_17_24'
# checkpoint_dir_name = 'checkpoint_312'  # for 20K episodes
# binding = True



default_cond_params = train_cross_play.default_cond_params
cond_params_list = list(train_cross_play.cond_params_iter())

default_i = None
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
        for j in range(2):
            cur_trial_info['agents'].append(state['agent%s' % j]['model_state'])
        trial_info.append(cur_trial_info)
        
        if cur_trial_info['params']['cond_params'] == default_cond_params:
            default_i = len(trial_info)-1
            
matches = {}
matches['custom_vs_custom_1'] = [trial for trial in trial_info if (trial['params']['match_mode'] == 'custom_vs_custom_1')]
matches['custom_vs_custom_2'] = [trial for trial in trial_info if (trial['params']['match_mode'] == 'custom_vs_custom_2')]
matches['default_vs_custom'] = [trial for trial in trial_info if (trial['params']['match_mode'] == 'default_vs_custom')]
matches['custom_vs_default'] = [trial for trial in trial_info if (trial['params']['match_mode'] == 'custom_vs_default')]
matches['default_vs_default'] = [trial_info[default_i],] * len(matches['custom_vs_custom_1'])

match_indices = defaultdict(list)
for match_regime, match_trials in matches.items():
    if match_regime == 'default_vs_default':
        match_indices['default_vs_default'] = range(len(matches['default_vs_default']))
    else:
        for cond_params_i, cond_params in enumerate(cond_params_list):
            eq_list = [(i, trial) for (i, trial) in enumerate(match_trials) if trial['params']['cond_params'] == cond_params]
            ind = eq_list[0][0]
            match_indices[match_regime].append(ind)
            

results_filename = os.path.join('results', 'results_' + '_'.join(root_path.split('/')[-3:]) + '.pickle')
results = defaultdict(list)
for agent_0_match_regime, agent_1_match_regime in (
                                                   ('custom_vs_custom_1', 'custom_vs_custom_1'),  # normal eval
                                                   ('custom_vs_custom_1', 'custom_vs_custom_2'),  # simplest (1)
                                                   
                                                   ('custom_vs_custom_1', 'default_vs_default'),  # 2.1
                                                   ('default_vs_default', 'custom_vs_custom_1'),  # 2.2
                                                   
                                                   ('default_vs_custom', 'custom_vs_custom_1'),  # 3.1
                                                   ('custom_vs_custom_1', 'default_vs_custom'),  # 4.1
    
                                                   ('default_vs_default', 'default_vs_custom'),  # 3.2
                                                   ('default_vs_custom', 'default_vs_default'),  # 4.2
                                                  ):  
    for cond_params_i in trange(len(cond_params_list)):
        trial_0_i = match_indices[agent_0_match_regime][cond_params_i]
        trial_1_i = match_indices[agent_1_match_regime][cond_params_i]
        trial_0 = matches[agent_0_match_regime][trial_0_i]
        trial_1 = matches[agent_1_match_regime][trial_1_i]

        if agent_0_match_regime.split('_vs_')[0] == 'default':
            agent_0_params = default_cond_params
        else:
            agent_0_params = trial_0['params']['cond_params']
        if agent_1_match_regime.split('_vs_')[1] == 'default':
            agent_1_params = default_cond_params
        else:
            agent_1_params = trial_1['params']['cond_params']

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
            'hidden_embedding_sizes': [agent_0_params['hidden_embedding_size'],
                                       agent_1_params['hidden_embedding_size']],
            'utility_types': [f"skew_{agent_0_params['utilities_skew']:.1f}",
                              f"skew_{agent_1_params['utilities_skew']:.1f}"],
            'prosociality_levels': [agent_0_params['prosociality_level'],
                                    agent_1_params['prosociality_level']],
            'fairness_coeffs': [agent_0_params['fairness_coeff'],
                                agent_1_params['fairness_coeff']],
        }

        alt_offers = AltOffersTraining(config)
        alt_offers.agent_models[0].load_state_dict(trial_0['agents'][0])
        alt_offers.agent_models[1].load_state_dict(trial_1['agents'][1])

        all_rewards = {}
        for i in range(eval_episodes):
            env_state = AltOffersEnv(batch_size=alt_offers.args['batch_size'], random_state=alt_offers.train_random_state,
                                 utility_types=alt_offers.args['utility_types'])
            action_logprobs, entropy_loss_by_agent, arb_logprobs_accum, arb_entropy_loss, rewards, num_turns, alive_masks, \
            new_agent_forward_stats, new_arb_forward_stats = run_episode(
                args=alt_offers.args,
                env_state=env_state,
                agent_models=alt_offers.agent_models,
                arbitrator_model=alt_offers.arbitrator_model if alt_offers.args['enable_arbitrator'] else None,
                render_flag=False,
                deterministic=False,
            )
            for reward_name, val in rewards.items():
                if reward_name not in all_rewards:
                    all_rewards[reward_name] = rewards[reward_name].numpy()
                else:
                    all_rewards[reward_name] = np.concatenate([all_rewards[reward_name], rewards[reward_name].numpy()])
        results[(agent_0_match_regime, agent_1_match_regime)].append(all_rewards)

pickle.dump(results, open(results_filename, 'wb'))
