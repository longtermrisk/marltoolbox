import argparse
import datetime
import ray
from ray import tune
import os
import numpy as np
import copy

from marltoolbox.algos.alternating_offers.alt_offers_training import AltOffersTraining
from marltoolbox.utils import miscellaneous, log
from marltoolbox.examples.tune_class_api.alternating_offers.cond_params import cond_params_iter, cond_params_iter_pop_training, default_cond_params

config = {
    'batch_size': 128,
    'training_episodes': 20000,
#     'training_episodes': 64,
    
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
    
#     'communication_mode': tune.grid_search(['binding', 'cheap']),
#     'match_mode': tune.grid_search(['custom_vs_custom', 'default_vs_custom', 'custom_vs_default']),
#     'cond_params': tune.grid_search(list(cond_params_iter())),
    
    # quicker version for testing
#     'communication_mode': tune.grid_search(['cheap']),
#     'match_mode': tune.grid_search(['default_vs_custom']),
#     'cond_params': tune.grid_search(list(cond_params_iter_small())),
    
#     'name': 'cross_play_20K_episodes_cheap',
#     'communication_mode': tune.grid_search(['cheap']),
#     'repetition_i': tune.grid_search(list(range(0, 5))),
#     'match_mode': tune.grid_search(['custom_vs_custom_1', 'custom_vs_custom_2', 'default_vs_custom', 'custom_vs_default']),
#     'cond_params': tune.grid_search(list(cond_params_iter())),
    
    'name': 'cross_play_20K_episodes_binding',
    'communication_mode': tune.grid_search(['binding']),
    'repetition_i': tune.grid_search(list(range(0, 5))),
    'match_mode': tune.grid_search(['custom_vs_custom_1', 'custom_vs_custom_2', 'default_vs_custom', 'custom_vs_default']),
    'cond_params': tune.grid_search(list(cond_params_iter())),
    
# #     'name': 'cross_play_20K_episodes_best_response_fixed_agent_0',
#     'name': 'cross_play_20K_episodes_best_response_fixed_agent_1',
#     'communication_mode': tune.grid_search(['binding']),
#     'repetition_i': tune.grid_search(list(range(0, 5))),
#     'match_mode': tune.grid_search(['custom_vs_custom_1',]),
#     'cond_params': tune.grid_search(list(cond_params_iter())),
    
#     'name': 'tmp',
#     'communication_mode': tune.grid_search(['binding']),
#     'repetition_i': tune.grid_search(list(range(0, 1))),
#     'match_mode': tune.grid_search(['custom_vs_custom_1',]),
#     'cond_params': tune.grid_search(list(cond_params_iter())[:1]),
    
#     'name': 'population_experiments',
#     'communication_mode': tune.grid_search(['binding']),
#     'repetition_i': tune.grid_search(list(range(0, 5))),
#     'match_mode': tune.grid_search([
#     #   for ('default_vs_custom_pop_1', 'custom_pop_vs_custom_1') eval - population training
#         'pop_2_default_constant_vs_pop_2_custom_binary', 'pop_6_default_constant_vs_pop_6_custom_binary',
#         'pop_6_default_constant_vs_pop_6_custom_uniform',
#         'pop_2_default_constant_vs_pop_2_custom_constant', 'pop_6_default_constant_vs_pop_6_custom_constant',

#         'pop_2_custom_binary_vs_pop_2_custom_constant_1', 'pop_6_custom_binary_vs_pop_6_custom_constant_1',
#         'pop_6_custom_uniform_vs_pop_6_custom_constant_1',
#         'pop_2_custom_constant_vs_pop_2_custom_constant_1', 'pop_6_custom_constant_vs_pop_6_custom_constant_1',

#     #   for ('custom_1_vs_custom_pop', 'custom_pop_vs_custom_2') eval - baseline
#         'pop_2_custom_constant_vs_pop_2_custom_binary', 'pop_6_custom_constant_vs_pop_6_custom_binary',
#         'pop_6_custom_constant_vs_pop_6_custom_uniform',
#         'pop_2_custom_constant_vs_pop_2_custom_constant_3', 'pop_6_custom_constant_vs_pop_6_custom_constant_3',

#         'pop_2_custom_binary_vs_pop_2_custom_constant_2', 'pop_6_custom_binary_vs_pop_6_custom_constant_2',
#         'pop_6_custom_uniform_vs_pop_6_custom_constant_2',
#         'pop_2_custom_constant_vs_pop_2_custom_constant_2', 'pop_6_custom_constant_vs_pop_6_custom_constant_2',
#     ]),
#     'cond_params': tune.grid_search(list(cond_params_iter_pop_training())),
    
#     'name': 'cross_play_20K_episodes_binding_tmp',
#     'communication_mode': tune.grid_search(['binding']),
#     'repetition_i': tune.grid_search(list(range(0, 5))),
#     'match_mode': tune.grid_search(['custom_vs_custom_1', 'custom_vs_custom_2', 'default_vs_custom', 'custom_vs_default']),
#     'cond_params': tune.grid_search(list(cond_params_iter())),
}

class CrossPlayTraining(AltOffersTraining):
    def setup(self, config):
        config = copy.deepcopy(config)
        
        if config['communication_mode'] == 'binding':
            config['enable_binding_comm'] = True
            config['enable_cheap_comm'] = False
        elif config['communication_mode'] == 'cheap':
            config['enable_binding_comm'] = False
            config['enable_cheap_comm'] = True
        else:
            raise NotImplementedError
        del config['communication_mode']
        
        for property_name in ['utility_types', 'prosociality_levels', 'hidden_embedding_sizes', 'fairness_coeffs']:
            config[property_name] = []
        if config['match_mode'][-2] == '_' and config['match_mode'][-1].isdigit():
            config['match_mode'] = config['match_mode'][:-2]
        modes = config['match_mode'].split('_vs_')
        for mode in modes:
            if mode == 'custom':
                config['utility_types'].append(f'skew_{config["cond_params"]["utilities_skew"]}')
                config['prosociality_levels'].append(config['cond_params']['prosociality_level'])
                config['hidden_embedding_sizes'].append(config['cond_params']['hidden_embedding_size'])
                config['fairness_coeffs'].append(config['cond_params']['fairness_coeff'])
            elif mode == 'default':
                config['utility_types'].append(f'skew_{default_cond_params["utilities_skew"]}')
                config['prosociality_levels'].append(default_cond_params['prosociality_level'])
                config['hidden_embedding_sizes'].append(default_cond_params['hidden_embedding_size'])
                config['fairness_coeffs'].append(default_cond_params['fairness_coeff'])
            elif mode == 'pop_2_custom_binary':
                config['utility_types'].append([f'skew_{config["cond_params"]["utilities_skew"]}',]*2)
                config['prosociality_levels'].append([config['cond_params']['prosociality_level'],]*2)
                config['hidden_embedding_sizes'].append([config['cond_params']['hidden_embedding_size'],]*2)
                config['fairness_coeffs'].append([0.0, config['cond_params']['fairness_coeff']])
            elif mode == 'pop_6_custom_binary':
                config['utility_types'].append([f'skew_{config["cond_params"]["utilities_skew"]}',]*6)
                config['prosociality_levels'].append([config['cond_params']['prosociality_level'],]*6)
                config['hidden_embedding_sizes'].append([config['cond_params']['hidden_embedding_size'],]*6)
                config['fairness_coeffs'].append([0.0,]*3 + [config['cond_params']['fairness_coeff'],]*3)
            elif mode == 'pop_6_custom_uniform':
                config['utility_types'].append([f'skew_{config["cond_params"]["utilities_skew"]}',]*6)
                config['prosociality_levels'].append([config['cond_params']['prosociality_level'],]*6)
                config['hidden_embedding_sizes'].append([config['cond_params']['hidden_embedding_size'],]*6)
                config['fairness_coeffs'].append(np.arange(0.0, 0.6, 0.1).tolist())
            elif mode == 'pop_2_custom_constant':
                config['utility_types'].append([f'skew_{config["cond_params"]["utilities_skew"]}',]*2)
                config['prosociality_levels'].append([config['cond_params']['prosociality_level'],]*2)
                config['hidden_embedding_sizes'].append([config['cond_params']['hidden_embedding_size'],]*2)
                config['fairness_coeffs'].append([config['cond_params']['fairness_coeff'],]*2)
            elif mode == 'pop_6_custom_constant':
                config['utility_types'].append([f'skew_{config["cond_params"]["utilities_skew"]}',]*6)
                config['prosociality_levels'].append([config['cond_params']['prosociality_level'],]*6)
                config['hidden_embedding_sizes'].append([config['cond_params']['hidden_embedding_size'],]*6)
                config['fairness_coeffs'].append([config['cond_params']['fairness_coeff'],]*6)
            elif mode == 'pop_2_default_constant':
                config['utility_types'].append([f'skew_{default_cond_params["utilities_skew"]}',]*2)
                config['prosociality_levels'].append([default_cond_params['prosociality_level'],]*2)
                config['hidden_embedding_sizes'].append([default_cond_params['hidden_embedding_size'],]*2)
                config['fairness_coeffs'].append([default_cond_params['fairness_coeff'],]*2)
            elif mode == 'pop_6_default_constant':
                config['utility_types'].append([f'skew_{default_cond_params["utilities_skew"]}',]*6)
                config['prosociality_levels'].append([default_cond_params['prosociality_level'],]*6)
                config['hidden_embedding_sizes'].append([default_cond_params['hidden_embedding_size'],]*6)
                config['fairness_coeffs'].append([default_cond_params['fairness_coeff'],]*6)
            else:
                raise NotImplementedError

#         del config['match_mode']
#         del config['cond_params']
        super().setup(config)
        
if __name__ == '__main__':
    # each logical step of training contains several episodes, each episode is a batch of games
    training_steps = config['training_episodes'] // config['episodes_per_step']
    print(f'Num of training steps: {training_steps}')
    print(f'Episodes per step: {config["episodes_per_step"]}')
    # arbitrator in population training hasn't been tested
    assert not (config['enable_arbitrator'] and isinstance(config['hidden_embedding_sizes'][0], list))

    exp_name_expanded, exp_dir = log.log_in_current_day_dir(config['name'])

    analysis = tune.run(
        name=exp_name_expanded,
        run_or_experiment=CrossPlayTraining,
        stop={"training_iteration": training_steps},
        config=config,
        checkpoint_freq=1000,
        checkpoint_at_end=True,
        metric='prosocial_reward',
        mode='max',
    )

    log.save_metrics(analysis, exp_name_expanded, "metrics.pickle")
    # log.pprint_saved_metrics(os.path.join(os.path.expanduser('~/ray_results'), exp_name_expanded, 'metrics.pickle'))
