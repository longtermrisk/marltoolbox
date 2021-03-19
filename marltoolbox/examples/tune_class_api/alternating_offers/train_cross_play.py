import argparse
import datetime
import ray
from ray import tune
import os
import numpy as np
import copy

from marltoolbox.algos.alternating_offers.alt_offers_training import AltOffersTraining
from marltoolbox.utils import miscellaneous, log

# import warnings
# warnings.filterwarnings("ignore")

# def warn(*args, **kwargs):
#     pass
# import warnings
# warnings.warn = warn

# ray.init(log_to_driver=False, configure_logging=True, logging_level=50)

default_cond_params = {'utilities_skew': 0, 'prosociality_level': 0.3, 'hidden_embedding_size': 30, 'fairness_coeff': 0}

def cond_params_iter():
    yield default_cond_params  # default parameters
    
    for utilities_skew in np.arange(-1, 1.1, 0.4):
        cur_params = copy.deepcopy(default_cond_params)
        cur_params['utilities_skew'] = utilities_skew
        yield cur_params
#         yield {'utilities_skew': utilities_skew, 'prosociality_level': 0.3, 'hidden_embedding_size': 30, 'fairness_coeff': 0}
        
    for fairness_coeff in np.arange(0.0, 0.7, 0.1):
        cur_params = copy.deepcopy(default_cond_params)
        cur_params['fairness_coeff'] = fairness_coeff
        yield cur_params
#         yield {'utilities_skew': 0, 'prosociality_level': 0.3, 'hidden_embedding_size': 30, 'fairness_coeff': fairness_coeff}
        
    for prosociality_level in np.arange(0, 1.1, 0.2):
        cur_params = copy.deepcopy(default_cond_params)
        cur_params['prosociality_level'] = prosociality_level
        yield cur_params
#         yield {'utilities_skew': 0, 'prosociality_level': prosociality_level, 'hidden_embedding_size': 30, 'fairness_coeff': 0}
        
    for hidden_embedding_size in [5, 10, 20, 40, 80, 160]:
        cur_params = copy.deepcopy(default_cond_params)
        cur_params['hidden_embedding_size'] = hidden_embedding_size
        yield cur_params
#         yield {'utilities_skew': 0, 'prosociality_level': 0.3, 'hidden_embedding_size': hidden_embedding_size, 'fairness_coeff': 0}

def cond_params_iter_small():
    yield default_cond_params  # default parameters
    
    for utilities_skew in np.arange(-1, 1.1, 0.4):
        cur_params = copy.deepcopy(default_cond_params)
        cur_params['utilities_skew'] = utilities_skew
        yield cur_params
#         yield {'utilities_skew': utilities_skew, 'prosociality_level': 0.3, 'hidden_embedding_size': 30, 'fairness_coeff': 0}
        
    for fairness_coeff in np.arange(0.0, 0.7, 0.1):
        cur_params = copy.deepcopy(default_cond_params)
        cur_params['fairness_coeff'] = fairness_coeff
        yield cur_params
#         yield {'utilities_skew': 0, 'prosociality_level': 0.3, 'hidden_embedding_size': 30, 'fairness_coeff': fairness_coeff}

config = {
    'name': 'cross_play_20K_episodes',
    'batch_size': 128,
    'training_episodes': 20000,
    
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
    
    'communication_mode': tune.grid_search(['cheap']),
    'match_mode': tune.grid_search(['custom_vs_custom_1', 'custom_vs_custom_2', 'default_vs_custom', 'custom_vs_default']),
    'cond_params': tune.grid_search(list(cond_params_iter())),
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
        
        if config['match_mode'].startswith('custom_vs_custom'):
            config['utility_types'] = [f'skew_{config["cond_params"]["utilities_skew"]}',]*2
            config['prosociality_levels'] = [config['cond_params']['prosociality_level'],]*2
            config['hidden_embedding_sizes'] = [config['cond_params']['hidden_embedding_size'],]*2
            config['fairness_coeffs'] = [config['cond_params']['fairness_coeff'],]*2
        elif config['match_mode'] == 'custom_vs_default':
            config['utility_types'] = [f'skew_{config["cond_params"]["utilities_skew"]}', f'skew_{default_cond_params["utilities_skew"]}']
            config['prosociality_levels'] = [config['cond_params']['prosociality_level'], default_cond_params['prosociality_level']]
            config['hidden_embedding_sizes'] = [config['cond_params']['hidden_embedding_size'], default_cond_params['hidden_embedding_size']]
            config['fairness_coeffs'] = [config['cond_params']['fairness_coeff'], default_cond_params['fairness_coeff']]
        elif config['match_mode'] == 'default_vs_custom':
            config['utility_types'] = [f'skew_{default_cond_params["utilities_skew"]}', f'skew_{config["cond_params"]["utilities_skew"]}']
            config['prosociality_levels'] = [default_cond_params['prosociality_level'], config['cond_params']['prosociality_level']]
            config['hidden_embedding_sizes'] = [default_cond_params['hidden_embedding_size'], config['cond_params']['hidden_embedding_size']]
            config['fairness_coeffs'] = [default_cond_params['fairness_coeff'], config['cond_params']['fairness_coeff']]
        else:
            raise NotImplementedError
        del config['match_mode']
        del config['cond_params']
        
        super().setup(config)
        
# each logical step of training contains several episodes, each episode is a batch of games
training_steps = config['training_episodes'] // config['episodes_per_step']
print(f'Num of training steps: {training_steps}')
print(f'Episodes per step: {config["episodes_per_step"]}')

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
