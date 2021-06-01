import copy
import numpy as np


default_cond_params = {'utilities_skew': 0, 'prosociality_level': 0.3, 'hidden_embedding_size': 30, 'fairness_coeff': 0}
# default_cond_params = {'utilities_skew': 0, 'prosociality_level': 1.0, 'hidden_embedding_size': 30, 'fairness_coeff': 0}

def cond_params_iter():
    yield default_cond_params  # default parameters
    
    for utilities_skew in [round(i, 1) for i in np.arange(-1, 1.1, 0.4)]:
        cur_params = copy.deepcopy(default_cond_params)
        cur_params['utilities_skew'] = utilities_skew
        yield cur_params
        
    for fairness_coeff in [round(i, 1) for i in np.arange(0.0, 0.7, 0.1)]:
        cur_params = copy.deepcopy(default_cond_params)
        cur_params['fairness_coeff'] = fairness_coeff
        yield cur_params
        
    for prosociality_level in [round(i, 1) for i in np.arange(0, 1.1, 0.2)]:
        cur_params = copy.deepcopy(default_cond_params)
        cur_params['prosociality_level'] = prosociality_level
        yield cur_params
        
    for hidden_embedding_size in [5, 10, 20, 40, 80, 160]:
        cur_params = copy.deepcopy(default_cond_params)
        cur_params['hidden_embedding_size'] = hidden_embedding_size
        yield cur_params
        
def cond_params_iter_pop_training():
    yield default_cond_params  # default parameters
        
    for fairness_coeff in [round(i, 1) for i in np.arange(0.0, 0.6, 0.1)]:
        cur_params = copy.deepcopy(default_cond_params)
        cur_params['fairness_coeff'] = fairness_coeff
        yield cur_params
        
def round_cond_params(cond_params):
    res = cond_params.copy()
    res['utilities_skew'] = round(res['utilities_skew'], 1)
    res['fairness_coeff'] = round(res['fairness_coeff'], 1)
    res['prosociality_level'] = round(res['prosociality_level'], 1)
    return res
        
#     for fairness_coeff in np.arange(0.0, 0.6, 0.1):
#         cur_params = copy.deepcopy(default_cond_params)
#         cur_params['fairness_coeff'] = fairness_coeff
#         cur_params['prosociality_level'] = 1.0
#         yield cur_params