##########
# Part of the code modified from:
# https://github.com/asappresearch/emergent-comms-negotiation
##########

import torch
import numpy as np


def sample_items(batch_size, num_values=6, num_items=3, random_state=np.random):
    pool = torch.from_numpy(random_state.choice(num_values, (batch_size, num_items), replace=True))
    return pool

def sample_utility(batch_size, utility_type, num_values=6, num_items=3, random_state=np.random):
    u = torch.zeros(num_items).long()
    while u.sum() == 0:
        if utility_type == 'uniform':
            u = torch.from_numpy(random_state.choice(num_values, (batch_size, num_items), replace=True))
        elif utility_type == '1_5_only':
            u = torch.from_numpy(random_state.choice([1, 5], (batch_size, num_items), replace=True))
        elif utility_type == '3_4_5_only':
            u = torch.from_numpy(random_state.choice([3, 4, 5], (batch_size, num_items), replace=True))
        elif utility_type == 'max_on_0':
            u = random_state.choice(num_values, (batch_size, num_items), replace=True)
            u[:, 0] = 5
            u = torch.from_numpy(u)
        elif utility_type == 'min_on_0':
            u = random_state.choice(num_values, (batch_size, num_items), replace=True)
            u[:, 0] = 0
            u = torch.from_numpy(u)
        elif utility_type == 'max_on_1':
            u = random_state.choice(num_values, (batch_size, num_items), replace=True)
            u[:, 1] = 5
            u = torch.from_numpy(u)
        elif utility_type == 'min_on_1':
            u = random_state.choice(num_values, (batch_size, num_items), replace=True)
            u[:, 1] = 0
            u = torch.from_numpy(u)
        elif utility_type.startswith('skew'):
            # skew_val defined skew for all items aat once
            # (preference for certain items on average stronger than for others)
            skew_val = float(utility_type.split('_')[1])
            
            zero_skew_item_i = (num_items-1) / 2
            item_skews = [(item_i - zero_skew_item_i) / zero_skew_item_i * skew_val for item_i in range(num_items)]
            
            item_probs = []
            for item_i in range(num_items):
                zero_skew_value_i = (num_values-1)/2
                value_skews = [(value_i - zero_skew_value_i) / zero_skew_value_i * item_skews[item_i] for value_i in range(num_values)]
                cur_probs = [1/num_values * (1+value_skews[value_i]) for value_i in range(num_values)]
                assert np.allclose(np.array(cur_probs).sum(), 1)
                item_probs.append(cur_probs)
                
            item_utilities = [random_state.choice(num_values, (batch_size, 1), replace=True, p=item_probs[item_i]) for item_i in range(num_items)]
            u = torch.from_numpy(np.concatenate(item_utilities, 1))
        else:
            print(utility_type)
            raise NotImplementedError
    return u

def sample_N(batch_size, random_state=np.random):
    N = random_state.poisson(7, batch_size)
    N = np.maximum(4, N)
    N = np.minimum(10, N)
    N = torch.from_numpy(N)
    return N

def generate_batch(batch_size, utility_types, random_state=np.random):
    '''Sample game parameters (items, utilities, the maximum duration of the game)
    '''
    pool = sample_items(batch_size=batch_size, num_values=6, num_items=3, random_state=random_state)
    utilities = []
    utilities.append(sample_utility(batch_size=batch_size, num_values=6, num_items=3, random_state=random_state, utility_type=utility_types[0]))
    utilities.append(sample_utility(batch_size=batch_size, num_values=6, num_items=3, random_state=random_state, utility_type=utility_types[1]))
    N = sample_N(batch_size=batch_size, random_state=random_state)
    return {
        'pool': pool,
        'utilities': utilities,
        'N': N
    }

# def generate_test_batches(batch_size, num_batches, random_state, utility_type):
#     # r = np.random.RandomState(seed)
#     test_batches = []
#     for i in range(num_batches):
#         batch = generate_batch(batch_size=batch_size, random_state=random_state, utility_type=utility_type)
#         test_batches.append(batch)
#     return test_batches

# def hash_long_batch(int_batch, num_values):
#     seq_len = int_batch.size()[1]
#     multiplier = torch.LongTensor(seq_len)
#     v = 1
#     for i in range(seq_len):
#         multiplier[-i - 1] = v
#         v *= num_values
#     hashed_batch = (int_batch * multiplier).sum(1)
#     return hashed_batch

# def hash_batch(pool, utilities, N):
#     v = N
#     # use num_values=10, so human-readable
#     v = v * 1000 + hash_long_batch(pool, num_values=10)
#     v = v * 1000 + hash_long_batch(utilities[0], num_values=10)
#     v = v * 1000 + hash_long_batch(utilities[1], num_values=10)
#     return v

# def hash_batches(test_batches):
#     '''
#     we can store each game as a hash like:
#     [N - 1]pppuuuuuu
#     (where: [N - 1] is {4-10} - 1), ppp is the pool, like 442; and uuuuuu are the six utilities, like 354321
#     so, this integer has 10 digits, which I guess we can just store as a normal python integer?
#     '''
#     hashes = set()
#     for batch in test_batches:
#         hashed = hash_batch(**batch)
#         hashes |= set(hashed.tolist())
#         # for v in hashed:
#         #     hashes.add(v)
#     return hashes

# def overlaps(test_hashes, batch):
#     target_hashes = set(hash_batch(**batch).tolist())
#     return bool(test_hashes & target_hashes)

# def generate_training_batch(batch_size, test_hashes, random_state, utility_type):
#     batch = None
#     while batch is None or overlaps(test_hashes, batch):
#         batch = generate_batch(batch_size=batch_size, random_state=random_state, utility_type=utility_type)
#     return batch
