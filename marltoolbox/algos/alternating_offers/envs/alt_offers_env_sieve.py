##########
# Part of the code modified from:
# https://github.com/asappresearch/emergent-comms-negotiation
##########

import torch


class AliveSieve(object):
    '''Handles alive/dead masks for games
    '''
    def __init__(self, batch_size, enable_cuda):
        self.batch_size = batch_size
        self.enable_cuda = enable_cuda
        self.type_constr = torch.cuda if enable_cuda else torch
        self.alive_mask = self.type_constr.ByteTensor(batch_size).fill_(1)
        self.alive_idxes = self._mask_to_idxes(self.alive_mask)
        self.out_idxes = self.alive_idxes.clone()

    def mark_dead(self, dead_mask):
        '''Updates alive_mask, alive_idxes to mark some games as dead
        But doesn't remove those games yet
        '''
        if dead_mask.max() == 0:
            return
        dead_idxes = self._mask_to_idxes(dead_mask)
        self.alive_mask[dead_idxes] = 0
        self.alive_idxes = self._mask_to_idxes(self.alive_mask)

    def set_dead_global(self, target, v):
        '''Assign v to elements of target that are dead
        Used for assigning length to games that have just ended
        '''
        dead_idxes = self._get_dead_idxes()  # indices not in alive_mask
        if len(dead_idxes) == 0:
            return
        # alive/dead_idxes may not index the full tensor after self_sieve
        # so use out_idxes as an intermediate step
        target[self.out_idxes[dead_idxes]] = v

    def remove_dead(self):
        '''Removes all dead games from alive_mask, reducing the batch size
        '''
        self.out_idxes = self.out_idxes[self.alive_idxes]

        self.batch_size = self.alive_mask.int().sum()  # reduce batch size and only limit it to alive_idxes to reduce computation
        self.alive_mask = self.type_constr.ByteTensor(self.batch_size.item()).fill_(1)
        self.alive_idxes = self._mask_to_idxes(self.alive_mask)

    def all_dead(self):
        return self.alive_mask.max() == 0

    @staticmethod
    def _mask_to_idxes(mask):
        return mask.view(-1).nonzero().long().view(-1)

    def _get_dead_idxes(self):
        '''Returns indices which are not in alive_mask
        '''
        dead_mask = 1 - self.alive_mask
        return dead_mask.nonzero().long().view(-1)


class SievePlayback(object):
    '''
    Used for playback of all games in the batch
    For each timestep t, it will provide masks corresponding to games alive at time t
    '''
    def __init__(self, alive_masks, enable_cuda):
        self.alive_masks = alive_masks
        self.type_constr = torch.cuda if enable_cuda else torch

    def __iter__(self):
        batch_size = self.alive_masks[0].size()[0]
        global_idxes = self.type_constr.ByteTensor(batch_size).fill_(1).nonzero().long().view(-1)
        T = len(self.alive_masks)
        for t in range(T):
            self.batch_size = len(global_idxes)
            yield t, global_idxes
            mask = self.alive_masks[t]
            if mask.max() == 0:
                return
            global_idxes = global_idxes[mask.nonzero().long().view(-1)]
