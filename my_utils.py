'''
This module contains some utility classes and functions used in notebooks.

Including:
  ReplayBuffer - to store and sample batches of replays;
  construct_nn - to create sequential neural network of given size and with given output layer.
'''

import torch
from torch import nn

class ReplayBuffer:
    '''
    ReplayBuffer stores replays and samples however many needed when requested

    Args:
        obs_dim (int): Specifies observation shape
        act_dim (int): Specifies action shape
        size (int): Specifies the limit of how many replays can be stored.
            When reached, old ones are overwritten by newer replays
        dev (torch.device): Specifies which device to use
    '''

    def __init__(self, obs_dim, act_dim, size, dev):
        self.obs_buf = torch.zeros((size, obs_dim), dtype=torch.float32).to(dev)
        self.act_buf = torch.zeros((size, act_dim), dtype=torch.float32).to(dev)
        self.rew_buf = torch.zeros(size, dtype=torch.float32).to(dev)
        self.next_obs_buf = torch.zeros((size, obs_dim), dtype=torch.float32).to(dev)
        self.done_buf = torch.zeros(size, dtype=torch.float32).to(dev)

        self.ptr, self.size, self.limit = 0, 0, size

    def put(self, obs, act, rew, next_obs, done):
        '''
        Put a new replay entry into the buffer

        Args:
            obs (torch.tensor): observation from gym
            act (torch.tensor): an action taken upon the observation `obs`
            rew (torch.tensor): reward received after taking the action
            next_obs (torch.tensor): the new observation, typically a result of taking action
            done (torch.tensor): specifies whether the action was terminal
        '''

        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.next_obs_buf[self.ptr] = next_obs
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.limit
        self.size = min(self.size + 1, self.limit)

    def sample_batch(self, batch_size):
        '''
        Sample a batch of specified size from the buffer

        Args:
            batch_size (int): specifies how many elements to put into the batch

        Returns:
            dict: the requested batch
        '''

        idx = torch.randint(0, self.size, (batch_size,))
        return {
            'obs': self.obs_buf[idx],
            'act': self.act_buf[idx],
            'rew': self.rew_buf[idx],
            'next_obs': self.next_obs_buf[idx],
            'done': self.done_buf[idx]
        }

    def __len__(self):
        return self.size


class Logger():
    '''
    Logger stores valuable information and gives it out when requested in a fancy format
    '''

    def __init__(self):
        self.logs = dict()
        self.summary_ptrs = dict()
        self.summary_funcs = dict()

    def reset(self):
        '''
        Reset the logger to its initial state
        '''
        self.logs.clear()
        self.summary_ptrs.clear()
        self.summary_funcs.clear()

    def add_attribute(self, name, summary_funcs):
        '''
        Add an attribute to fill later

        Args:
            name (str): name of the attribute, use it to store data
            summary_funcs (list(callable(list))): functions to call when summarizing this attribute

        Raises:
            KeyError: there's already an entry with provided name
        '''
        if name in self.logs.keys():
            raise KeyError(f'Attribute {name} already exits!')

        self.logs[name] = []
        self.summary_ptrs[name] = 0

        if isinstance(summary_funcs, list):
            self.summary_funcs[name] = summary_funcs
        else:
            self.summary_funcs[name] = [summary_funcs]

    def put(self, name, value):
        '''
        Put another data-point of attribute with given name

        Args:
            name (str): name of the attribute, use it to store data
            value (any number or ndarray): data to log
        '''
        if name not in self.logs.keys():
            raise KeyError(f'No such attribute: {name}')
        self.logs[name].append(value)

    def summarize(self, *, attributes=None, fmt=True, from_beginning=False):
        '''
        Summarize all the attributes using functions provided earlier and put results into a string

        Args:
            attributes (list of str): which attributes to summarize over
            from_beginning (bool): whether to summarize from the beginning or last summarize call

        Returns:
            str: summary of stored data
        '''
        summary = []
        iterate_over = attributes if attributes else self.logs.keys()
        for attr in iterate_over:
            log = self.logs[attr]

            if from_beginning:
                ptr = 0
            else:
                ptr = self.summary_ptrs[attr]
                self.summary_ptrs[attr] = len(log)

            if fmt:
                summary += \
                    [f'{attr}_{f.__name__}={f(log[ptr:]):.4f}' for f in self.summary_funcs[attr]]
            else:
                summary += \
                    [(f'{attr}_{f.__name__}', f(log[ptr:])) for f in self.summary_funcs[attr]]

        return '; '.join(summary) if fmt else summary


def construct_nn(sizes: list, output=nn.Identity):
    '''
    Costructs nn.Sequential of layers of given size and with speciied output.

    Args:
        sizes: specifies size of each layer
        output: activation function of the last layer, nn.Identity by default

    Returns:
        nn.Sequential: has Linear layers of given size, separated by ReLU layers,
        and with specified output
    '''
    layers = []
    for i in range(len(sizes) - 1):
        act = nn.ReLU if i < len(sizes) - 2 else output
        layers += [nn.Linear(sizes[i], sizes[i+1]), act()]
    return nn.Sequential(*layers)
