import math
import time
import torch
import sys
from torch.optim.optimizer import Optimizer
import torch.distributed as dist
# from compression import *

from mpi4py import MPI
from torch.utils.dlpack import to_dlpack, from_dlpack
import cupy
import pickle

import scatter_compress
import scatter_no_compress



def torch2cupy(tensor):
    """
    :param tensor: PyTorch CUDA tensor.
    :return: CuPy tensor.
    """
    dx = to_dlpack(tensor)
    return cupy.fromDlpack(dx)


def cupy2torch(tensor):
    """
    :param tensor: CuPy tensor.
    :return: PyTorch tensor.
    """
    dx = tensor.toDlpack()
    return from_dlpack(dx).cuda()


class ComAdam(Optimizer):
    r"""Implements Adam algorithm.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, acc_step=1):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        self.acc_step = acc_step
        super(ComAdam, self).__init__(params, defaults)

        ###
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

    def __setstate__(self, state):
        super(ComAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def get_mpi_size(self):
        return self.size

    def get_mpi_rank(self):
        return self.rank

    def step(self, closure=None, adam_freeze=False, acc_key=0):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        key = 0
        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data).cuda()
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data).cuda() + 1.0

                    state['tensor_size'] = torch.numel(p.data)

                    # every 8 numbers make up a bucket, which will be packed as a unsighed int (8bits)
                    # Therefore, at least there should be one bucket per worker to apply scatter reduce
                    if state['tensor_size'] >= self.size * 8:
                        # bucket size is 8
                        num_buckets = (state['tensor_size'] + 7) // 8
                        last_bucket_size = 8 if state['tensor_size'] % 8 == 0 else state['tensor_size'] % 8
                        # split buckets over number of workers. Number of buckets of the last chunk may be smaller
                        state['chunk_buckets'] = (num_buckets + self.size - 1) // self.size
                        state['last_chunk_buckets'] = num_buckets - state['chunk_buckets'] * (self.size - 1)
                        state['last_chunk_size'] = (state['last_chunk_buckets'] - 1) * 8 + last_bucket_size
                        
                        if self.rank == self.size - 1:
                            state['server_error'] = torch.cuda.FloatTensor(state['last_chunk_size']).fill_(0)
                        else:
                            state['server_error'] = torch.cuda.FloatTensor(state['chunk_buckets']*8).fill_(0)

                        state['worker_error'] = torch.cuda.FloatTensor(state['tensor_size']).fill_(0)

                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data).cuda()

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1
                bias_correction2 = 1

                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data)

                # Since we pack 8 numbers into 1 uint8, 
                # at least there should be one bucket per worker to apply scatter reduce
                if adam_freeze is True and state['tensor_size'] >= self.size * 8:

                    # scatter reduce
                    buffer_m = beta1 * exp_avg + (1 - beta1) * grad

                    scatter_compress.com_reduce(buffer_m,
                                                state['worker_error'],
                                                state['server_error'],
                                                state['chunk_buckets'],
                                                state['last_chunk_buckets'],
                                                state['last_chunk_size'],
                                                self.rank,
                                                self.size, self.comm)
                    
                    # scatter_no_compress.com_reduce(buffer_m, self.rank, self.size, self.comm)

                    exp_avg.set_(buffer_m)

                    if amsgrad:
                        # Maintains the maximum of all 2nd moment running avg. till now
                        torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                        # Use the max. for normalizing running avg. of gradient
                        denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                    else:
                        denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                    step_size = group['lr'] / bias_correction1

                    p.data.addcdiv_(-step_size, exp_avg, denom)

                else:

                    fgrad = grad / dist.get_world_size()

                    tensor_recv = torch.zeros_like(fgrad).cuda()

                    self.comm.Allreduce(fgrad, tensor_recv)

                    fgrad = tensor_recv

                    exp_avg.mul_(beta1).add_(1 - beta1, fgrad)
                    exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, fgrad, fgrad)
                    if amsgrad:
                        # Maintains the maximum of all 2nd moment running avg. till now
                        torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                        # Use the max. for normalizing running avg. of gradient
                        denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                    else:
                        denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                    step_size = group['lr'] / bias_correction1

                    p.data.addcdiv_(-step_size, exp_avg, denom)
        return loss
