"""Utility functions
    This file contains utility functions that are not used in the core library,
    but are useful for building models or training code using the config system.
"""
import logging
import os
import sys
import math
import shutil
import numpy as np
import matplotlib.pyplot as plt


import torch
import torch.nn.functional as F
from termcolor import cprint
from loguru import logger


# classes
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)




# functions
def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output




def adjust_learning_rate(optimizer, epoch, args):
    decay = args.lr_drop_ratio if epoch in args.lr_drop_epoch else 1.0
    lr = args.lr * decay
    global current_lr
    current_lr = lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    args.lr = current_lr
    return current_lr



def accuracy(args, output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    if isinstance(output, tuple):
        output = output[0]

    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res



