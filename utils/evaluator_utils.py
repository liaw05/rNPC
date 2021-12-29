import numpy as np
from numpy.lib.arraysetops import isin
import pandas as pd
import os
import time
import torch
from tqdm import tqdm
from torchvision.utils import make_grid


def prepare_monitoring():
    """
    creates dictionaries, where train/val metrics are stored.
    """
    metrics = {'train': {}, 'val': {}}
    return metrics


def update_tensorboard(metrics, epoch, tensorboard_writer, do_validation=True):
    keys = metrics['train'].keys()
    scalar_dict_train = {}
    scalar_dict_val = {}
    for i, key in enumerate(keys):
        scalar_dict_train[key] = metrics['train'][key][-1]
        if do_validation and key in metrics['val']:
            scalar_dict_val[key] = metrics['val'][key][-1]
            
    tensorboard_writer.add_scalars('train', scalar_dict_train, epoch)
    tensorboard_writer.add_scalars('val', scalar_dict_val, epoch)


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


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix="", logger=None):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        if logger is None:
            self.printf = print
        else:
            self.printf = logger.info

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        self.printf('  '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'