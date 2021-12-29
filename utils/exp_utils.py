from torch.utils.tensorboard import SummaryWriter
import os
import sys
import csv
import torch.nn as nn
import logging
import subprocess
import glob,math
from models.rnpc_seg import Net
from utils.evaluator_utils import DiceLoss
from torch import optim
import torch
import numpy as np
import pandas as pd
import importlib
from collections import OrderedDict


def import_module(name, path):
    """
    correct way of importing a module dynamically in python 3.
    :param name: name given to module instance.
    :param path: path to module.
    :return: module: returned module instance.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prep_exp(exp_source, stored_source, use_stored_settings=False, is_train=True):
    """
    I/O handling, creating of experiment folder structure. Also creates a snapshot of configs/model scripts and copies them to the exp_dir.
    This way the exp_dir contains all info needed to conduct an experiment, independent to changes in actual source code. Thus, training/inference of this experiment can be started at anytime. Therefore, the model script is copied back to the source code dir as tmp_model (tmp_backbone).
    Provides robust structure for cloud deployment.
    :param dataset_path: path to source code for specific data set. (e.g. medicaldetectiontoolkit/lidc_exp)
    :param exp_path: path to experiment directory.
    :param use_stored_settings: boolean flag. When starting training: If True, starts training from snapshot in existing experiment directory, else creates experiment directory on the fly using configs/model scripts from source code.
    :return:
    """

    if use_stored_settings:
        cf_file = import_module('cf', os.path.join(stored_source, 'configs.py'))
        cf = cf_file.cf
        cf.model_path = os.path.join(stored_source, 'model.py')
        cf.dataset_path = os.path.join(stored_source, 'dataset.py')

    else:
        cf_file = import_module('cf', os.path.join(exp_source, 'configs.py'))
        cf = cf_file.cf
        cf.dataset_path = os.path.join(exp_source, 'dataset.py')
        cf.model_path = 'models/{}.py'.format(cf.model)

    if is_train and not use_stored_settings:
        if not os.path.exists(cf.save_sources):
            os.makedirs(cf.save_sources)
        subprocess.call('cp {} {}'.format(cf.model_path, os.path.join(cf.save_sources, 'model.py')), shell=True)
        subprocess.call('cp {} {}'.format(os.path.join(exp_source, 'configs.py'), os.path.join(cf.save_sources, 'configs.py')), shell=True)
        subprocess.call('cp {} {}'.format(os.path.join(exp_source, 'dataset.py'), os.path.join(cf.save_sources, 'dataset.py')), shell=True)

    if not os.path.exists(cf.log_dir):
        os.makedirs(cf.log_dir)
    if not os.path.exists(cf.out_models):
        os.makedirs(cf.out_models)

    return cf


def prep_exp_ddp(exp_source, stored_source, use_stored_settings=False, is_train=True):
    """
    I/O handling, creating of experiment folder structure. Also creates a snapshot of configs/model scripts and copies them to the exp_dir.
    This way the exp_dir contains all info needed to conduct an experiment, independent to changes in actual source code. Thus, training/inference of this experiment can be started at anytime. Therefore, the model script is copied back to the source code dir as tmp_model (tmp_backbone).
    Provides robust structure for cloud deployment.
    :param dataset_path: path to source code for specific data set. (e.g. medicaldetectiontoolkit/lidc_exp)
    :param exp_path: path to experiment directory.
    :param use_stored_settings: boolean flag. When starting training: If True, starts training from snapshot in existing experiment directory, else creates experiment directory on the fly using configs/model scripts from source code.
    :return:
    """

    if use_stored_settings:
        cf_file = import_module('cf', os.path.join(stored_source, 'configs.py'))
        cf = cf_file.cf
        cf.save_sources = stored_source
        cf.model_path = os.path.join(stored_source, 'model.py')
        cf.dataset_path = os.path.join(stored_source, 'dataset.py')

    else:
        cf_file = import_module('cf', os.path.join(exp_source, 'configs.py'))
        cf = cf_file.cf
        cf.dataset_path = os.path.join(exp_source, 'dataset.py')
        cf.model_path = 'models/{}.py'.format(cf.model)

    if is_train and not use_stored_settings:
        if not os.path.exists(cf.save_sources):
            os.makedirs(cf.save_sources)
        subprocess.call('cp {} {}'.format(cf.model_path, os.path.join(cf.save_sources, 'model.py')), shell=True)
        subprocess.call('cp {} {}'.format(os.path.join(exp_source, 'configs.py'), os.path.join(cf.save_sources, 'configs.py')), shell=True)
        subprocess.call('cp {} {}'.format(os.path.join(exp_source, 'dataset.py'), os.path.join(cf.save_sources, 'dataset.py')), shell=True)

    if not os.path.exists(cf.log_dir):
        os.makedirs(cf.log_dir)
    if not os.path.exists(cf.out_models):
        os.makedirs(cf.out_models)

    return cf

def get_tensorboard_seg(cf):
    # tensorboard 可视化
    tensorboard_path = cf.save_tensorboard

    my_tensorboard = SummaryWriter(tensorboard_path)

    return my_tensorboard


def getLog_seg(cf):
    '''
    日志文件函数
    :param args：命令行解析对象
    return 返回日志文件对象
    '''
    filename = open(cf.log_dir +'/log.log', encoding="utf-8", mode="a")  # w会覆盖原来的log
    logging.basicConfig(
            stream=filename,
            level=logging.INFO,
            format='%(asctime)s:%(levelname)s:%(message)s'
        )
    return logging



def getLoss_seg():
    # 损失函数
    diceloss = DiceLoss()
    
    return diceloss


def getOptim_seg(cf, model):
    # 优化器
    if cf.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(),lr=cf.init_lr,weight_decay=cf.weight_decay,betas=(0.9, 0.999), eps=1e-08)  # weight_decay=1e-5
    elif cf.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), momentum=cf.momentum, lr=cf.init_lr, weight_decay=cf.weight_decay)  # momentum=args.momentum

    return optimizer


def getLr_scheduler_seg(cf, optimizer):
    if cf.lr_adjustment_strategy == 'StepLR':
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cf.step_size, gamma=cf.gamma)
    elif cf.lr_adjustment_strategy == 'MultiStepLR':
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cf.milestones, gamma=cf.gamma, last_epoch=-1)
    elif cf.lr_adjustment_strategy == 'consine':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3, T_mult=2, eta_min=1e-5, last_epoch=-1)
    elif cf.lr_adjustment_strategy == 'CosineAnnealingLR':
        # T_max一次学习率周期的迭代次数，即 T_max 个 epoch 之后重新设置学习率
        # eta_min最小学习率，即在一个周期中，学习率最小会下降到 eta_min，默认值为 0
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-4, last_epoch=-1)

    return lr_scheduler


def load_checkpoint_seg(cf,net, optimizer=None, is_fine_tuning=False):
    checkpoint_path = cf.checkpoint_path 
    if os.path.exists(checkpoint_path):
        model_dict = net.state_dict()
        checkpoint = torch.load(checkpoint_path)
        pretrained_dict = checkpoint['state_dict']
        
        if is_fine_tuning:
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'last_linear' not in k}
        else:
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

        model_dict.update(pretrained_dict)
        net.load_state_dict(model_dict)

        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])

        return checkpoint['epoch'], checkpoint['val_dice']


    else:
        return None,None


def adjust_lr_cos_seg(optimizer, epoch, max_epochs=300, warm_epoch=10, lr_init=1e-3):
    """Decay the learning rate based on schedule"""
    # cosine lr schedule
    if epoch < warm_epoch:
        cur_lr = lr_init * (epoch*(1.0-0.1)/warm_epoch + 0.1)
    else:
        cur_lr = lr_init * 0.5 * (1. + math.cos(math.pi * (epoch-warm_epoch)/ (max_epochs-warm_epoch)))

    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = lr_init
        else:
            param_group['lr'] = cur_lr


def model_select_save_seg(cf, val_aucs, model, epoch, trained_epochs):

    index_ranking = np.argsort(-np.array(val_aucs))  # np.argsort从小到大排序,返回排序后的下标
    epoch_ranking = np.array(trained_epochs)[index_ranking]
    
    # check if current epoch is among the top-k epchs.
    if epoch in epoch_ranking[:5]:
        model_path = cf.out_models
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        
        torch.save(model.state_dict(), model_path+'/model_%03d.pth' % epoch)

        # delete params of the epoch that just fell out of the top-k epochs.
        #删除c-index最小的模型
        if len(epoch_ranking) > 5:
            epoch_rm = epoch_ranking[5]
            subprocess.call('rm {}'.format(os.path.join(model_path, 'model_%03d.pth' % epoch_rm)), shell=True)



def save_model_seg(cf, net, optimizer, epoch, val_dice):
    checkpoint_path = cf.out_models

    if torch.cuda.device_count() > 1:
        net_state_dict = net.module.state_dict()
    else:
        net_state_dict = net.state_dict()

    torch.save({
        'epoch': epoch,
        'state_dict': net_state_dict,
        'optimizer': optimizer.state_dict(),
        'val_dice': val_dice,
        }, os.path.join(checkpoint_path, 'checkpoint_model_final.tar'))



def model_select_save(cf, net, optimizer, monitor_metrics, epoch, trained_epochs):
    keys = monitor_metrics['train'].keys()
    for key in ['auc', 'acc', 'dice', 'c-index', 'loss']:
        if key in keys:
            break
    if cf.do_validation:
        val_losses = monitor_metrics['val'][key]
    else:
        val_losses = monitor_metrics['train'][key]

    val_losses = np.array(val_losses)
    index_ranking = np.argsort(val_losses) if key=='loss' else np.argsort(-val_losses)
    epoch_ranking = np.array(trained_epochs)[index_ranking]
    
    # check if current epoch is among the top-k epchs.
    if epoch in epoch_ranking[:cf.save_num_best_models]:
        select_model_dir = os.path.join(cf.out_models, 'select_model')
        if not os.path.exists(select_model_dir):
            os.makedirs(select_model_dir)
        save_model(net, optimizer, epoch, select_model_dir)
        # 更新为最佳模型地址
        cf.checkpoint_path = os.path.join(select_model_dir, 'model_%03d.tar' % epoch)

        # delete params of the epoch that just fell out of the top-k epochs.
        if len(epoch_ranking) > cf.save_num_best_models:
            epoch_rm = epoch_ranking[cf.save_num_best_models]
            subprocess.call('rm {}'.format(os.path.join(select_model_dir, 'model_%03d.tar' % epoch_rm)), shell=True)


def save_latest_model(cf, net, optimizer, epoch):
    latest_model_dir = os.path.join(cf.out_models, 'latest_model')
    if not os.path.exists(latest_model_dir):
        os.makedirs(latest_model_dir)
    if not epoch % cf.save_model_per_epochs:
        save_model(net, optimizer, epoch, cf.out_models)
    
    # save lastest checkpoints
    if hasattr(cf, 'save_num_latest_models'):     
        save_model(net, optimizer, epoch, latest_model_dir)
        save_epoch = [int(fn[:-4].split('_')[-1]) for fn in os.listdir(latest_model_dir)]
        if len(save_epoch) > cf.save_num_latest_models:       
            save_epoch.sort(reverse=False)
            epoch_rm = save_epoch[0]
            subprocess.call('rm {}'.format(os.path.join(latest_model_dir, 'model_%03d.tar' % epoch_rm)), shell=True)


def save_model(net, optimizer, epoch, model_dir):
    if hasattr(net, 'module'):
        net_state_dict = net.module.state_dict()
    else:
        net_state_dict = net.state_dict()

    torch.save({
        'epoch': epoch+1,
        'state_dict': net_state_dict,
        'optimizer': optimizer.state_dict(),
        }, 
        os.path.join(model_dir, 'model_%03d.tar' % epoch),
        _use_new_zipfile_serialization=False)


def load_checkpoint(checkpoint_path, net, optimizer=None):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    pretrain_dict = checkpoint['state_dict']
    try:
        if hasattr(net, 'module'):
            pretrain_dict = {'module.'+k:v for k,v in pretrain_dict.items()}
        net.load_state_dict(pretrain_dict)
        print('***load parameters completely***')
    except:
        print('***load part parameters***')
        model_dict = net.state_dict()
        pretrain_dict = {k:v for k,v in pretrain_dict.items() if k in model_dict}
        print('Num of param:', len(pretrain_dict))
        model_dict.update(pretrain_dict)
        net.load_state_dict(model_dict)

    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])

    return checkpoint['epoch']


def load_checkpoint_only(checkpoint_path, net):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    pretrain_dict = checkpoint['state_dict']
    try:
        if hasattr(net, 'module'):
            pretrain_dict = {'module.'+k:v for k,v in pretrain_dict.items()}
        net.load_state_dict(pretrain_dict)
        print('***load parameters completely***')
    except:
        print('***load part parameters***')
        model_dict = net.state_dict()
        pretrain_dict = {k:v for k,v in pretrain_dict.items() if k in model_dict}
        print('Num of param:', len(pretrain_dict))
        model_dict.update(pretrain_dict)
        net.load_state_dict(model_dict)
