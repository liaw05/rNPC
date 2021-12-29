#encoding=utf-8
import argparse
from logging import exception
import os
import time
import csv
import math
import torch
import numpy  as np
from torch import optim
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

from utils.logger import get_logger_simple
import utils.exp_utils as exp_utils
from utils.convert import tensor_to_cuda
from utils.csv_utils import write_csv

import warnings
warnings.filterwarnings("ignore")


def inference(cf):
    """
    perform testing for a given fold(or hold out set). save stats in evaluator.
    """
    # import model and dataset
    model = exp_utils.import_module('model', cf.model_path)
    dataset = exp_utils.import_module('dataset', cf.dataset_path)

    if not os.path.exists(cf.infer_dir):
        os.makedirs(cf.infer_dir)
    logger = get_logger_simple(cf.infer_dir)

    logger.info("performing inference with model {}".format(cf.model))

    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    net = model.Net(cf, logger)
    try:
        if cf.use_ema_checkpoint:
            exp_utils.load_checkpoint_ema(cf.checkpoint_path, net, cf.momentum)
            logger.info('load ema checkpoint.')
        elif cf.use_agv_checkpoint:
            exp_utils.load_checkpoint_agv(cf.checkpoint_path, net)
            logger.info('load agv checkpoint.')
        else:        
            exp_utils.load_checkpoint_only(cf.checkpoint_path, net)
            logger.info('load checkpoint: {}'.format(cf.checkpoint_path))
    except Exception as e:
        logger.error('load checkpoint error: ', e)
        return None
    
    net = net.to(device)

    infer_dataset = dataset.DataCustom(cf, logger, phase='infer')

    # 创建csv
    header = ['seriesuid', 'probability', 'pred', 'label']
    write_csv(cf.result_csv_path, header, mul=False, mod='w')

    infer_dataloaders = DataLoader(infer_dataset,
                                    batch_size=cf.batch_size,
                                    shuffle=False,
                                    num_workers=cf.workers,
                                    pin_memory=True)

    with torch.no_grad():
        net.eval()
        for batchidx, batch_data in enumerate(infer_dataloaders):
            # torch.cuda.empty_cache()
            if not batchidx % 10:
                logger.info('inference: {}/{}'.format(batchidx, len(infer_dataloaders)))

            img = batch_data['image']
            uid = batch_data['uid']
            label = batch_data['label'].cpu().numpy()[:,None]

            #to cuda
            img = tensor_to_cuda(img, device)
            output = net(img, 'infer')

            score = output.cpu().detach().numpy()
            # pred  = np.argmax(score, axis=1)[:,None]
            pred = (score>0.5).astype(int)
            rows = np.concatenate([np.array(uid)[:,None], score, pred, label], axis=1)
            write_csv(cf.result_csv_path, rows, mul=True, mod="a")

         
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of gpus for distributed training')
    parser.add_argument('--nodes', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--gpu', type=str, default='3',
                        help='assign which gpu to use.')
    parser.add_argument('--nr', type=int, default=0,
                        help='node rank.')

    parser.add_argument('--mode', type=str, default='infer',
                        help='one out of : train / infer / train_infer')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='train batch size.')
    parser.add_argument('--workers', type=int, default=4,
                        help='numbers of workers.')
    parser.add_argument('--add_name', type=str, default='',
                        help='additiaonal name of out csv.')
    parser.add_argument('--use_multi_checkpoint', action='store_true', default=False,
                        help='use multi checkpoints to predict.')
    parser.add_argument('--use_ema_checkpoint', action='store_true', default=False,
                        help='use ema checkpoints to predict.')
    parser.add_argument('--use_latest_checkpoint', action='store_true', default=False,
                        help='use latest checkpoints to predict.')
    parser.add_argument('--use_agv_checkpoint', action='store_true', default=False,
                        help='use average checkpoints to predict.')
    parser.add_argument('--resume_to_checkpoint', action='store_true', default=False,
                        help='resuming to checkpoint and continue training.')
    parser.add_argument('--use_stored_settings', action='store_true', default=False,
                        help='load configs from existing stored_source instead of exp_source. always done for testing, '
                             'but can be set to true to do the same for training. useful in job scheduler environment, '
                             'where source code might change before the job actually runs.')
    parser.add_argument('--exp_source', type=str, default='./experiments/exp_rnpc_det',
                        help='specifies, from which source experiment to load configs, data_loader and model.')
    parser.add_argument('--stored_source', type=str, default='',
                        help='specifies, from which source experiment to load configs, data_loader and model.')
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.ngpus = torch.cuda.device_count()
    add_name = args.add_name

    cf = exp_utils.prep_exp_ddp(args.exp_source, args.stored_source, args.use_stored_settings, is_train=False)
    # cf = utils.prep_exp(args.exp_source, args.stored_source, args.use_stored_settings, is_train=False)
    
    infer_dir = os.path.dirname(cf.result_csv_path)
    if not os.path.exists(infer_dir):
        os.makedirs(infer_dir)

    cf.infer_dir = infer_dir
    cf.batch_size = args.batch_size
    cf.workers = args.workers

    # use multi/single checkpoint
    cf.use_ema_checkpoint = False
    cf.use_agv_checkpoint = False
    if args.use_multi_checkpoint:
        select_model_dir = os.path.join(cf.out_models, 'select_model')
        save_epoch = [int(fn[:-4].split('_')[-1]) for fn in os.listdir(select_model_dir)]
        save_epoch.sort()
        for epoch in save_epoch:
            cf.checkpoint_path = os.path.join(select_model_dir, 'model_%03d.tar' % epoch)
            cf.result_csv_path = os.path.join(cf.output_dir, 'results-'+cf.output_dir.split('/')[-1]+'_e{:0=3}{}.csv'.format(epoch, add_name))
            inference(cf)
    else:    
        inference(cf)
    
    # latest model
    if args.use_latest_checkpoint:   
        select_model_dir = os.path.join(cf.out_models, 'latest_model')
        save_epoch = [int(fn[:-4].split('_')[-1]) for fn in os.listdir(select_model_dir)]
        save_epoch.sort()
        epoch = save_epoch[-1]
        cf.checkpoint_path = os.path.join(select_model_dir, 'model_%03d.tar' % epoch)
        cf.result_csv_path = os.path.join(cf.output_dir, 'results-'+cf.output_dir.split('/')[-1]+'_lst_e{:0=3}{}.csv'.format(epoch, add_name))
        inference(cf)
