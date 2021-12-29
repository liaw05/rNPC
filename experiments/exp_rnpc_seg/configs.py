# coding=utf-8
import os
import numpy as np
from easydict import EasyDict as edict


cf = edict()
#model setting
cf.model = 'rnpc_seg'
cf.mode_dl= 'segmentation' # 'segmentation', 'classification', 'detection'

#save dir
cf.output_dir = './output/20210527_rnpc_seg'

# train and val data
cf.train_val_test20_data_pkl = "/data/data_local_to_data/NPC_3D_code/dataset/NPC_data_0727.pkl"
cf.train_val_roi_pkl = "/data/data_local_to_data/NPC_3D_code/dataset/roi_0727.pkl"


# infer data
cf.test20_roi_pkl = "/data/data_local_to_data/NPC_3D_code/dataset/roi_test.pkl"
cf.infer_data_pkl = "/data/data_local_to_data/NPC_3D_code/dataset/NPC_data_test_0727.pkl"
cf.infer_roi_pkl = "/data/data_local_to_data/NPC_3D_code/dataset/roi_NPC_test_new0727.pkl"

# resume checkpoint path
# cf.checkpoint_path = os.path.join(cf.output_dir, 'out_models', 'model_045.tar')
cf.checkpoint_path = '/data/data_local_to_data/rNPC/output/20210527_rnpc_seg/out_models/model_002.pth'

#train
cf.num_epochs = 100
cf.weight_decay = 2e-4
cf.save_num_best_models = 5
cf.momentum = 0.9

# parameters
#zyx
cf.input_size = [80, 256, 256]

cf.optimizer = 'Adam'
cf.init_lr = 1e-3
cf.lr_adjustment_strategy = 'StepLR' #'step','consine','constant'
cf.step_size = 40
cf.gamma = 0.5

# infer
cf.infer_mask = os.path.join(cf.output_dir, 'mask')
cf.log_dir = os.path.join(cf.output_dir, 'logs')
cf.out_models = os.path.join(cf.output_dir, 'out_models')
cf.save_sources = os.path.join(cf.output_dir, 'sources')
cf.save_tensorboard = os.path.join(cf.output_dir, 'tensorboard')
