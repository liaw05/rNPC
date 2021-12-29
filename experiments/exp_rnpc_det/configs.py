# coding=utf-8
import os
import numpy as np
from easydict import EasyDict as edict


cf = edict()
#model setting
cf.model = 'rnpc_det_cls'
cf.mode_dl= 'classification' # 'segmentation', 'classification', 'detection'

#save dir
cf.output_dir = './output/20210527_rnpc_det_cls'

# train and val data
cf.data_dirs = [
    "/data_local/data/train_data/rNPC/rNPC_SYSUCC_trainval_t1c_voi_newn4/",
    "/data_local/data/train_data/rNPC/rNPC_SYSUCC_test_t1c_voi_newn4/",
    "/data_local/data/train_data/rNPC/rNPC_ex_hospital_t1c_voi_newn4/",
]

cf.train_csv_paths = [
    "/data/Metrics/rNPC_SYSUCC/rNPCSYSUCC_train.csv",
    "/data/Metrics/rNPC_ex_test/rNPC_foshan_info_t1c_train.csv",
    "/data/Metrics/rNPC_ex_test/rNPC_foshan_info_t1c_train.csv",
]

cf.val_csv_paths = [
    "/data/Metrics/rNPC_SYSUCC/rNPCSYSUCC_val.csv",
    "/data/Metrics/rNPC_ex_test/rNPC_foshan_info_t1c_val.csv",
]
# infer data
cf.infer_csv_paths = [
    "/data/Metrics/rNPC_SYSUCC/rNPCSYSUCC_val.csv",
    "/data/Metrics/rNPC_test/rNPC_SYSUCC_test01_04.csv",
    "/data/Metrics/rNPC_ex_test/rNPC_exhospital_test_t1c.csv",
]

# resume checkpoint path
# cf.checkpoint_path = os.path.join(cf.output_dir, 'out_models', 'model_045.tar')
cf.checkpoint_path = 'output/det_cls_model/model_059.tar'

#train
cf.num_epochs = 60
cf.weight_decay = 2e-4
cf.save_num_best_models = 5
cf.save_num_latest_models = 2
cf.save_model_per_epochs = 40
cf.validataion_per_epoch = 2
cf.warm_epoch = 4
cf.momentum = 0.8
cf.rest_time = 0.1
cf.do_validation = True
cf.infer_zoom_crop_order = False

# parameters
#zyx
cf.out_channels = 7
cf.input_size = [32, 256, 256]
cf.resolution = [6, 1, 1]
cf.fpn_base_d = [(6,40,40)]
cf.fpn_nodule_sizes = [[0, 160]]
cf.fpn_strides = [(1,8,8)]
# 负样本最多选择个数
cf.max_negative_target_fpn = [6000]  # None
cf.window = np.array([0, 0, 0, cf.input_size[0]-1, cf.input_size[1]-1, cf.input_size[2]-1])

cf.optimizer = 'SGD'
cf.init_lr = 1e-2
cf.lr_adjustment_strategy = 'consine' #'step','consine','constant'
#data augment
cf.rd_scale = (0.8, 1.2)

# infer
cf.infer_mask = os.path.join(cf.output_dir, 'mask')
cf.log_dir = os.path.join(cf.output_dir, 'logs')
cf.out_models = os.path.join(cf.output_dir, 'out_models')
cf.save_sources = os.path.join(cf.output_dir, 'sources')

cf.is_convert_to_voxel_coord = True
cf.is_convert_to_world_coord = True # if True, convert coord to world coord, if False, convert to voxel coord.
cf.result_csv_path = os.path.join(cf.output_dir, 'results-20210527_rnpc_det_cls.csv')
