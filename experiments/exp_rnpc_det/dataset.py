#encoding=utf-8
import os
import time
import glob
import random

import numpy as np
import pandas as pd
import torch

from utils.data_preprocess import resample_image
from utils.data_augmentation import random_flip_func, random_crop_pad_func, random_flip_func, random_swap_xy


class DataCustom(torch.utils.data.Dataset):
    def __init__(self, cf, logger=None, phase='train'):
        super(DataCustom, self).__init__()
        self.phase = phase
        self.path_list = []
        self.resolution = cf.resolution
        self.input_size = cf.input_size
        self.rd_scale = cf.rd_scale
        self.cf = cf
        self.data_dirs = cf.data_dirs
        self.label_mapping = LabelMapping(cf) 

        if phase == 'train':
            self.csv_path = cf.train_csv_paths
        elif phase == 'val' or cf.infer_csv_paths == None:
            self.csv_path = cf.val_csv_paths
        else:
            self.csv_path = cf.infer_csv_paths
        
        self.path_list = []
        if self.csv_path is not None:
            for data_dir in self.data_dirs:
                for csv_path_s in self.csv_path:
                    uids = pd.read_csv(csv_path_s)['seriesuid']
                    pathes = [os.path.join(data_dir, str(uid)+'.npz') for uid in uids 
                        if os.path.exists(os.path.join(data_dir, str(uid)+'.npz'))]
                    self.path_list.extend(pathes)
        else:
            for data_dir in self.data_dirs: 
                pathes = glob.glob(os.path.join(data_dir,'*/*'))
                self.path_list.extend(pathes)
        
        print('{} Num: {}'.format(phase, len(self.path_list)))
    
    def __len__(self):
        return len(self.path_list)
        
    def __getitem__(self, idx):
        t = time.time()
        np.random.seed(int(str(t % 1)[2:7]))
        path = self.path_list[idx]

        data_dict = dict(np.load(path))
        if 'series_id' in data_dict:
            series_id = data_dict['series_id']
        else:
            series_id = os.path.splitext(os.path.basename(path))[0]
        
        spacing_zyx = data_dict['spacing_zyx']
        origin_zyx = data_dict['origin_zyx']
        transfmat = data_dict["transfmat"]
        offset_zyx = data_dict["offset_zyx"]

        image = data_dict['image_voi_zyx'].astype(np.float32).squeeze()
        centerd = data_dict['annot_center_zyxd']
        # filter annot
        if centerd.any() and len(centerd)>1:
            centerd = [c for c in centerd if (c[3:6]*spacing_zyx).max()>10]
            centerd = np.array(centerd)

        if centerd.any():
            class_label = 1
        else:
            class_label = 0
            centerd = np.zeros((1,6)) #pseudo annotation

        #normalize image
        lower_bound = np.percentile(image, 0.5)
        upper_bound = np.percentile(image, 99.5)
        mask = (image > lower_bound) & (image < upper_bound)
        image = np.clip(image, lower_bound, upper_bound)
        mn = image[mask].mean()
        std = image[mask].std()
        image = (image - mn) / std
        pad_value = (lower_bound - mn) / std

        if self.phase == 'train':
            is_random = True
        else:
            is_random = False

        # resample, random scale and crop
        sp_scale = np.array(spacing_zyx)/self.resolution
        image, centerd = resample_image(image, sp_scale, centerd=centerd, rd_scale=self.rd_scale, order=1, is_random=is_random)
        image, centerd = random_crop_pad_func(image, self.input_size, centerd=centerd, return_croppad_id=False, 
                                                pad_value=pad_value, is_random=is_random)

        if self.phase != 'infer':
            # normal aug: random flip and swap
            if self.phase == 'train':
                image, centerd = random_flip_func(image, centerd=centerd, axis=[0,1,2])
                image, centerd = random_swap_xy(image, centerd=centerd)
            
            if not class_label:
                centerd = np.array([])

            labels = self.label_mapping(centerd, image.shape, spacing_zyx).astype(np.float32)
            image = image[np.newaxis, ...].copy()
            return {'image': image, 'labels':labels, 'class_label':class_label}
            
        else:
            # infer
            image = image[None,].copy()
            return {'image': image.astype(np.float32), 'uid': series_id, 'label':class_label}


class LabelMapping(object):
    def __init__(self, cf):
        self.fpn_strides = cf.fpn_strides
        self.out_channels = cf.out_channels
        self.num_neg_fpn = cf.max_negative_target_fpn
        self.fpn_base_d = cf.fpn_base_d
        self.fpn_nodule_sizes = cf.fpn_nodule_sizes
        
    def __call__(self, centerd, input_size, spacing_zyx):

        labels = []
        for l, stride in enumerate(self.fpn_strides):
            output_z, output_y, output_x = input_size[0] // stride[0], input_size[1] // stride[1], input_size[2] // stride[1]
            label = -1 * np.ones((output_z, output_y, output_x, self.out_channels))

            #randomly select negtive samples
            if self.num_neg_fpn is not None:
                neg_z, neg_h, neg_w = np.where(label[:, :, :, 0] == -1)
                neg_idcs = random.sample(
                    range(len(neg_z)), min(self.num_neg_fpn[l], len(neg_z)))
                s_neg_z, s_neg_h, s_neg_w = neg_z[neg_idcs], neg_h[neg_idcs], neg_w[neg_idcs]
                label[neg_z, neg_h, neg_w, 0] = 0
                label[s_neg_z, s_neg_h, s_neg_w, 0] = -1

            if not centerd.any():
                labels.append(label.reshape(-1, self.out_channels))
                continue

            centers = centerd[:,:3]
            diameters = centerd[:, 3:6]
            if centerd.shape[1]==7:
                clc = centerd[:, 6]
            else:
                clc = np.ones(len(centerd))

            centers_out = centers / np.array(stride)

            if np.any(diameters <= 0):
                print('Error diameter(<0)!')
                continue

            for i, center_out in enumerate(centers_out):
                diameter_i = diameters[i] 
                center_int = center_out.astype(np.int32)
                max_value = np.array([output_z - 1, output_y - 1, output_x - 1])
                center_int = np.minimum(center_int, max_value)
                center_int = np.maximum(center_int, 0)

                label[center_int[0] - 1 : center_int[0] + 2, center_int[1] - 1 : center_int[1] + 2, center_int[2] - 1 : center_int[2] + 2, 0] = 0

                label[center_int[0], center_int[1], center_int[2], 0] = clc[i]
                label[center_int[0], center_int[1], center_int[2], 1:4] = center_out - center_int
                label[center_int[0], center_int[1], center_int[2], 4:7] = np.log(diameter_i/self.fpn_base_d[l])

            labels.append(label.reshape(-1, self.out_channels))
        
        labels = np.concatenate(labels, axis=0)

        return labels