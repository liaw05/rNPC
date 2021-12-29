import numpy as np
from numpy.lib.arraysetops import isin
import pandas as pd
import os
import surface_distance as surfdist
import time
import torch
import torch.nn as nn
import SimpleITK as sitk
import torch.nn.functional as F
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


class DiceLoss(nn.Module):
    '''
    计算Dice Loss。
    :param predict：预测的结果，[b,c,z,y,x]
    :param gt：ground truth,[b,c,z,y,x]
    return 返回dice_loss和前景的dice
    '''
    def __init__(self, dims=[2,3,4]):
        super(DiceLoss, self).__init__()
        self.dims = dims

    def forward(self, predict, gt, is_softmax=False):
        pd, ph, pw = predict.size(2), predict.size(3), predict.size(4)
        d, h, w = gt.size(2), gt.size(3), gt.size(4)
        if ph != h or pw != w or pd != d:
            print('ph!=h')
            predict = F.upsample(input=predict, size=(d, h, w), mode='trilinear')
            # gt = F.upsample(input=gt, size=(pd, ph, pw), mode='nearest')

        predict = predict.float()
        gt = gt.float()
       
        # intersection0 = torch.sum((1-predict)*(1-gt), dim=self.dims)
        # union0 = torch.sum((1-predict), dim=self.dims) + torch.sum((1-gt), dim=self.dims)
        # dice0 = (2. * intersection0 + 1e-5) / (union0 + 1e-5)

        intersection1 = torch.sum(predict*gt, dim=self.dims)
        union1 = torch.sum(predict, dim=self.dims) + torch.sum(gt, dim=self.dims)
        dice1 = (2. * intersection1 + 1e-5) / (union1 + 1e-5)
        # dice_loss = 1 - 0.9*dice1 - 0.1*dice0

        return 1.0 - torch.mean(dice1)


def Dice(predicts, targets):
    '''
    计算验证集的Dice。
    :param predicts：预测的结果，[b,c,z,y,x]
    :param targets：ground truth,[b,c,z,y,x]
    :param smooth：平滑因子，防止分母为0
    return 返回dice
    '''
    diceloss = DiceLoss()

    return 1.0 - diceloss(predicts, targets)



def Dice_test(predicts, targets):
    '''
    计算测试集的Dice,由于测试集通过resize转换为crop是的大小，其维度是3[z,y,x]
    :param predicts：预测的结果，[z,y,x]
    :param targets：ground truth,[z,y,x]
    :param smooth：平滑因子，防止分母为0
    return 返回dice
    '''
    smooth = 1e-5
    predicts[predicts>=0.5] = 1
    predicts[predicts<0.5] = 0
    inter = torch.sum(targets * predicts)
    union = torch.sum(targets) + torch.sum(predicts)
    dice = (2 * inter + smooth) / (union + smooth)

    return dice


def ASD_test(mask_preds, mask_gts,spacing):
    '''
    计算测试集的ASD,由于测试集通过resize转换为crop是的大小，其维度是3[z,y,x]
    :param predicts：预测的结果，[z,y,x]
    :param targets：ground truth,[z,y,x]
    return 返回asd
    '''
    mask_preds[mask_preds>=0.5] = 1
    mask_preds[mask_preds<0.5] = 0
    
    mask_preds = torch.squeeze(mask_preds).cpu().numpy()
    mask_gts = torch.squeeze(mask_gts).cpu().numpy()
    mask_pred = np.round(mask_preds).astype(bool)
    mask_gt = np.round(mask_gts).astype(bool)
    surface_distances = surfdist.compute_surface_distances(mask_gt, mask_pred, spacing_mm=(spacing[2],spacing[1],spacing[0]))
    avg_surf_dist = surfdist.compute_average_surface_distance(surface_distances)
   
    return np.mean(avg_surf_dist)


def save_nii_seg(cf,infer,outputs, targets,dict_data,origin_shape,spacing,origin,direction,zxy):
    '''
    保存测试结果
    :param outputs：预测的数据，tensor格式的数据
    :param targets：ground truth,tensor格式的数据
    :param dict_data：文件的名称
    '''
    dir = os.path.join(cf.infer_mask,infer)
    if not os.path.exists(dir):
        os.makedirs(dir)

    outputs = torch.squeeze(outputs).cpu().numpy()
    targets = torch.squeeze(targets).cpu().numpy()

    file_name = dict_data[0]
    out_file = dir + '/'+ file_name+'.nii.gz'
    
    outputs_origin = np.zeros(origin_shape)

    outputs[outputs>=0.5] = 1
    outputs[outputs<0.5] = 0
    
    outputs_origin[:,zxy[4]:zxy[5]+1,zxy[2]:zxy[3]+1] = outputs
    
    itk_image = sitk.GetImageFromArray(outputs_origin)
    itk_image.SetSpacing(spacing)
    itk_image.SetOrigin(origin)
    itk_image.SetDirection(direction)
   
    sitk.WriteImage(itk_image,out_file)