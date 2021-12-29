from __future__ import print_function, division
import torch.nn.functional as F
import torch.nn as nn
import torch


class RegressLoss(nn.Module):
    def __init__(self, reduction='sum'):
        super().__init__()
        self.smoothl1_loss = nn.SmoothL1Loss(reduction=reduction)

    def forward(self, inputs, target, label_weight):
        '''Calculate the smooth-ls loss.
        Args:
            inputs (float tensor of size [batch_num, class_num]):
                The direct prediction of classification fc layer.
            target (float tensor of size [batch_num, class_num]):
                Binary class target for each sample.
            label_weight (float tensor of size [batch_num, class_num]):
                the value is 1 if the sample is valid and 0 if ignored.
        '''
        
        valid_indices = label_weight > 0
        device = inputs.device
        if valid_indices.sum():
            t = target[valid_indices]
            p = inputs[valid_indices]
            return self.smoothl1_loss(p, t)
        else:
            return torch.tensor(0).to(device).float()


class BCEFocalLoss(nn.Module):

    def __init__(self, gamma_pos=2, gamma_neg=2, alpha=0.75, reduction='sum'):
        '''reduction: mean/sum'''
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, target, label_weight=None, loss_weight=None):
        '''Calculate the focal loss.
        Args:
            inputs (float tensor of size [batch_num, class_num]):
                The direct prediction of classification fc layer.
            target (float tensor of size [batch_num, class_num]):
                Binary class target for each sample.
            label_weight (float tensor of size [batch_num, class_num]):
                the value is 1 if the sample is valid and 0 if ignored.
        '''
        if label_weight is not None:
            valid_indices = label_weight > 0
            t = target[valid_indices]
            p = inputs[valid_indices]
        else:
            t = target
            p = inputs
            
        p = F.sigmoid(p)

        # pt = p if t > 0 else 1-p
        pt = p * t + (1 - p) * (1 - t)
        # w = alpha if t > 0 else 1-alpha
        w = self.alpha*t + (1-self.alpha)*(1-t)
        w = t*w*(1-pt).pow(self.gamma_pos) + (1-t)*w*(1-pt).pow(self.gamma_neg)
        if loss_weight is not None:
            loss_weight = loss_weight[valid_indices]
            w = w*loss_weight
        w = w.detach()

        return F.binary_cross_entropy(p, t, w, reduction=self.reduction)


class BCELoss(nn.Module):

    def __init__(self, alpha=None, reduction='sum'):
        '''reduction: mean/sum'''
        super().__init__()
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, target, is_sigmoid=False):
        '''Calculate the focal loss.
        Args:
            inputs (float tensor of size [batch_num, class_num]):
                The direct prediction of classification fc layer.
            target (float tensor of size [batch_num, class_num]):
                Binary class target for each sample.
        '''
        device = inputs.device
        t = target
        p = inputs
        if t.size(0)==0:
            return torch.tensor(0).to(device).float()

        if is_sigmoid:
            p = F.sigmoid(p)

        # w = alpha if t > 0 else 1-alpha
        if self.alpha is not None:
            w = self.alpha * t + (1 - self.alpha) * (1 - t)
        else:
            w = None

        return F.binary_cross_entropy(p, t, w, reduction=self.reduction)
    