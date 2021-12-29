import torch
import torch.nn as nn
import numpy as np
import random

from models.backbone.resnet_block import PreActivationBlock, PreActivationBlockAniso
from models.backbone.resnet3d import Resnet3dAniso
from models.backbone.fpn import SFPN
from models.backbone.head import HeadFeatBnReLu
from models.utils.losses import BCELoss, RegressLoss, BCEFocalLoss


class RnpcDetCls(nn.Module):

    def __init__(self):

        super(RnpcDetCls, self).__init__()
        # cnn part
        self.n_blocks = [3, 4, 6, 3] #c2, c3, c4
        self.start_filters = 16
        self.filters = [32, 64, 64, 64]
        self.strides = [(1,2,2), (1,2,2), (1,2,2), 2]
        self.in_channels = 1
        self.out_features = 64
        self.blocks = [PreActivationBlockAniso, PreActivationBlockAniso, PreActivationBlock, PreActivationBlock]

        self.backbone = Resnet3dAniso(self.blocks, self.n_blocks, self.filters, self.strides, in_channels=self.in_channels, start_filters=self.start_filters)
        self.neck = SFPN(self.out_features, self.filters[-2:], pre_activate=False)
        
        self.head_feats = HeadFeatBnReLu(self.out_features, 4)
        self.classifier = nn.Conv3d(self.out_features, 1, 1, 1, 0, bias=True)
        self.regressor = nn.Conv3d(self.out_features, 6, 1, 1, 0, bias=True)

        self.classifier.bias.data[0].fill_(-3.19)
        self.regressor.bias.data[0:3].fill_(0.5)

        # exam classifier
        self.topK = 2
        self.classifier_exam = nn.Sequential(
            nn.Linear(self.out_features*self.topK, self.out_features, bias=True),
            nn.BatchNorm1d(self.out_features),
            nn.ReLU(inplace=True),
            nn.Linear(self.out_features, 1, bias=True),
        )

    def forward(self, inputs):
        """
        :param x: input image of shape (b, c, z, y, x)
        :return: list of output feature maps per pyramid level, each with shape (b, c, z, y, x).
        """
        features = self.backbone(inputs)
        features = self.neck(features)

        hout = self.head_feats(features[:1])

        cout = [self.classifier(f) for f in hout]
        rout = [self.regressor(f) for f in hout]
        det_out = [torch.cat([cout[i],rout[i]], dim=1) for i in range(len(cout))]

        # exam
        det_feat = det_out[0][:,:1]
        batch, c, z, y, x = det_feat.size()
        topk_class_exam, topk_ind_exam = torch.topk(det_feat.view(batch, -1), self.topK)
        feat_topk_exam = _tranpose_and_gather_feat(hout[0], topk_ind_exam) #b*k*c
        feat_topk_exam = [feat_topk_exam[:,i] for i in range(feat_topk_exam.size(1))]
        feat_topk_exam = torch.cat(feat_topk_exam, dim=1)
        logits = self.classifier_exam(feat_topk_exam)

        return det_out, logits
         

def _tranpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 4, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(4))
    # feat: batch, z*y*x, c
    feat = _gather_feat(feat, ind)
    # feat: b,k,c
    return feat


def _gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    # feat: 2 dims
    return feat


class Net(nn.Module):

    def __init__(self, cf, logger):

        super(Net, self).__init__()
        self.cf = cf
        self.logger = logger
        self.setup_seed(20)

        # bulid model
        self.encoder_q = RnpcDetCls()
        self.compute_loss_func = Loss()
    
    def setup_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
    
    def reshape_cat(self, features):
        '''features: list, reshape [b, c, z, y, x] to [b, z*y*x, c] and cat with dim 1.'''
        axes = (0, 2, 3, 4, 1)
        features = [feature.permute(*axes) for feature in features]
        features = [feature.view(feature.size()[0], -1, feature.size()[-1]) for feature in features]
        features = torch.cat(features, dim=1)
        return features

    def forward(self, batch_inputs, phase, epoch=None):
        if phase != 'infer':
            image, labels, class_label = batch_inputs['image'], batch_inputs['labels'], batch_inputs['class_label']
            return self.train_val_forward(image, labels, class_label, epoch=epoch)
        else:
            image = batch_inputs
            return self.test_forward_cls(self.encoder_q, image.cuda(), is_flip=True)

    def train_val_forward(self, image, labels, class_label=None, epoch=None):
        """
        Input:
            image: B*C*D*H*W
        Output:
            a dict of loss and a dict of feature map
        """
        device = image.device
        # compute query out
        det_out, logits = self.encoder_q(image)  # queries: Nx1
        det_out = self.reshape_cat(det_out)

        # compute loss
        loss_dict = self.compute_loss_func(det_out, labels, logits, class_label)
        feat_dict = {}
        return loss_dict, feat_dict
    
    def test_forward_cls(self, net, image, is_flip=True, axis=[2,3,4]):
        # image: b,c,z,y,x
        fpn_out = net(image)[-1]
        p_out = fpn_out.sigmoid_()
        if not is_flip:
            return p_out
        for axi in axis:
            fnp_out_fi = net(torch.flip(image, dims=[axi]))[-1]
            p_out += fnp_out_fi.sigmoid_()

        return p_out/(1+len(axis))


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.regress_loss_func = RegressLoss(reduction="mean")
        self.class_loss_func = BCEFocalLoss(gamma_pos=2, gamma_neg=2, alpha=0.25, reduction='sum')
        self.bce_loss_func = BCELoss(alpha=0.7, reduction='mean')

    def forward(self, batch_out, batch_labels, logits, class_label):
        device = batch_out.device
        batch_class_loss = torch.tensor(0).to(device).float()
        batch_offset_loss = torch.tensor(0).to(device).float()
        batch_size_loss = torch.tensor(0).to(device).float()

        for i in range(batch_out.size(0)):
            output = batch_out[i, ...]
            labels = batch_labels[i,...]
            output = output.view(-1, output.size()[-1]).contiguous()
            labels = labels.view(-1, labels.size()[-1]).contiguous()
     
            # pos sample
            pos_idcs = labels[:, 0] > 0.5
            # neg sample, -1为负样本,0忽略
            neg_idcs = labels[:, 0] < -0.5
            label_weight = pos_idcs + neg_idcs

            target_l = torch.zeros_like(labels[:, 0])
            target_l[pos_idcs] = 1.0

            num_pos = max(1, pos_idcs.sum())
            class_loss = self.class_loss_func(output[:, 0], target_l, label_weight) / num_pos
            offset_loss = self.regress_loss_func(output[:, 1:4], labels[:, 1:4], pos_idcs)
            size_loss = self.regress_loss_func(output[:, 4:7], labels[:, 4:7], pos_idcs)
            
            batch_class_loss += class_loss / batch_out.size(0)
            batch_offset_loss += offset_loss / batch_out.size(0)
            batch_size_loss += size_loss / batch_out.size(0)
        
        exam_loss = self.bce_loss_func(logits[:,0], class_label.float(), is_sigmoid=True)
        loss = batch_class_loss + batch_offset_loss + batch_size_loss + exam_loss

        #eval index
        logits = logits.sigmoid_()

        return {'loss': loss, 'exam_loss':exam_loss, 'class_loss': batch_class_loss, 'offset_loss': batch_offset_loss}