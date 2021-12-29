import torch.nn as nn
import torch.nn.functional as F
import torch
from functools import partial
from torch.autograd import Variable


def conv3x3x3(in_planes, out_planes, stride=1, dilation=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        dilation=dilation,
        stride=stride,
        padding=dilation,
        bias=False)


def conv1x3x3(in_planes, out_planes, stride=1, dilation=1):
    # 1x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=(1,3,3),
        dilation=dilation,
        stride=stride,
        padding=(0,1,1),
        bias=False)


def downsample_basic_block(x, planes, stride, no_cuda=False):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if not no_cuda:
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out



class PreActivationBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(PreActivationBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride, dilation=dilation)
        self.bn1 = nn.BatchNorm3d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        return out


class PreActivationBlockAniso(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(PreActivationBlockAniso, self).__init__()
        self.conv1 = conv1x3x3(inplanes, planes, stride, dilation=dilation)
        self.bn1 = nn.BatchNorm3d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv1x3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        return out