import torch
import torch.nn as nn
import torch.nn.functional as F


def passthrough(x, **kwargs):
    return x

def ELUCons(elu, nchan):
    if elu:
        return nn.ELU(inplace=True)
    else:
        return nn.PReLU(nchan)

# normalization between sub-volumes is necessary
# for good performance
class ContBatchNorm3d(nn.modules.batchnorm._BatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))
        # super(ContBatchNorm3d, self)._check_input_dim(input)

    def forward(self, input):
        self._check_input_dim(input)
        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            True, self.momentum, self.eps)


class LUConv(nn.Module):
    def __init__(self, nchan, elu,kernel,padding_,dilation_):
        '''
        function: 经过卷积，bn,relu操作后的输出结果
        nchan：输入的通道数
        elu: true=nn.ELU,false=nn.PReLU
        kernel：卷积核大小
        padding_：填充的大小
        dilation_：空洞卷积率
        return 经过卷积，bn,relu操作后的输出结果
        '''
        super(LUConv, self).__init__()
        self.relu1 = ELUCons(elu, nchan)
        self.conv1 = nn.Conv3d(nchan, nchan, kernel_size=kernel, padding=padding_,dilation=dilation_)
        self.bn1 = ContBatchNorm3d(nchan)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        return out


def _make_nConv(nchan, depth, elu,kernel,padding_):
    '''
    function: 制作多层卷积
    nchan：输入的通道数
    depth: 卷积层数量
    elu: true=nn.ELU,false=nn.PReLU
    kernel：卷积核大小
    padding_：填充的大小
    dropout：是否dropout
    return nn.Sequential
    '''
    layers = []
    for _ in range(depth):
        layers.append(LUConv(nchan, elu,kernel,padding_,1))
    return nn.Sequential(*layers)


class InputTransition(nn.Module):
    def __init__(self, outChans, elu):
        '''
        function: 输入的第一层的卷积操作
        outChans：输出的通道数
        elu: true=nn.ELU,false=nn.PReLU
        '''
        super(InputTransition, self).__init__()
        self.conv1 = nn.Conv3d(1, 16, kernel_size=(1,5,5), padding=(0,2,2))
        self.bn1 = ContBatchNorm3d(16)
        self.relu1 = ELUCons(elu, 16)

    def forward(self, x):
        # do we want a PRELU here as well?
        out = self.bn1(self.conv1(x))
        # split input in to 16 channels
        x16 = torch.cat((x, x, x, x, x, x, x, x,
                         x, x, x, x, x, x, x, x), 1)
        out = self.relu1(torch.add(out, x16))
        return out


class DownTransition(nn.Module):
    def __init__(self, inChans, nConvs, elu,kernel,padding_,down_kernel,stride_, dropout=False):
        '''
        function: encode stage
        inChans：输入的通道数
        nConvs：卷积层数量
        elu: true=nn.ELU,false=nn.PReLU
        kernel：卷积核大小
        padding_：填充的大小
        dropout：是否dropout
        '''
        super(DownTransition, self).__init__()
        outChans = 2*inChans
        self.down_conv = nn.Conv3d(inChans, outChans, kernel_size=down_kernel, stride=stride_)
        self.bn1 = ContBatchNorm3d(outChans)
        self.do1 = passthrough
        self.relu1 = ELUCons(elu, outChans)
        self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(outChans, nConvs, elu,kernel,padding_)
     

    def forward(self, x):
        down = self.relu1(self.bn1(self.down_conv(x)))
        out = self.do1(down)
        out = self.ops(out)
    
        out = self.relu2(torch.add(out, down))
        return out


class UpTransition(nn.Module):
    def __init__(self, inChans, outChans, nConvs, elu,kernel,padding_,down_kernel,stride_, dropout=False):
        '''
        function: decode stage
        inChans：输入的通道数
        outChans: 输出的通道数
        nConvs：卷积层数量
        elu: true=nn.ELU,false=nn.PReLU
        kernel：卷积核大小
        padding_：填充的大小
        dropout：是否dropout
        '''
        super(UpTransition, self).__init__()
        self.up_conv = nn.ConvTranspose3d(inChans, outChans // 2, kernel_size=down_kernel, stride=stride_)
        self.bn1 = ContBatchNorm3d(outChans // 2)
        self.do1 = passthrough
        self.do2 = nn.Dropout3d()
        self.relu1 = ELUCons(elu, outChans // 2)
        self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(outChans, nConvs, elu,kernel,padding_)
       

    def forward(self, x, skipx):
        out = self.do1(x)
        skipxdo = self.do2(skipx)
        # t = self.up_conv(out)
        out = self.relu1(self.bn1(self.up_conv(out)))
        xcat = torch.cat((out, skipxdo), 1)
        out = self.ops(xcat)
     
        out = self.relu2(torch.add(out, xcat))
        return out


class OutputTransition(nn.Module):
    def __init__(self, inChans, elu, nll):
        '''
        function: 最后一层的卷积操作
        inChans：输入的通道数
        elu: true=nn.ELU,false=nn.PReLU
        nll: true=log_softmax,false=sigmoid
        '''
        super(OutputTransition, self).__init__()
        self.conv1 = nn.Conv3d(inChans, 2, kernel_size=(1,5,5), padding=(0,2,2))
        self.bn1 = ContBatchNorm3d(2)
        self.conv2 = nn.Conv3d(2, 1, kernel_size=1)
        self.relu1 = ELUCons(elu, 1)
        if nll:
            self.softmax = F.log_softmax
        else:
            # self.softmax = F.softmax
            self.softmax = torch.sigmoid

    def forward(self, x):
        # convolve 32 down to 2 channels
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        out = self.softmax(out)

        return out


class VNet(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, elu=True, nll=False):
        '''
        elu: true=nn.ELU,false=nn.PReLU
        nll: true=log_softmax,false=sigmoid
        '''
        super(VNet, self).__init__()
        self.in_tr = InputTransition(16, elu)
        self.down_tr32 = DownTransition(16, 1,elu,(1,5,5),(0,2,2),(2,2,2),(2,2,2))
        self.down_tr64 = DownTransition(32, 2, elu,(1,5,5),(0,2,2),(2,2,2),(2,2,2))
        self.down_tr128 = DownTransition(64, 3, elu,(1,5,5),(0,2,2),(2,2,2),(2,2,2), dropout=True)
        self.down_tr256 = DownTransition(128, 2, elu,(5,5,5),(2,2,2),(2,2,2),(2,2,2), dropout=True)
        self.up_tr256 = UpTransition(256, 256, 2, elu,(1,5,5),(0,2,2),(2,2,2),(2,2,2),  dropout=True)
        self.up_tr128 = UpTransition(256, 128, 2, elu,(1,5,5),(0,2,2),(2,2,2),(2,2,2),  dropout=True)
        self.up_tr64 = UpTransition(128, 64, 1, elu,(1,5,5),(0,2,2),(2,2,2),(2,2,2))
        self.up_tr32 = UpTransition(64, 32, 1, elu,(1,5,5),(0,2,2),(2,2,2),(2,2,2))
        self.out_tr = OutputTransition(32, elu, nll)

    def forward(self, x):
        # print(x.shape)
        out16 = self.in_tr(x)
        out32 = self.down_tr32(out16)
        out64 = self.down_tr64(out32)
        out128 = self.down_tr128(out64)
        out256 = self.down_tr256(out128)
        out = self.up_tr256(out256, out128)
        out = self.up_tr128(out, out64)
        out = self.up_tr64(out, out32)
        out = self.up_tr32(out, out16)
        out = self.out_tr(out)

        return out
