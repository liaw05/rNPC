import torch.nn as nn


class HeadFeatBnReLu(nn.Module):

    def __init__(self, in_channels, num_layers, conv_op=nn.Conv3d):
        super(HeadFeatBnReLu, self).__init__()
        self.num_layers = num_layers
        self.conv_list = nn.ModuleList(
            [nn.Sequential(nn.BatchNorm3d(in_channels), nn.ReLU(inplace=True),
                conv_op(in_channels, in_channels, 3, 1, padding=1, bias=False)) for i in range(num_layers)]
        )
        self.bn_relu = nn.Sequential(nn.BatchNorm3d(in_channels),
                                    nn.ReLU(inplace=True),
                                    )

    def forward(self, inputs):
        feats = []
        for feat in inputs:
            for bn_relu_conv in self.conv_list:
                feat = bn_relu_conv(feat)

            feats.append(self.bn_relu(feat))

        return feats
