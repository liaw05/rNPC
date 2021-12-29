import torch.nn as nn


class Resnet3dAniso(nn.Module):
    '''
    第一层卷积核kernel=(3,3,3), stride=(1,1,1).
    '''

    def __init__(self, blocks, n_blocks, stage_filters, strides, dilations=None, in_channels=1, start_filters=16):

        super(Resnet3dAniso, self).__init__()

        self.blocks = blocks
        self.n_blocks = n_blocks
        self.inplanes = start_filters
        self.filters = stage_filters
        self.strides = strides
        self.in_channels = in_channels
        if dilations is None:
            self.dilations = [1 for _ in range(len(stage_filters))]
        else:
            self.dilations = dilations

        # feature extractor
        self.C1 = nn.Conv3d(self.in_channels, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)

        self.conv_list = nn.ModuleList(
            [self._make_layer(self.blocks[i], self.filters[i], self.n_blocks[i], self.strides[i], self.dilations[i]) for i in range(len(self.n_blocks))]
        )

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        layers = []
        downsample = None
        if not isinstance(stride, (list, tuple)):
            stride = (stride, stride, stride)
        stride = tuple(stride)
        if stride != (1,1,1) or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                        nn.Conv3d(
                            self.inplanes,
                            planes * block.expansion,
                            kernel_size=1,
                            stride=stride,
                            bias=False),
                        # nn.BatchNorm3d(planes),
                        )
        layers.append(block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: input image of shape (b, c, z, y, x)
        :return: list of output feature maps per pyramid level, each with shape (b, c, z, y, x).
        """
        x = self.C1(x)

        feats = []
        for stage_conv in self.conv_list:
            x = stage_conv(x)
            feats.append(x)

        return feats[-5:]
