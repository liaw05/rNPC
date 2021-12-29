import torch.nn as nn
from torch.nn import functional as F



class SFPN(nn.Module):

    def __init__(self, out_features, conv_channels, pre_activate=True):

        super(SFPN, self).__init__()

        self.out_features = out_features
        self.filters = conv_channels
        # lateral modules
        self.lateral_modules = nn.ModuleList()

        # top-down
        if not pre_activate:
            self.top_conv = nn.Conv3d(self.filters[-1], self.out_features, 1, 1, 0, bias=False)
        else:
            self.top_conv = nn.Sequential(
                nn.BatchNorm3d(self.filters[-1]),
                nn.ReLU(inplace=True),
                nn.Conv3d(self.filters[-1], self.out_features, 1, 1, 0, bias=False),
            )
        self.num_stages = len(self.filters)
        for i in range(self.num_stages-1):
            if not pre_activate:
                self.lateral_modules.append(nn.Conv3d(self.filters[-(i+2)], self.out_features, 1, 1, 0, bias=False))
            else:
                self.lateral_modules.append(
                    nn.Sequential(
                        nn.BatchNorm3d(self.filters[-(i+2)]),
                        nn.ReLU(inplace=True),
                        nn.Conv3d(self.filters[-(i+2)], self.out_features, 1, 1, 0, bias=False),
                    )
                )
    
    def _upsample3d(self, x, size):
        size = size[-3:]
        return F.upsample(x, size=size, mode='trilinear')
    
    def forward(self, inputs):

        # top-down
        fpn_output = [self.top_conv(inputs[-1])]
        for i in range(self.num_stages-1):
            lat = self.lateral_modules[i](inputs[-(i+2)])
            up = self._upsample3d(fpn_output[-1], lat.size())
            fpn_output.append(lat+up)
        
        fpn_output = fpn_output[::-1]

        return fpn_output