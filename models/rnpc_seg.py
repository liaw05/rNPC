from models.backbone.vnet import VNet
import torch.nn as nn

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.net = VNet(elu=False, nll=False)

    def forward(self, inputs):
        out = self.net(inputs)
        return out