import torch.nn as nn
import torch
import torch.nn.functional as F
import os
# from torchvision import models
# import torch

def weight_init(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


class SCNN(nn.Module):

    def __init__(self):
        """Declare all needed layers."""
        super(SCNN, self).__init__()

        # Linear classifier.

        self.num_class = 39
        self.features = nn.Sequential(nn.Conv2d(3,48,3,1,1),nn.BatchNorm2d(48),nn.ReLU(inplace=True),
                                      nn.Conv2d(48,48,3,2,1),nn.BatchNorm2d(48),nn.ReLU(inplace=True),
                                      nn.Conv2d(48,64,3,1,1),nn.BatchNorm2d(64),nn.ReLU(inplace=True),
                                      nn.Conv2d(64,64,3,2,1),nn.BatchNorm2d(64),nn.ReLU(inplace=True),
                                      nn.Conv2d(64,64,3,1,1),nn.BatchNorm2d(64),nn.ReLU(inplace=True),
                                      nn.Conv2d(64,64,3,2,1),nn.BatchNorm2d(64),nn.ReLU(inplace=True),
                                      nn.Conv2d(64,128,3,1,1),nn.BatchNorm2d(128),nn.ReLU(inplace=True),
                                      nn.Conv2d(128,128,3,1,1),nn.BatchNorm2d(128),nn.ReLU(inplace=True),
                                      nn.Conv2d(128,128,3,2,1),nn.BatchNorm2d(128),nn.ReLU(inplace=True))
        weight_init(self.features)
        self.pooling = nn.AvgPool2d(14,1)
        self.projection = nn.Sequential(nn.Conv2d(128,256,1,1,0), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
                                        nn.Conv2d(256,256,1,1,0), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        weight_init(self.projection)
        self.classifier = nn.Linear(256,self.num_class)
        weight_init(self.classifier)

        self.pooling = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        feat = self.features(x)
        x = self.pooling(feat)
        x = x.squeeze(3).squeeze(2)
        x = F.normalize(x, p=2)
        return x, feat

    def save_bn(self, name='saved_bn.pt'):
        bns = nn.ModuleList()
        for module in self.backbone.modules():
            if isinstance(module, nn.BatchNorm2d):
                bns.append(module)

        bn_name = os.path.join(self.config.ckpt_path, name)
        torch.save(bns, bn_name)


    def load_bn(self, bn_path):
        bns = torch.load(bn_path)
        idx = 0
        for module in self.backbone.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.load_state_dict(bns[idx].state_dict())
                idx = idx + 1
