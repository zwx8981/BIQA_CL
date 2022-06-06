import torch.nn as nn
from torchvision import models
from copy import deepcopy
import torch
import torch.nn.functional as F
import os
import math

class View(nn.Module):
    """Changes view using a nn.Module."""

    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(*self.shape)


def weight_init(param):
    for m in param.modules():
        if isinstance(m, nn.Conv2d):
             nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
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

    def forward(self, X):
        feat = self.features(X)
        X = self.pooling(feat)
        X = X.squeeze(3).squeeze(2)
        X = F.normalize(X, p=2)
        return X, feat

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


class BaseCNN_vanilla(nn.Module):
    def __init__(self, config):
        """Declare all needed layers."""
        nn.Module.__init__(self)

        self.config = config
        self.n_task = config.n_task
        if self.config.backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=True)
        elif self.config.backbone == 'resnet34':
            self.backbone = models.resnet34(pretrained=True)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        outdim = 1
        self.fc = nn.ModuleList()

        fc = nn.Linear(512, outdim, bias=False)
        for i in range(0, self.n_task):
            self.fc.append(deepcopy(fc))

        if self.config.fc:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, im):
        """Forward pass of the network.
        """
        features = []

        x = self.backbone.conv1(im)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.backbone.avgpool(x)
        x = x.squeeze(3).squeeze(2)
        x = F.normalize(x, p=2)

        output = []
        for idx, fc in enumerate(self.fc):
            for W in fc.parameters():
                #W = F.normalize(W, p=2, dim=1)
                if not self.config.train:
                    fc.weight.data = F.normalize(W, p=2, dim=1)
                    #fc.weight.data = fc.weight.data
            output.append(fc(x))

        return output, x


class MetaIQA(nn.Module):
    def __init__(self, config):
        """Declare all needed layers."""
        nn.Module.__init__(self)

        self.config = config
        self.n_task = config.n_task
        if self.config.backbone == 'resnet18':
            self.resnet_layer = models.resnet18(pretrained=True)
        elif self.config.backbone == 'resnet34':
            self.resnet_layer = models.resnet34(pretrained=True)

        self.net = BaselineModel1(1, 0.5, 1000)
        state_dict = torch.load(r'./metaiqa.pth', map_location='cpu')
        self.load_state_dict(state_dict, strict=True)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.ModuleList()

    def forward(self, im):
        """Forward pass of the network.
        """
        x = self.resnet_layer(im)
        output = []
        output.append(self.net(x))

        return output, x

class BaselineModel1(nn.Module):
    def __init__(self, num_classes, keep_probability, inputsize):

        super(BaselineModel1, self).__init__()
        self.fc1 = nn.Linear(inputsize, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.drop_prob = (1 - keep_probability)
        self.relu1 = nn.PReLU()
        self.drop1 = nn.Dropout(self.drop_prob)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.relu2 = nn.PReLU()
        self.drop2 = nn.Dropout(p=self.drop_prob)
        self.fc3 = nn.Linear(512, num_classes)
        self.sig = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Weight initialization reference: https://arxiv.org/abs/1502.01852
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            # elif isinstance(m, nn.Linear):
            #     m.weight.data.normal_(0, 0.02)
            #     m.bias.data.zero_()

    def forward(self, x):
        """
        Feed-measure pass.
        :param x: Input tensor
        : return: Output tensor
        """
        out = self.fc1(x)

        out = self.bn1(out)
        out = self.relu1(out)
        out = self.drop1(out)
        out = self.fc2(out)

        out = self.bn2(out)
        out = self.relu2(out)
        out = self.drop2(out)
        out = self.fc3(out)
        #out = self.sig(out)
        # out_a = torch.cat((out_a, out_p), 1)

        # out_a = self.sig(out)
        return out