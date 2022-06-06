import torch.nn as nn
from torchvision import models
import torch
import torch.nn.functional as F
from SCNN import SCNN
from copy import deepcopy


class DBCNN(nn.Module):

    def __init__(self, config):
        """Declare all needed layers."""
        nn.Module.__init__(self)
        # Convolution and pooling layers of VGG-16.
        self.n_task = config.n_task
        self.backbone = models.resnet18(pretrained=True)
        self.config = config

        scnn = SCNN()
        scnn = torch.nn.DataParallel(scnn).cuda()
        scnn.load_state_dict(torch.load(config.scnn_root))
        self.sfeatures = scnn.module.features
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.ModuleList()
        # using deepcopy to insure each fc layer initialized from the same parameters
        # (for fair comparision with sequentail/individual training)
        if self.config.JL:
            fc = nn.Linear(512 * 128, 1, bias=True)
        else:
            fc = nn.Linear(512 * 128, 1, bias=False)
        # Initialize the fc layers.
        nn.init.kaiming_normal_(fc.weight.data)
        for i in range(0, self.n_task):
            self.fc.append(deepcopy(fc))

        #always freeze SCNN
        for param in self.sfeatures.parameters():
            param.requires_grad = False

        if config.fc:
            # Freeze all previous layers.
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        """Forward pass of the network.
        """
        N = x.size()[0]

        x1 = self.backbone.conv1(x)
        x1 = self.backbone.bn1(x1)
        x1 = self.backbone.relu(x1)
        x1 = self.backbone.maxpool(x1)

        x1 = self.backbone.layer1(x1)
        x1 = self.backbone.layer2(x1)
        x1 = self.backbone.layer3(x1)
        x1 = self.backbone.layer4(x1)

        H = x1.size()[2]
        W = x1.size()[3]
        assert x1.size()[1] == 512

        x2 = self.sfeatures(x)
        H2 = x2.size()[2]
        W2 = x2.size()[3]
        assert x2.size()[1] == 128

        sfeat = self.pooling(x2)
        sfeat = sfeat.squeeze(3).squeeze(2)
        sfeat = F.normalize(sfeat, p=2)

        if (H != H2) | (W != W2):
            x2 = F.upsample_bilinear(x2, (H, W))

        x1 = x1.view(N, 512, H * W)
        x2 = x2.view(N, 128, H * W)
        x = torch.bmm(x1, torch.transpose(x2, 1, 2)) / (H * W)  # Bilinear
        assert x.size() == (N, 512, 128)
        x = x.view(N, 512 * 128)
        #x = torch.sqrt(x + 1e-8)
        x = F.normalize(x)

        output = []

        if not self.config.JL:
            for idx, fc in enumerate(self.fc):
                if not self.config.train:
                    for W in fc.parameters():
                        fc.weight.data = F.normalize(W, p=2, dim=1)
                output.append(fc(x))
        else:
            for idx, fc in enumerate(self.fc):
                output.append(fc(x))


        return output, sfeat

class DBCNN2(nn.Module):

    def __init__(self, config):
        """Declare all needed layers."""
        nn.Module.__init__(self)
        # Convolution and pooling layers of VGG-16.
        self.n_task = config.n_task
        self.backbone = models.vgg16(pretrained=True)
        self.config = config

        scnn = SCNN()
        scnn = torch.nn.DataParallel(scnn).cuda()
        scnn.load_state_dict(torch.load(config.scnn_root))
        self.sfeatures = scnn.module.features
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.ModuleList()
        # using deepcopy to insure each fc layer initialized from the same parameters
        # (for fair comparision with sequentail/individual training)
        if self.config.JL:
            fc = nn.Linear(512 * 128, 1, bias=True)
        else:
            fc = nn.Linear(512 * 128, 1, bias=False)
        # Initialize the fc layers.
        nn.init.kaiming_normal_(fc.weight.data)
        for i in range(0, self.n_task):
            self.fc.append(deepcopy(fc))

        #always freeze SCNN

        if config.fc:
            # Freeze all previous layers.
            for param in self.backbone.parameters():
                param.requires_grad = False

            for param in self.sfeatures.parameters():
                param.requires_grad = False

    def forward(self, x):
        """Forward pass of the network.
        """
        N = x.size()[0]
        x1 = self.backbone.features(x)

        H = x1.size()[2]
        W = x1.size()[3]
        assert x1.size()[1] == 512

        x2 = self.sfeatures(x)
        H2 = x2.size()[2]
        W2 = x2.size()[3]
        assert x2.size()[1] == 128

        if (H != H2) | (W != W2):
            x2 = F.upsample_bilinear(x2, (H, W))

        x1 = x1.view(N, 512, H * W)
        x2 = x2.view(N, 128, H * W)
        x = torch.bmm(x1, torch.transpose(x2, 1, 2)) / (H * W)  # Bilinear
        assert x.size() == (N, 512, 128)
        x = x.view(N, 512 * 128)
        #x = torch.sqrt(x + 1e-8)
        x = F.normalize(x)

        output = []

        if not self.config.JL:
            for idx, fc in enumerate(self.fc):
                if not self.config.train:
                    for W in fc.parameters():
                        fc.weight.data = F.normalize(W, p=2, dim=1)
                output.append(fc(x))
        else:
            for idx, fc in enumerate(self.fc):
                output.append(fc(x))

        sfeat = 0
        return output, sfeat