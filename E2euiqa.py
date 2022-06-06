import torch
import torch.nn as nn
# import torch.nn.functional as F
from torch.nn import init
from Gdn import Gdn2d, Gdn1d
from Pool import GlobalAvgPool2d
from Spp import SpatialPyramidPooling2d

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight.data)
    elif classname.find('Gdn2d') != -1:
        init.eye_(m.gamma.data)
        init.constant_(m.beta.data, 1e-4)
    elif classname.find('Gdn1d') != -1:
        init.eye_(m.gamma.data)
        init.constant_(m.beta.data, 1e-4)


# def build_net(norm=Gdn, layer=5, width=24):
#     layers = [
#         nn.Conv2d(3, width, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
#         norm(width)
#     ]
#
#     for l in range(1, layer):
#         layers += [nn.Conv2d(width,  width, kernel_size=3, stride=1, padding=2**l,  dilation=2**l,  bias=True),
#                    norm(width)
#                    ]
#
#     layers += [
#         nn.Conv2d(width, width, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
#         norm(width),
#         nn.Conv2d(width,  2, kernel_size=1, stride=1, padding=0, dilation=1, bias=True)
#     ]
#
#     net = nn.Sequential(*layers)
#     net.apply(weights_init)
#
#     return net


def build_model(normc=Gdn2d, normf=Gdn1d, layer=3, width=48):
    layers = [
        nn.Conv2d(3, width, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
        normc(width),
        # nn.Conv2d(width, width, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
        # normc(width),
        # nn.AvgPool2d(kernel_size=2)
        nn.MaxPool2d(kernel_size=2)
    ]

    for l in range(1, layer):
        layers += [nn.Conv2d(width,  width, kernel_size=3, stride=1, padding=1,  dilation=1,  bias=True),
                   normc(width),
                   # nn.Conv2d(width, width, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                   # normc(width),
                   # nn.AvgPool2d(kernel_size=2)
                   nn.MaxPool2d(kernel_size=2)
                   ]

    layers += [nn.Conv2d(width,  width, kernel_size=3, stride=1, padding=1,  dilation=1,  bias=True),
               normc(width),
               # GlobalAvgPool2d()
               # SpatialPyramidPooling2d()
               SpatialPyramidPooling2d(pool_type='max_pool')
               ]
    layers += [nn.Linear(width*14, 128, bias=True),
               # normf(width),
               nn.ReLU(),
               # nn.Linear(128, 128, bias=True),
               # nn.ReLU(),
               nn.Linear(128, 2, bias=True)
               ]
    net = nn.Sequential(*layers)
    net.apply(weights_init)

    return net


class E2EUIQA(nn.Module):
    # end-to-end unsupervised image quality assessment model
    def __init__(self):
        super(E2EUIQA, self).__init__()
        self.cnn = build_model()

    def forward(self, x):
        r = self.cnn(x)
        mean = r[:, 0].unsqueeze(dim=-1)
        var = torch.exp(r[:, 1]).unsqueeze(dim=-1)

        output = []
        output.append(mean)

        return output, var

    def init_model(self, path):
        self.load_state_dict(torch.load(path)['state_dict'])

class LFC_features(nn.Module):
    def __init__(self, lfc):
        super(LFC_features, self).__init__()
        self.lfc = list(*lfc.children())[0:11]
        self.lfc = nn.Sequential(*self.lfc)
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
    def forward(self, x):
        x = self.lfc(x)
        features = self.avg_pool(x)
        features = features.squeeze(3).squeeze(2)
        return features




