import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd import Variable


class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        y = torch.mean(torch.mean(x, dim=-1), dim=-1)
        return y

