# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from torch.nn import Module, Conv2d, BatchNorm2d


class ResBlock(Module):
    r"""Residual prelu block

    Args:
        ni (int): Number of channels in the input
        no (int): Number of channels produced by the block
        kernel_size (int or tuple): Size of the convolving kernel. Default: 3
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        conv_module (torch.nn.Module): Module to use in forward. Default: Conv2d
    """
    def __init__(self, ni, no, kernel_size=3, stride=1, conv_module=Conv2d):
        super(ResBlock, self).__init__()
        self.conv0 = conv_module(ni, no, kernel_size, stride, padding=1)
        self.bn0 = BatchNorm2d(ni)
        self.conv1 = conv_module(no, no, kernel_size, 1, 1)
        self.bn1 = BatchNorm2d(no)
        # 1x1 conv to correct the number of channels for summation
        self.convdim = Conv2d(ni, no, 1, stride=stride) if ni != no else None

    def forward(self, x):
        o1 = F.relu(self.bn0(x), inplace=True)
        y = self.conv0(o1)
        o2 = F.relu(self.bn1(y), inplace=True)
        z = self.conv1(o2)
        if self.convdim is not None:
            return z + self.convdim(o1)
        else:
            return z + x


class ResGroup(Module):
    r"""Group of residual blocks with the same number of output channels

        Args:
        ni (int): Number of channels in the input
        no (int): Number of channels produced by the block
        kernel_size (int or tuple): Size of the convolving kernel.
        stride (int or tuple, optional): Stride of the convolution.
        num_blocks (int): Number of residual blocks per group
        conv_module (torch.nn.Module): Module to use in forward. Default: Conv2d
    """
    def __init__(self, ni, no, kernel_size, stride, num_blocks, conv_module=Conv2d):
        super(ResGroup, self).__init__()
        self.block_list = torch.nn.ModuleList()
        for i in range(num_blocks):
            self.block_list.append(ResBlock(ni, no, kernel_size, stride, conv_module))
            ni = no
            stride = 1

    def forward(self, x):
        for block in self.block_list:
            x = block(x)
        return x


class ResNet(Module):
    r"""WRN inspired implementation of ResNet

    Args:
        depth (int): Overall depth of the network
        width (int): scalar that will be multiplied with blueprint to get the width per group
        num_classes (int): Number of categories for supervised learning
        blueprint (iterable): the smallest possible widths per group
    """
    def __init__(self, depth, width, num_classes, skeleton=(16, 32, 64), conv_module=Conv2d):
        super(ResNet, self).__init__()
        self.skeleton = skeleton
        self.num_blocks = (depth - 4) // 6
        self.widths = [i * width for i in skeleton]
        self.conv0 = Conv2d(3, skeleton[0], 3, 1, padding=1)
        self.group_list = torch.nn.ModuleList()
        self.bn = BatchNorm2d(self.widths[2])
        self.linear = torch.nn.Linear(self.widths[2], num_classes)

        ni = skeleton[0]
        stride = 1
        for width in self.widths:
            no = width
            self.group_list.append(ResGroup(ni, no, 3, stride, self.num_blocks, conv_module))
            ni = no
            stride = 2

    def forward(self, x):
        x = self.conv0(x)
        for i, _ in enumerate(self.widths):
            x = self.group_list[i](x)
        o = F.relu(self.bn(x), inplace=True)
        o = F.avg_pool2d(o, o.size()[2], 1, 0)
        o = o.view(o.size(0), -1)
        o = self.linear(o)
        return o
