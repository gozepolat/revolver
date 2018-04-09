# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from torch.nn import Module
from stacked.modules.scoped_nn import ScopedConv2d, \
    ScopedBatchNorm2d, ScopedLinear, ScopedModuleList
from stacked.meta.scoped import ScopedMeta
from six import add_metaclass


@add_metaclass(ScopedMeta)
class ResBlock(Module):
    r"""Residual prelu block

    Args:
        ni (int): Number of channels in the input
        no (int): Number of channels produced by the block
        kernel_size (int or tuple): Size of the convolving kernel. Default: 3
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        conv_module (torch.nn.Module): Module to use in forward. Default: ScopedConv2d
    """
    def __init__(self, scope, ni, no, kernel_size=3, stride=1, conv_module=ScopedConv2d):
        super(ResBlock, self).__init__()
        self.scope = scope
        self.conv0 = conv_module("%s/conv0" % scope, ni, no, kernel_size, stride, padding=1)
        self.bn0 = ScopedBatchNorm2d("%s/bn0" % scope, ni)
        self.conv1 = conv_module("%s/conv1" % scope, no, no, kernel_size, 1, 1)
        self.bn1 = ScopedBatchNorm2d("%s/bn1" % scope, no)
        # 1x1 conv to correct the number of channels for summation
        self.convdim = ScopedConv2d("%s/convdim" % scope,
                                    ni, no, 1, stride=stride) if ni != no else None

    def forward(self, x):
        o1 = F.relu(self.bn0(x), inplace=True)
        y = self.conv0(o1)
        o2 = F.relu(self.bn1(y), inplace=True)
        z = self.conv1(o2)
        if self.convdim is not None:
            return z + self.convdim(o1)
        else:
            return z + x


@add_metaclass(ScopedMeta)
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
    def __init__(self, scope, ni, no, kernel_size, stride, num_blocks, conv_module=ScopedConv2d):
        super(ResGroup, self).__init__()
        self.scope = scope
        self.block_list = ScopedModuleList("%s/block_list" % scope)
        for i in range(num_blocks):
            k = i
            self.block_list.append(
                ResBlock("%s/block%d" % (scope, k), ni, no, kernel_size, stride, conv_module))
            ni = no
            stride = 1

    def forward(self, x):
        for block in self.block_list:
            x = block(x)
        return x


@add_metaclass(ScopedMeta)
class ScopedResNet(Module):
    r"""WRN inspired implementation of ResNet

    Args:
        depth (int): Overall depth of the network
        width (int): scalar that will be multiplied with blueprint to get the width per group
        num_classes (int): Number of categories for supervised learning
        blueprint (iterable): the smallest possible widths per group
    """
    def __init__(self, scope, depth, width, num_classes, blueprint=(16, 32, 64), conv_module=ScopedConv2d):
        super(ScopedResNet, self).__init__()
        self.scope = scope
        self.blueprint = blueprint
        self.num_blocks = (depth - 4) // 6
        self.widths = [i * width for i in blueprint]
        self.conv0 = ScopedConv2d("%s/conv0" % scope, 3, blueprint[0], 3, 1, padding=1)
        self.group_list = ScopedModuleList("%s/group_list" % scope)
        self.bn = ScopedBatchNorm2d("%s/bn" % scope, self.widths[2])
        self.linear = ScopedLinear("%s/linear" % scope, self.widths[2], num_classes)

        ni = blueprint[0]
        stride = 1
        k = 0
        for width in self.widths:
            no = width
            self.group_list.append(ResGroup("%s/group%d" % (scope, k),
                                            ni, no, 3, stride, self.num_blocks, conv_module))
            ni = no
            stride = 2
            k += 1

    def forward(self, x):
        x = self.conv0(x)
        for i, _ in enumerate(self.widths):
            x = self.group_list[i](x)
        o = F.relu(self.bn(x))
        o = F.avg_pool2d(o, o.size()[2], 1, 0)
        o = o.view(o.size(0), -1)
        o = self.linear(o)
        return o
