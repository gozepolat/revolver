# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from torch.nn import Module
from stacked.modules.scoped_nn import ScopedConv2d, \
    ScopedBatchNorm2d, ScopedLinear, ScopedModuleList
from stacked.meta.scoped import ScopedMeta
from six import add_metaclass


@add_metaclass(ScopedMeta)
class ScopedResBlock(Module):
    r"""Residual prelu block

    Args:
        scope: Scope for the self (ScopedResBlock instance)
        ni (int): Number of channels in the input
        no (int): Number of channels produced by the block
        kernel_size (int or tuple): Size of the convolving kernel. Default: 3
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        conv_module (torch.nn.Module): Module to use in forward. Default: ScopedConv2d
    """
    block_depth = 2

    def __init__(self, scope, meta, ni, no, kernel_size=3, stride=1, conv_module=ScopedConv2d):
        super(ScopedResBlock, self).__init__()
        self.scope = scope
        self.conv0 = conv_module(meta['conv0'].scope, ni, no, kernel_size, stride, padding=1)
        self.bn0 = ScopedBatchNorm2d(meta['bn0'].scope, ni)
        self.conv1 = conv_module(meta['conv1'].scope, no, no, kernel_size, 1, 1)
        self.bn1 = ScopedBatchNorm2d(meta['bn1'].scope, no)
        # 1x1 conv to correct the number of channels for summation
        self.convdim = ScopedConv2d(meta['convdim'].scope,
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
class ScopedResGroup(Module):
    r"""Group of residual blocks with the same number of output channels

        Args:
        scope: Scope for the self (ScopedResBlock instance)
        ni (int): Number of channels in the input
        no (int): Number of channels produced by the block
        kernel_size (int or tuple): Size of the convolving kernel.
        stride (int or tuple, optional): Stride of the convolution.
        num_blocks (int): Number of residual blocks per group
        conv_module (torch.nn.Module): Module to use in forward. Default: Conv2d
    """
    def __init__(self, scope, meta, ni, no, kernel_size, stride, num_blocks, conv_module=ScopedConv2d):
        super(ScopedResGroup, self).__init__()
        self.scope = scope
        self.block_list = ScopedModuleList(meta['block_list'].scope)
        for i in range(num_blocks):
            _meta = meta['block_elements'][i]
            self.block_list.append(
                ScopedResBlock(_meta.scope, _meta.meta, ni, no, kernel_size, stride, conv_module))
            ni = no
            stride = 1

    def forward(self, x):
        for block in self.block_list:
            x = block(x)
        return x


class ResNet(Module):
    r"""WRN inspired implementation of ResNet

    Args:
        scope: Scope for the self (ScopedResBlock instance)
        depth (int): Overall depth of the network
        width (int): scalar that will be multiplied with blueprint to get the width per group
        num_classes (int): Number of categories for supervised learning
        blueprint (iterable): the smallest possible widths per group
    """
    def __init__(self, scope, meta, depth, width, num_classes, skeleton=(16, 32, 64), conv_module=ScopedConv2d):
        super(ResNet, self).__init__()
        self.scope = scope
        self.blueprint = skeleton
        self.num_groups = len(skeleton)
        self.num_blocks = self.get_num_blocks_per_group(depth, self.num_groups)
        self.widths = [i * width for i in skeleton]

        self.conv0 = ScopedConv2d(meta['conv0'].scope, 3, skeleton[0], 3, 1, padding=1)
        self.group_list = ScopedModuleList(meta['group_list'].scope)
        self.bn = ScopedBatchNorm2d(meta['bn'].scope, self.widths[2])
        self.linear = ScopedLinear(meta['linear'].scope, self.widths[2], num_classes)

        ni = skeleton[0]
        stride = 1
        k = 0
        for width in self.widths:
            no = width
            _meta = meta['group_elements'][k]
            self.group_list.append(ScopedResGroup(_meta.scope, _meta.meta,
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
        return

    @staticmethod
    def get_num_blocks_per_group(depth, num_groups):
        return (depth - 4) // (num_groups * ScopedResBlock.block_depth)


class ResNetGraphMeta(object):
    pass
