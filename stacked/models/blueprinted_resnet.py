# -*- coding: utf-8 -*-
import torch.nn.functional as F
from torch.nn import Module
from stacked.modules.scoped_nn import ScopedConv2d, \
    ScopedBatchNorm2d, ScopedLinear, ModuleList, ScopedReLU
from stacked.meta.scoped import ScopedMeta
from stacked.meta.blueprint import Blueprint, make_module
from stacked.utils.transformer import all_to_none
from six import add_metaclass


@add_metaclass(ScopedMeta)
class ScopedResBlock(Module):
    r"""Pre-ResNet block

    Args:
        scope: Scope for the self (ScopedResBlock instance)
        blueprint: Description of inner scopes, member modules, and args
    """

    def __init__(self, scope, blueprint):
        super(ScopedResBlock, self).__init__()
        self.scope = scope

        # containers for the units
        self.act = ModuleList()
        self.conv = ModuleList()
        self.bn = ModuleList()

        self.depth = len(blueprint['children'])

        for unit in blueprint['children']:
            act, bn, conv = unit
            self.act.append(make_module(act))
            self.conv.append(make_module(conv))
            self.bn.append(make_module(bn))

        # 1x1 conv to correct the number of channels for summation
        convdim = blueprint['convdim']
        self.convdim = make_module(convdim)

    def forward(self, x):
        o1 = self.act[0](self.bn[0](x))

        o2 = o1
        for i in range(1, self.depth):
            y = self.conv[i-1](o2)
            o2 = self.act[i](self.bn[i](y))
        z = self.conv[-1](o2)

        if self.convdim is not None:
            return z + self.convdim(o1)
        else:
            return z + x

    @staticmethod
    def describe_default(prefix, parent, depth, conv_module, bn_module, act_module,
                         ni, no, kernel_size, stride, padding):
        """Create a default ResBlock blueprint

        Args:
            prefix (str): Prefix from which the member scopes will be created
            parent (Blueprint): None or the instance of the parent blueprint
            depth: Number of conv/act/bn units in the block
            conv_module (type): CNN module to use in forward. e.g. ScopedConv2d
            bn_module (type): Batch normalization module. e.g. ScopedBatchNorm2d
            act_module (type): Activation module e.g ScopedReLU
            ni (int): Number of channels in the input
            no (int): Number of channels produced by the block
            kernel_size (int or tuple): Size of the convolving kernel. Default: 3
            stride (int or tuple, optional): Stride for the first convolution
            padding (int or tuple, optional): Padding for the first convolution
        """
        def scope(suffix):
            return '%s/%s_%d' % (prefix, suffix, kernel_size)

        children = []
        default = Blueprint(prefix, parent, False, ScopedResBlock)
        convdim_type = conv_module if ni != no else all_to_none
        convdim = Blueprint(scope('convdim%d_%d' % (ni, no)), default, False, convdim_type,
                            args=[ni, no, 1, stride])
        default['convdim'] = convdim
        # default: after ni = no, everything is the same
        for i in range(depth):
            act = Blueprint(scope('act%d_%d' % (ni, no)), default, False,
                            act_module, args=[True])
            bn = Blueprint(scope('bn%d_%d' % (ni, no)), default, False,
                           bn_module, args=[ni])
            conv = Blueprint(scope('conv%d_%d' % (ni, no)), default, False, conv_module,
                             args=[ni, no, kernel_size, stride, padding])
            unit = (act, bn, conv)
            children.append(unit)
            padding = 1
            stride = 1
            ni = no
        default['children'] = children

        # the only arg for ScopedResBlock is the blueprint itself (would be already known)
        # default['args'] = [default] - unnecessary
        return default


@add_metaclass(ScopedMeta)
class ScopedResGroup(Module):
    r"""Group of residual blocks with the same number of output channels

        Args:
        scope: Scope for the self (ScopedResBlock instance)
        blueprint: Description of the scopes and member module types
        ni (int): Number of channels in the input
        no (int): Number of channels produced by the block
        kernel_size (int or tuple): Size of the convolving kernel.
        stride (int or tuple, optional): Stride of the convolution.
        num_blocks (int): Number of residual blocks per group
        conv_module (torch.nn.Module): Module to use in forward. Default: Conv2d
    """
    def __init__(self, scope, blueprint):
        super(ScopedResGroup, self).__init__()
        self.scope = scope
        self.block_container = ModuleList()
        for block in blueprint['children']:
            self.block_container.append(ScopedResBlock(block['name'], block))

    def forward(self, x):
        for block in self.block_container:
            x = block(x)
        return x

    @staticmethod
    def describe_default(prefix, parent, group_depth, block_depth, conv_module, bn_module,
                         act_module, ni, no, kernel_size, stride, padding):
        """Create a default ResGroup blueprint

        Args:
            prefix (str): Prefix from which the member scopes will be created
            parent (Blueprint): None or the instance of the parent blueprint
            group_depth: Number of blocks in the group
            block_depth: Number of [conv/act/bn] units in the block
            conv_module (type): CNN module to use in forward. e.g. ScopedConv2d
            bn_module (type): Batch normalization module. e.g. ScopedBatchNorm2d
            act_module (type): Activation module e.g ScopedReLU
            ni (int): Number of channels in the input
            no (int): Number of channels produced by the block
            kernel_size (int or tuple): Size of the convolving kernel. Default: 3
            stride (int or tuple, optional): Stride for the first convolution
            padding (int or tuple, optional): Padding for the first convolution
        """

        default = Blueprint(prefix, parent, False, ScopedResGroup)
        children = []
        for i in range(group_depth):
            block_name = '%s/block%d_%d' % (prefix, ni, no)
            block = ScopedResBlock.describe_default(block_name, default, block_depth,
                                                    conv_module, bn_module, act_module,
                                                    ni, no, kernel_size, stride, padding)
            children.append(block)
            padding = 1
            stride = 1
            ni = no

        default['children'] = children
        return default


@add_metaclass(ScopedMeta)
class ScopedResNet(Module):
    r"""WRN inspired implementation of ResNet

    Args:
        scope (string): Scope for the self (ScopedResBlock instance)
        blueprint: Description of the scopes and member module types
    """
    def __init__(self, scope, blueprint):
        super(ScopedResNet, self).__init__()
        self.scope = scope

        act = blueprint['act']
        self.act = make_module(act)
        conv0 = blueprint['conv0']
        self.conv0 = make_module(conv0)
        self.group_container = ModuleList()
        bn = blueprint['bn']
        self.bn = make_module(bn)
        linear = blueprint['linear']
        self.linear = make_module(linear)
        for group in blueprint['children']:
            self.group_container.append(ScopedResGroup(group['name'], group))

    def forward(self, x):
        x = self.conv0(x)
        for group in self.group_container:
            x = group(x)
        o = self.act(self.bn(x))
        o = F.avg_pool2d(o, o.size()[2], 1, 0)
        o = o.view(o.size(0), -1)
        o = self.linear(o)
        return o

    @staticmethod
    def get_num_blocks_per_group(depth, num_groups, block_depth):
        return (depth - 4) // (num_groups * block_depth)

    @staticmethod
    def describe_default(prefix='ResNet', parent=None, skeleton=(16, 32, 64),
                         num_classes=10, depth=28, width=1,
                         block_depth=2, conv_module=ScopedConv2d,
                         bn_module=ScopedBatchNorm2d, linear_module=ScopedLinear,
                         act_module=ScopedReLU, kernel_size=3, padding=1):
        """Create a default ResBlock blueprint

        Args:
            prefix (str): Prefix from which the member scopes will be created
            parent (Blueprint): None or the instance of the parent blueprint
            skeleton (iterable): Smallest possible widths per group
            num_classes (int): Number of categories for supervised learning
            depth (int): Overall depth of the network
            width (int): Scalar to get the scaled width per group
            block_depth: Number of [conv/act/bn] units in the block
            conv_module (type): CNN module to use in forward. e.g. ScopedConv2d
            bn_module (type): Batch normalization module. e.g. ScopedBatchNorm2d
            linear_module (type): Linear module for classification e.g. ScopedLinear
            act_module (type): Activation module e.g ScopedReLU
            kernel_size (int or tuple): Size of the convolving kernel.
            padding (int or tuple, optional): Padding for the first convolution
        """

        def scope(suffix):
            return '%s/%s' % (prefix, suffix)

        widths = [i * width for i in skeleton]
        stride = 1
        children = []

        num_groups = len(skeleton)
        group_depth = ScopedResNet.get_num_blocks_per_group(depth, num_groups, block_depth)
        ni = skeleton[0]
        no = widths[-1]

        default = Blueprint(prefix, parent, False, ScopedResNet)
        default['bn'] = Blueprint(scope('bn_%d' % no), default, False, bn_module, args=[no])
        default['act'] = Blueprint(scope('act%d' % no), default, False, act_module, args=[True])
        default['conv0'] = Blueprint(scope('conv03_%d' % ni), default, False,
                                     conv_module, args=[3, ni, 3, 1, 1])
        default['linear'] = Blueprint(scope('linear%d_%d' % (no, num_classes)), default, False,
                                      linear_module, args=[no, num_classes])
        for width in widths:
            no = width
            block = ScopedResGroup.describe_default(scope('group%d_%d' % (ni, no)), default,
                                                    group_depth, block_depth,
                                                    conv_module, bn_module, act_module,
                                                    ni, no, kernel_size, stride, padding)
            children.append(block)
            padding = 1
            stride = 2
            ni = no
        default['children'] = children
        # the only argument required (blueprint=default), and it will be already available
        # default['args'] = [default]
        return default








