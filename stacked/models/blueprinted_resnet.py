# -*- coding: utf-8 -*-
import torch.nn.functional as F
from torch.nn import Module
from stacked.modules.scoped_nn import ScopedConv2d, \
    ScopedBatchNorm2d, ScopedLinear, ScopedModuleList, ScopedReLU
from stacked.meta.scoped import ScopedMeta
from stacked.meta.blueprint import Blueprint
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
        self.act = ScopedModuleList(blueprint['act_container']['name'])
        self.conv = ScopedModuleList(blueprint['conv_container']['name'])
        self.bn = ScopedModuleList(blueprint['bn_container']['name'])

        self.depth = len(blueprint['children'])

        for unit in blueprint['children']:
            act, bn, conv = unit
            self.act.append(act['type'](act['name'], *act['args']))
            self.conv.append(conv['type'](conv['name'], *conv['args']))
            self.bn.append(bn['type'](bn['name'], *bn['args']))

        # 1x1 conv to correct the number of channels for summation
        convdim = blueprint['convdim']
        self.convdim = convdim['type'](convdim['name'], *convdim['args'])

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
    def describe_default(prefix, depth, conv_module, bn_module, act_module,
                         ni, no, kernel_size, stride, padding):
        """Create a default ResBlock blueprint

        Args:
            prefix (str): Prefix from which the member scopes will be created
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

        convdim_type = conv_module if ni != no else all_to_none
        convdim = Blueprint(scope('convdim%d_%d' % (ni, no)), False, convdim_type,
                            args=[ni, no, 1, stride])
        children = []

        # default: after ni = no, everything is the same
        for i in range(depth):
            act = Blueprint(scope('act%d_%d' % (ni, no)), False,
                            act_module, args=[True])
            bn = Blueprint(scope('bn%d_%d' % (ni, no)), False,
                           bn_module, args=[ni])
            conv = Blueprint(scope('conv%d_%d' % (ni, no)), False, conv_module,
                             args=[ni, no, kernel_size, stride, padding])
            unit = (act, bn, conv)
            children.append(unit)
            padding = 1
            stride = 1
            ni = no

        description = {'children': children, 'convdim': convdim,
                       'act_container': Blueprint(scope('act_container%d_%d' % (ni, no))),
                       'conv_container': Blueprint(scope('conv_container%d_%d' % (ni, no))),
                       'bn_container': Blueprint(scope('bn_container%d_%d' % (ni, no)))}

        default = Blueprint(prefix, False, ScopedResBlock, description=description)
        # the only argument required for the ScopedResBlock is the blueprint itself
        default['args'] = [default]
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
        self.block_container = ScopedModuleList(blueprint['block_container']['name'])
        for block in blueprint['children']:
            self.block_container.append(ScopedResBlock(block['name'], block))

    def forward(self, x):
        for block in self.block_container:
            x = block(x)
        return x

    @staticmethod
    def describe_default(prefix, group_depth, block_depth, conv_module, bn_module,
                         act_module, ni, no, kernel_size, stride, padding):
        """Create a default ResGroup blueprint

        Args:
            prefix (str): Prefix from which the member scopes will be created
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

        def scope(suffix):
            return '%s/%s_%d' % (prefix, suffix, kernel_size)

        children = []
        for i in range(group_depth):
            block_name = scope('block%d_%d' % (ni, no))
            block = ScopedResBlock.describe_default(block_name, block_depth,
                                                    conv_module, bn_module, act_module,
                                                    ni, no, kernel_size, stride, padding)
            children.append(block)
            padding = 1
            stride = 1
            ni = no

        container_name = scope('block_container%d_%d' % (ni, no))
        description = {'children': children,
                       'block_container': Blueprint(container_name)}

        default = Blueprint(prefix, False, ScopedResGroup, description=description)
        # the only argument required (blueprint=default)
        default['args'] = [default]
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
        self.act = act['type'](act['name'], *act['args'])
        conv0 = blueprint['conv0']
        self.conv0 = conv0['type'](conv0['name'], *conv0['args'])
        self.group_container = ScopedModuleList(blueprint['group_container']['name'])
        bn = blueprint['bn']
        self.bn = bn['type'](bn['name'], *bn['args'])
        linear = blueprint['linear']
        self.linear = linear['type'](linear['name'], *linear['args'])
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
    def describe_default(prefix='ResNet', skeleton=(16, 32, 64),
                         num_classes=10, depth=28, width=1,
                         block_depth=2, conv_module=ScopedConv2d,
                         bn_module=ScopedBatchNorm2d, linear_module=ScopedLinear,
                         act_module=ScopedReLU, kernel_size=3, padding=1):
        """Create a default ResBlock blueprint

        Args:
            prefix (str): Prefix from which the member scopes will be created
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
            return '%s/%s_%d' % (prefix, suffix, kernel_size)

        widths = [i * width for i in skeleton]
        ni = skeleton[0]
        stride = 1
        children = []

        num_groups = len(skeleton)
        group_depth = ScopedResNet.get_num_blocks_per_group(depth, num_groups,
                                                            block_depth)
        for width in widths:
            no = width
            block = ScopedResGroup.describe_default(scope('group%d_%d' % (ni, no)),
                                                    group_depth, block_depth,
                                                    conv_module, bn_module, act_module,
                                                    ni, no, kernel_size, stride, padding)
            children.append(block)
            padding = 1
            stride = 2
            ni = no

        container_name = scope('group_container%d_%d' % (ni, no))
        description = {'children': children,
                       'bn': Blueprint(scope('bn%d_%d' % (ni, no)), False, bn_module,
                                       args=[widths[2]]),
                       'act': Blueprint(scope('act%d_%d' % (ni, no)), False, act_module,
                                        args=[True]),
                       'conv0': Blueprint(scope('conv0%d_%d' % (ni, no)), False,
                                          conv_module, args=[3, skeleton[0], 3, 1, 1]),
                       'linear': Blueprint(scope('linear%d_%d' % (ni, no)), False,
                                           linear_module, args=[widths[2], num_classes]),
                       'group_container': Blueprint(container_name)}

        default = Blueprint(prefix, False, ScopedResNet, description=description)
        # the only argument required (blueprint=default)
        default['args'] = [default]
        return default








