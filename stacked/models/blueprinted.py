# -*- coding: utf-8 -*-
import torch.nn.functional as F
from torch.nn import Module
from stacked.modules.scoped_nn import ScopedConv2d, \
    ScopedBatchNorm2d, ScopedLinear, ScopedReLU
from stacked.meta.scope import ScopedMeta
from stacked.meta.sequential import Sequential
from stacked.meta.blueprint import Blueprint, make_module
from stacked.utils.transformer import all_to_none
from stacked.modules.conv import get_conv_out_shape
from six import add_metaclass


@add_metaclass(ScopedMeta)
class ScopedConvUnit(Module):
    """BN-act-conv unit

    Args:
        scope: Scope for the self (ScopedConvUnit instance)
        blueprint: Description of bn, act, and conv
        Ignores the rest of the args
    """
    def __init__(self, scope, blueprint, *_, **__):
        super(ScopedConvUnit, self).__init__()
        self.scope = scope

        self.bn = make_module(blueprint['bn'])
        self.act = make_module(blueprint['act'])
        self.conv = make_module(blueprint['conv'])

    def forward(self, x):
        x = self.act(self.bn(x))
        x = self.conv(x)
        return x

    @staticmethod
    def set_unit_description(default, prefix, input_shape, ni, no, kernel_size,
                             stride, padding, conv_module, act_module, bn_module):
        """Set descriptions for act, bn, and conv"""
        # describe act and bn
        default['act'] = Blueprint('%s/act' % prefix, '%d_%d' % (ni, no), default,
                                   False, act_module, kwargs={'inplace': True})
        default['bn'] = Blueprint('%s/bn' % prefix, '%d_%d' % (ni, no), default,
                                  False, bn_module, kwargs={'num_features': ni})
        # describe conv
        kwargs = {'in_channels': ni, 'out_channels': no,
                  'kernel_size': kernel_size, 'stride': stride, 'padding': padding}
        conv = Blueprint('%s/conv' % prefix, '%d_%d_%d' % (ni, no, kernel_size),
                         default, False, conv_module, kwargs=kwargs)
        conv['input_shape'] = input_shape
        conv['output_shape'] = get_conv_out_shape(input_shape, no,
                                                  kernel_size, stride, padding)
        default['conv'] = conv

    @staticmethod
    def describe_default(prefix, suffix, parent, input_shape, ni, no, kernel_size,
                         stride, padding, act_module, bn_module, conv_module):
        """Create a default ScopedConvUnit blueprint"""
        default = Blueprint(prefix, suffix, parent, False, ScopedConvUnit)
        default['input_shape'] = input_shape

        ScopedConvUnit.set_unit_description(default, prefix, input_shape, ni, no,
                                            kernel_size, stride, padding,
                                            conv_module, act_module, bn_module)

        default['output_shape'] = default['conv']['output_shape']
        default['kwargs'] = {'blueprint': default}
        return default


@add_metaclass(ScopedMeta)
class ScopedResBlock(Sequential):
    """Pre-ResNet block

    Args:
        scope: Scope for the self (ScopedResBlock instance)
        blueprint: Description of inner scopes, member modules, and args
    """

    def __init__(self, scope, blueprint, *_, **__):
        super(ScopedResBlock, self).__init__(blueprint)
        self.scope = scope
        self.depth = len(blueprint['children'])

        self.bn = make_module(blueprint['bn'])
        self.act = make_module(blueprint['act'])
        self.conv = make_module(blueprint['conv'])

        # 1x1 conv to correct the number of channels for summation
        self.convdim = make_module(blueprint['convdim'])

    def forward(self, x):
        o = self.act(self.bn(x))
        z = self.conv(o)

        for unit in self.container:
            z = unit(z)

        if self.convdim is not None:
            return z + self.convdim(o)
        else:
            return z + x

    @staticmethod
    def __set_default_items(prefix, default, input_shape, ni, no, kernel_size,
                            stride, padding, conv_module, act_module, bn_module):
        default['input_shape'] = input_shape

        # describe bn, act, conv
        ScopedConvUnit.set_unit_description(default, prefix, input_shape, ni, no,
                                            kernel_size, stride, padding,
                                            conv_module, act_module, bn_module)
        # describe convdim
        convdim_type = conv_module if ni != no else all_to_none
        kwargs = {'in_channels': ni, 'out_channels': no,
                  'kernel_size': 1, 'stride': stride, 'padding': 0}
        convdim = Blueprint('%s/convdim' % prefix, '%d_%d' % (ni, no), default,
                            False, convdim_type, kwargs=kwargs)
        convdim['input_shape'] = input_shape
        convdim['output_shape'] = get_conv_out_shape(input_shape, no, 1, stride, 0)
        default['convdim'] = convdim
        return default['conv']['output_shape']

    @staticmethod
    def __set_default_children(prefix, default, shape, ni, no, kernel_size, stride,
                               padding, conv_module, act_module, bn_module, depth):
        children = []
        for i in range(depth):
            unit_prefix = '%s/unit' % prefix
            suffix = '%d_%d' % (ni, no)
            unit = ScopedConvUnit.describe_default(unit_prefix, suffix, default, shape,
                                                   ni, no, kernel_size, stride, padding,
                                                   act_module, bn_module, conv_module)
            shape = unit['output_shape']
            children.append(unit)

        default['children'] = children
        return shape

    @staticmethod
    def describe_default(prefix, suffix, parent, depth, conv_module, bn_module,
                         act_module, ni, no, kernel_size, stride, padding, input_shape):
        """Create a default ScopedResBlock blueprint

        Args:
            prefix (str): Prefix from which the member scopes will be created
            suffix (str): Suffix to append the name of the scoped object
            parent (Blueprint): None or the instance of the parent blueprint
            depth: Number of (bn, act, conv) units in the block
            conv_module (type): CNN module to use in forward. e.g. ScopedConv2d
            bn_module (type): Batch normalization module. e.g. ScopedBatchNorm2d
            act_module (type): Activation module e.g ScopedReLU
            ni (int): Number of channels in the input
            no (int): Number of channels produced by the block
            kernel_size (int or tuple): Size of the convolving kernel. Default: 3
            stride (int or tuple, optional): Stride for the first convolution
            padding (int or tuple, optional): Padding for the first convolution
            input_shape (tuple): (N, C_{in}, H_{in}, W_{in})
        """
        default = Blueprint(prefix, suffix, parent, False, ScopedResBlock)

        input_shape = ScopedResBlock.__set_default_items(prefix, default, input_shape,
                                                         ni, no, kernel_size, stride,
                                                         padding, conv_module, act_module,
                                                         bn_module)
        # in_channels = no, and stride = 1 for children
        input_shape = ScopedResBlock.__set_default_children(prefix, default, input_shape,
                                                            no, no, kernel_size, 1,
                                                            padding, conv_module, act_module,
                                                            bn_module, depth)
        default['output_shape'] = input_shape
        default['kwargs'] = {'blueprint': default}
        return default


@add_metaclass(ScopedMeta)
class ScopedResGroup(Sequential):
    """Group of residual blocks with the same number of output channels"""
    def __init__(self, scope, blueprint, *_, **__):
        self.scope = scope
        super(ScopedResGroup, self).__init__(blueprint)

    @staticmethod
    def describe_default(prefix, suffix, parent, group_depth, block_depth, conv_module,
                         bn_module, act_module, ni, no, kernel_size, stride, padding,
                         input_shape):
        """Create a default ResGroup blueprint

        Args:
            prefix (str): Prefix from which the member scopes will be created
            suffix (str): Suffix to append the name of the scoped object
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
            input_shape (tuple): (N, C_{in}, H_{in}, W_{in})
        """
        default = Blueprint(prefix, suffix, parent, False, ScopedResGroup)
        children = []
        default['input_shape'] = input_shape
        for i in range(group_depth):
            block_prefix = '%s/block' % prefix
            suffix = '%d_%d' % (ni, no)
            block = ScopedResBlock.describe_default(block_prefix, suffix, default,
                                                    block_depth, conv_module, bn_module,
                                                    act_module, ni, no, kernel_size,
                                                    stride, padding, input_shape)
            input_shape = block['output_shape']
            children.append(block)
            padding = 1
            stride = 1
            ni = no
        default['children'] = children

        default['output_shape'] = input_shape
        default['kwargs'] = {'blueprint': default}
        return default


@add_metaclass(ScopedMeta)
class ScopedResNet(Sequential):
    """WRN inspired implementation of ResNet

    Args:
        scope (string): Scope for the self (ScopedResBlock instance)
        blueprint: Description of the scopes and member module types
    """
    def __init__(self, scope, blueprint):
        super(ScopedResNet, self).__init__(blueprint)
        self.scope = scope

        self.act = make_module(blueprint['act'])
        self.conv = make_module(blueprint['conv'])
        self.bn = make_module(blueprint['bn'])
        self.linear = make_module(blueprint['linear'])

    def forward(self, x):
        x = self.conv(x)
        for group in self.container:
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
    def __set_default_items(prefix, default, shape, ni, no, kernel_size, num_classes,
                            bn_module, act_module, conv_module, linear_module):
        """Set blueprint items that are not Sequential type"""

        default['input_shape'] = shape
        default['bn'] = Blueprint('%s/bn' % prefix, '%d' % no, default, False,
                                  bn_module, kwargs={'num_features': no})
        default['act'] = Blueprint('%s/act' % prefix, '%d' % no, default, False,
                                   act_module, kwargs={'inplace': True})
        # describe conv
        kwargs = {'in_channels': shape[1], 'out_channels': ni,
                  'kernel_size': kernel_size, 'stride': 1, 'padding': 1}
        suffix = '%d_%d_%d' % (shape[1], ni, kernel_size)
        default['conv'] = Blueprint('%s/conv' % prefix, suffix,
                                    default, False, conv_module, kwargs=kwargs)
        default['conv']['input_shape'] = shape
        shape = get_conv_out_shape(shape, ni, kernel_size, 1, 1)
        default['conv']['output_shape'] = shape

        # describe linear, (shapes will be set after children)
        kwargs = {'in_features': no, 'out_features': num_classes}
        default['linear'] = Blueprint('%s/linear' % prefix,
                                      '%d_%d' % (no, num_classes),
                                      default, False, linear_module,
                                      kwargs=kwargs)
        return shape

    @staticmethod
    def __set_default_children(prefix, default, ni, widths, group_depth,
                               block_depth, conv_module, bn_module, act_module,
                               kernel_size, stride, padding, shape):
        """Sequentially set children blueprints"""
        children = []
        for width in widths:
            no = width
            block = ScopedResGroup.describe_default('%s/group' % prefix, '%d_%d' % (ni, no),
                                                    default, group_depth, block_depth,
                                                    conv_module, bn_module, act_module,
                                                    ni, no, kernel_size, stride, padding,
                                                    shape)
            shape = block['output_shape']
            children.append(block)
            padding = 1
            stride = 2
            ni = no
        default['children'] = children
        return shape

    @staticmethod
    def __get_default(prefix, suffix, parent, shape, ni, no, kernel_size,
                      num_classes, bn_module, act_module, conv_module, linear_module,
                      widths, group_depth, block_depth, stride, padding):
        """Set the items and the children of the default blueprint object"""
        default = Blueprint(prefix, suffix, parent, False, ScopedResNet)
        shape = ScopedResNet.__set_default_items(prefix, default, shape, ni, no,
                                                 kernel_size, num_classes, bn_module,
                                                 act_module, conv_module, linear_module)

        shape = ScopedResNet.__set_default_children(prefix, default, ni, widths,
                                                    group_depth, block_depth, conv_module,
                                                    bn_module, act_module, kernel_size,
                                                    stride, padding, shape)

        default['linear']['input_shape'] = (shape[0], shape[1])
        default['linear']['output_shape'] = (shape[0], num_classes)
        default['output_shape'] = (shape[0], num_classes)
        default['kwargs'] = {'blueprint': default}
        return default

    @staticmethod
    def describe_default(prefix='ResNet', suffix='', parent=None, skeleton=(16, 32, 64),
                         num_classes=10, depth=28, width=1,
                         block_depth=2, conv_module=ScopedConv2d,
                         bn_module=ScopedBatchNorm2d, linear_module=ScopedLinear,
                         act_module=ScopedReLU, kernel_size=3, padding=1,
                         input_shape=None):
        """Create a default ResBlock blueprint

        Args:
            prefix (str): Prefix from which the member scopes will be created
            suffix (str): Suffix to append the name of the scoped object
            parent (Blueprint): None or the instance of the parent blueprint
            skeleton (iterable): Smallest possible widths per group
            num_classes (int): Number of categories for supervised learning
            depth (int): Overall depth of the network
            width (int): Scalar to get the scaled width per group
            block_depth (int): Number of [conv/act/bn] units in the block
            conv_module (type): CNN module to use in forward. e.g. ScopedConv2d
            bn_module (type): Batch normalization module. e.g. ScopedBatchNorm2d
            linear_module (type): Linear module for classification e.g. ScopedLinear
            act_module (type): Activation module e.g ScopedReLU
            kernel_size (int or tuple): Size of the convolving kernel.
            padding (int or tuple, optional): Padding for the first convolution
            input_shape (tuple): (N, C_{in}, H_{in}, W_{in})
        """
        if input_shape is None:
            # assume batch_size = 1, in_channels: 3, h: 32, and w : 32
            input_shape = (1, 3, 32, 32)

        widths = [i * width for i in skeleton]
        stride = 1
        num_groups = len(skeleton)
        group_depth = ScopedResNet.get_num_blocks_per_group(depth, num_groups, block_depth)
        ni = skeleton[0]
        no = widths[-1]

        default = ScopedResNet.__get_default(prefix, suffix, parent, input_shape, ni, no,
                                             kernel_size, num_classes, bn_module, act_module,
                                             conv_module, linear_module, widths, group_depth,
                                             block_depth, stride, padding)
        return default




