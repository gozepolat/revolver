# -*- coding: utf-8 -*-
from stacked.meta.scope import ScopedMeta
from stacked.meta.sequential import Sequential
from stacked.meta.blueprint import Blueprint
from stacked.models.blueprinted.resblock import ScopedResBlock
from stacked.models.blueprinted.conv_unit import ScopedConvUnit
from stacked.modules.scoped_nn import ScopedBatchNorm2d, \
    ScopedReLU, ScopedConv2d
from stacked.utils.transformer import all_to_none
from six import add_metaclass
from stacked.utils.common import time_to_drop
import numpy as np


@add_metaclass(ScopedMeta)
class ScopedResGroup(Sequential):
    """Group of residual blocks with the same number of output channels"""
    def __init__(self, scope, blueprint, *_, **__):
        self.scope = scope
        self.blueprint = blueprint

        super(ScopedResGroup, self).__init__(blueprint)
        self.callback = blueprint['callback']
        self.drop_p = blueprint['drop_p']

    def forward(self, x):
        return self.function(self.container,
                             self.callback,
                             self.scope,
                             self.drop_p,
                             self.training,
                             id(self), x)

    @staticmethod
    def function(container, callback, scope, drop_p, train, module_id, x):
        length = len(container)
        i = 0
        while i < length:
            x = container[i](x)
            i += 1
            if time_to_drop(train, drop_p):
                i = np.random.randint(i, length + 1)

        callback(scope, module_id, x)
        return x

    @staticmethod
    def describe_default(prefix, suffix, parent, input_shape,
                         in_channels, out_channels, kernel_size, stride, padding=1,
                         dilation=1, groups=1, bias=True,
                         act_module=ScopedReLU, bn_module=ScopedBatchNorm2d,
                         conv_module=ScopedConv2d, callback=all_to_none,
                         conv_kwargs=None, bn_kwargs=None, act_kwargs=None,
                         unit_module=ScopedConvUnit, block_depth=2,
                         dropout_p=0.0, residual=True, block_module=ScopedResBlock,
                         group_depth=2, drop_p=0.0, *_, **__):
        """Create a default ResGroup blueprint

        Args:
            prefix (str): Prefix from which the member scopes will be created
            suffix (str): Suffix to append the name of the scoped object
            parent (Blueprint): None or the instance of the parent blueprint
            conv_module (type): CNN module to use in forward. e.g. ScopedConv2d
            bn_module: Batch normalization module. e.g. ScopedBatchNorm2d
            act_module (type): Activation module e.g ScopedReLU
            in_channels (int): Number of channels in the input
            out_channels (int): Number of channels produced by the block
            kernel_size (int or tuple): Size of the convolving kernel. Default: 3
            stride (int or tuple, optional): Stride for the first convolution
            padding (int or tuple, optional): Padding for the first convolution
            input_shape (tuple): (N, C_{in}, H_{in}, W_{in})
            dilation: Spacing between kernel elements.
            groups: Number of blocked connections from input to output channels.
            bias: Add a learnable bias if True
            unit_module: Children modules used as units in block_modules
            callback: function to call after the output in forward is calculated
            conv_kwargs: extra conv arguments to be used in children
            bn_kwargs: extra bn args, if bn module requires other than 'num_features'
            act_kwargs: extra act args, if act module requires other than defaults
            unit_module: Children modules used as block modules
            block_depth: Number of [conv/act/bn] units in the block
            dropout_p: Probability of dropout in the blocks
            residual (bool): True if a shortcut connection will be used
            block_module: Children modules used as block modules
            group_depth: Number of blocks in the group
            drop_p: Probability of vertical drop
        """
        default = Blueprint(prefix, suffix, parent, False, ScopedResGroup)
        children = []
        default['input_shape'] = input_shape
        default['kwargs'] = {'blueprint': default,
                             'kernel_size': kernel_size,
                             'stride': stride,
                             'padding': padding,
                             'dilation': dilation,
                             'groups': groups,
                             'bias': bias}

        for i in range(group_depth):
            block_prefix = '%s/block' % prefix
            suffix = '%d_%d_%d_%d_%d_%d_%d_%d' % (in_channels, out_channels,
                                                  kernel_size, stride,
                                                  padding, dilation, groups, bias)

            block = block_module.describe_default(block_prefix, suffix,
                                                  default, input_shape,
                                                  in_channels, out_channels,
                                                  kernel_size, stride, padding,
                                                  dilation, groups, bias,
                                                  act_module, bn_module, conv_module,
                                                  callback, conv_kwargs,
                                                  bn_kwargs, act_kwargs, unit_module,
                                                  block_depth, dropout_p, residual)
            input_shape = block['output_shape']
            children.append(block)

            # for the next groups, stride and in_channels are changed
            stride = 1
            in_channels = out_channels

        default['drop_p'] = drop_p
        default['callback'] = callback
        default['children'] = children
        default['depth'] = len(children)
        default['output_shape'] = input_shape
        return default
