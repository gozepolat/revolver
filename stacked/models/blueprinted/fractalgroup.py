# -*- coding: utf-8 -*-
from stacked.meta.scope import ScopedMeta
from stacked.meta.sequential import Sequential
from stacked.meta.blueprint import Blueprint, make_module
from stacked.models.blueprinted.resblock import ScopedResBlock
from stacked.modules.scoped_nn import ScopedBatchNorm2d, \
    ScopedReLU, ScopedConv2d
from stacked.utils.transformer import all_to_none
from six import add_metaclass
import numpy as np


@add_metaclass(ScopedMeta)
class ScopedFractalGroup(Sequential):
    """Group of residual blocks with the same number of output channels"""
    def __init__(self, scope, blueprint, *_, **__):
        self.scope = scope
        self.blueprint = blueprint

        super(ScopedFractalGroup, self).__init__(blueprint)
        self.callback = blueprint['callback']
        self.drop_p = blueprint['drop_p']
        self.left = make_module(blueprint['left'])

    def forward(self, x, left_scope=None, left_x=None):
        return self.function(left_scope,
                             left_x,
                             self.left,
                             self.container,
                             self.callback,
                             self.scope,
                             self.drop_p,
                             self.training,
                             id(self), x)

    @staticmethod
    def function(left_scope, left_x, left,
                 container, callback, scope, drop_p,
                 train, module_id, x):
        # don't reuse left_x for modules with different scope
        if left_scope != left.scope:
            left_x = left(x)

        length = len(container)
        i = 0
        while i < length:
            if i == 0:
                x = container[i](x, left.scope, left_x)
            else:
                x = container[i](x)
            i += 1
            if train and np.random.random() < drop_p:
                i = np.random.randint(i, length + 1)

        callback(scope, module_id, x)
        return (x + left_x) * 0.5

    @staticmethod
    def describe_default(prefix, suffix, parent, input_shape,
                         in_channels, out_channels, kernel_size, stride, padding=1,
                         dilation=1, groups=1, bias=True, block_module=ScopedResBlock,
                         act_module=ScopedReLU, bn_module=ScopedBatchNorm2d,
                         conv_module=ScopedConv2d, group_depth=1, block_depth=2,
                         callback=all_to_none, drop_p=0.0,
                         dropout_p=0.0, residual=True, conv_kwargs=None,
                         bn_kwargs=None, act_kwargs=None):
        """Create a default FractalGroup blueprint

        Args:
            prefix (str): Prefix from which the member scopes will be created
            suffix (str): Suffix to append the name of the scoped object
            parent (Blueprint): None or the instance of the parent blueprint
            group_depth: Number of blocks in the group
            block_depth: Number of [conv/act/bn] units in the block
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
            block_module: Children modules used as block modules
            callback: function to call after the output in forward is calculated
            drop_p: Probability of vertical drop
            dropout_p: Probability of dropout in the blocks
            residual (bool): True if a shortcut connection will be used
            conv_kwargs: extra conv arguments to be used in children
            bn_kwargs: extra bn args, if bn module requires other than 'num_features'
            act_kwargs: extra act args, if act module requires other than defaults
        """
        default = Blueprint(prefix, suffix, parent, False, ScopedFractalGroup)
        children = []
        default['input_shape'] = input_shape
        for i in range(group_depth):
            block_prefix = '%s/block' % prefix
            suffix = '%d_%d_%d_%d_%d_%d_%d_%d' % (in_channels, out_channels,
                                                  kernel_size, stride,
                                                  padding, dilation, groups, bias)

            # block_module, group_depth, drop_p to be added as kwargs
            block = block_module.describe_default(block_prefix, suffix,
                                                  default, input_shape,
                                                  in_channels, out_channels,
                                                  kernel_size, stride, padding,
                                                  dilation, groups, bias,
                                                  act_module, bn_module, conv_module,
                                                  block_depth, callback,
                                                  dropout_p, residual, conv_kwargs,
                                                  bn_kwargs, act_kwargs)
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
        default['kwargs'] = {'blueprint': default,
                             'kernel_size': kernel_size,
                             'stride': stride,
                             'padding': padding,
                             'dilation': dilation,
                             'groups': groups,
                             'bias': bias}
        return default
