# -*- coding: utf-8 -*-
from stacked.meta.scope import ScopedMeta
from stacked.meta.sequential import Sequential
from stacked.meta.blueprint import Blueprint, make_module
from stacked.models.blueprinted.convunit import ScopedConvUnit
from stacked.models.blueprinted.resblock import ScopedResBlock
from stacked.modules.scoped_nn import ScopedBatchNorm2d, \
    ScopedReLU, ScopedConv2d
from stacked.utils.transformer import all_to_none
from stacked.utils.common import time_to_drop
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

        # don't reuse left_x for different scopes
        if left_scope != left.scope:
            left_x = left(x)

        # drop left or right uniformly
        drop_left = np.random.randint(0, 2) == 1
        drop_allowed = time_to_drop(train, drop_p)

        # drop right
        if drop_allowed and not drop_left:
            callback(scope, module_id, left_x)
            return left_x

        length = len(container)
        assert(length > 0)
        right_x = container[0](x, left.scope, left_x)

        i = 1
        while i < length:
            if time_to_drop(train, drop_p):
                i = np.random.randint(i, length + 1)
            right_x = container[i](right_x)
            i += 1

        # drop left
        if drop_allowed and drop_left:
            callback(scope, module_id, right_x)
            return right_x

        callback(scope, module_id, x)
        return (right_x + left_x) * 0.5

    @staticmethod
    def describe_default(prefix, suffix, parent, input_shape,
                         in_channels, out_channels, kernel_size, stride, padding=1,
                         dilation=1, groups=1, bias=True,
                         act_module=ScopedReLU, bn_module=ScopedBatchNorm2d,
                         conv_module=ScopedConv2d, callback=all_to_none,
                         conv_kwargs=None, bn_kwargs=None, act_kwargs=None,
                         unit_module=ScopedConvUnit, block_depth=2,
                         dropout_p=0.0, residual=True, block_module=ScopedResBlock,
                         group_depth=1, drop_p=0.0, fractal_depth=1, mutation_p=0.2,
                         *_, **__):
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
            kernel_size (int or tuple): Size of the conv kernel. Default: 3
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
            fractal_depth: recursion depth for fractal group module
            mutation_p (float): How much mutation is allowed as default
        """
        default = Blueprint(prefix, suffix, parent, False, ScopedFractalGroup)
        children = []
        default['input_shape'] = input_shape

        module = ScopedFractalGroup
        if fractal_depth < 2:
            module = block_module

        default['left'] = unit_module.describe_default('%s/unit' % prefix, suffix,
                                                       default, input_shape,
                                                       in_channels, out_channels,
                                                       kernel_size, stride, padding,
                                                       dilation, groups, bias,
                                                       act_module, bn_module, conv_module,
                                                       callback, conv_kwargs,
                                                       bn_kwargs, act_kwargs,
                                                       dropout_p=dropout_p)

        block_prefix = '%s/block' % prefix
        for i in range(group_depth):
            suffix = '%d_%d_%d_%d_%d_%d_%d_%d' % (in_channels, out_channels,
                                                  kernel_size, stride,
                                                  padding, dilation, groups, bias)

            block = module.describe_default(block_prefix, suffix,
                                            default, input_shape,
                                            in_channels, out_channels,
                                            kernel_size, stride, padding,
                                            dilation, groups, bias,
                                            act_module, bn_module, conv_module,
                                            callback, conv_kwargs,
                                            bn_kwargs, act_kwargs,
                                            unit_module=unit_module,
                                            block_depth=block_depth,
                                            dropout_p=dropout_p, residual=residual,
                                            block_module=block_module,
                                            group_depth=group_depth, drop_p=drop_p,
                                            fractal_depth=fractal_depth - 1)
            input_shape = block['output_shape']
            children.append(block)

            # for the next groups, stride and in_channels are changed
            stride = 1
            in_channels = out_channels

        default['drop_p'] = drop_p
        default['callback'] = callback
        default['children'] = children
        default['fractal_depth'] = fractal_depth
        default['depth'] = len(children)
        default['output_shape'] = input_shape
        default['kwargs'] = {'blueprint': default,
                             'kernel_size': kernel_size,
                             'stride': stride,
                             'padding': padding,
                             'dilation': dilation,
                             'groups': groups,
                             'bias': bias}
        default['mutation_p'] = mutation_p
        return default
