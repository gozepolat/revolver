# -*- coding: utf-8 -*-
import torch
from stacked.meta.scope import ScopedMeta
from stacked.meta.sequential import Sequential
from stacked.meta.blueprint import Blueprint
from stacked.models.blueprinted.resblock import ScopedResBlock
from stacked.models.blueprinted.conv_unit import ScopedConvUnit
from stacked.modules.scoped_nn import ScopedBatchNorm2d, \
    ScopedReLU, ScopedConv2d
from stacked.utils.transformer import all_to_none
from six import add_metaclass
from torch.nn.functional import dropout


@add_metaclass(ScopedMeta)
class ScopedDenseConcatGroup(Sequential):
    """Group of residual blocks with the same number of output channels"""
    def __init__(self, scope, blueprint, *_, **__):
        self.scope = scope
        self.blueprint = blueprint

        super(ScopedDenseConcatGroup, self).__init__(blueprint)
        self.callback = None
        self.dropout_p = None
        self.drop_p = None
        self.depth = None
        self.scalar_weights = None
        self.update(True)

    def update(self, init=False):
        if not init:
            super(ScopedDenseConcatGroup, self).update()

        blueprint = self.blueprint

        self.callback = blueprint['callback']
        self.drop_p = blueprint['drop_p']
        self.dropout_p = blueprint['dropout_p']
        self.depth = len(self.container)

    def forward(self, x):
        return self.function(self.container,
                             self.callback,
                             self.depth,
                             self.scope,
                             self.training,
                             self.dropout_p,
                             id(self), x)

    @staticmethod
    def function(container, callback, depth, scope,
                 training, dropout_p, module_id, x):
        assert(depth > 1)

        # adjust input resolution
        x = container[0](x)

        for j in range(1, depth):
            if dropout_p > 0:
                x = dropout(x, training=training, p=dropout_p)
            o = container[j](x)
            x = torch.cat((x, o), axis=1)

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
                         group_depth=2, drop_p=0.0, dense_unit_module=ScopedConvUnit,
                         *_, **__):
        """Create a default DenseGroup blueprint

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
            dilation (int): Spacing between kernel elements.
            groups (int): Number of blocked connections from input to output channels.
            bias (bool): Add a learnable bias if True
            unit_module: Children modules used as units in block_modules
            callback: function to call after the output in forward is calculated
            conv_kwargs: extra conv arguments to be used in children
            bn_kwargs: extra bn args, if bn module requires other than 'num_features'
            act_kwargs: extra act args, if act module requires other than defaults
            unit_module: Children modules for the head block that changes resolution
            block_depth (int): Number of [conv/act/bn] units in the block
            dropout_p (float): Probability of dropout in the blocks
            residual (bool): True if a shortcut connection will be used
            block_module: Children modules used as block modules
            group_depth (int): Number of blocks in the group
            drop_p (float): Probability of vertical drop
            dense_unit_module: Children modules that will be used in dense connections
        """
        default = Blueprint(prefix, suffix, parent, False, ScopedDenseConcatGroup)
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

            # for the next blocks, stride and in_channels are changed
            stride = 1
            if i == 0:
                in_channels = out_channels
            else:
                in_channels += out_channels
            block_module = dense_unit_module

        default['drop_p'] = drop_p
        default['dropout_p'] = dropout_p
        default['callback'] = callback
        default['children'] = children
        default['depth'] = len(children)
        default['output_shape'] = input_shape
        return default
