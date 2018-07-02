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
from torch.nn import Parameter, ParameterList
from torch.nn.init import normal
from torch.nn.functional import softmax


@add_metaclass(ScopedMeta)
class ScopedDenseGroup(Sequential):
    """Group of residual blocks with the same number of output channels"""
    def __init__(self, scope, blueprint, *_, **__):
        self.scope = scope
        self.blueprint = blueprint

        super(ScopedDenseGroup, self).__init__(blueprint)
        self.callback = blueprint['callback']
        self.drop_p = blueprint['drop_p']

        self.depth = len(self.container)
        self.scalar_weights = ParameterList()

        for i in range(2, self.depth):
            self.scalar_weights.append(Parameter(normal(torch.ones(i)),
                                                 requires_grad=True))

    def forward(self, x):
        return self.function(self.container,
                             self.callback,
                             self.depth,
                             self.scalar_weights,
                             self.scope,
                             id(self), x)

    @staticmethod
    def weighted_sum(outputs, scalars):
        index = len(outputs) - 2

        if index == -1:
            return outputs[0]

        assert(index >= 0)
        weights = softmax(scalars[index])

        summed = 0.0
        for i, o in enumerate(outputs):
            summed = o * weights[i] + summed

        return summed

    @staticmethod
    def function(container, callback, depth, scalars, scope, module_id, x):
        assert(depth > 1)

        # adjust input resolution
        x = container[0](x)

        outputs = []
        for j in range(1, depth):
            x = container[j](x)
            outputs.append(x)
            x = ScopedDenseGroup.weighted_sum(outputs, scalars)

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
        default = Blueprint(prefix, suffix, parent, False, ScopedDenseGroup)
        children = []
        default['input_shape'] = input_shape
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
            block_module = ScopedConvUnit

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
