# -*- coding: utf-8 -*-
from stacked.meta.scope import ScopedMeta
from stacked.meta.sequential import Sequential
from stacked.meta.blueprint import Blueprint
from stacked.models.blueprinted.resblock import ScopedResBlock
from stacked.modules.scoped_nn import ScopedBatchNorm2d, \
    ScopedReLU, ScopedConv2d
from stacked.utils.transformer import all_to_none
from six import add_metaclass
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
        skip = len(container)

        # optionally use vertical dropout
        for i, module in enumerate(container):
            if train and skip > i > 0 and np.random.random() < drop_p:
                skip = np.random.randint(i + 1, skip + 1)
                break

            x = module(x)

        for module in container[skip:]:
            x = module(x)

        callback(scope, module_id, x)
        return x

    @staticmethod
    def describe_default(prefix, suffix, parent, input_shape,
                         in_channels, out_channels, kernel_size, stride, padding=1,
                         dilation=1, groups=1, bias=True,
                         act_module=ScopedReLU, bn_module=ScopedBatchNorm2d,
                         conv_module=ScopedConv2d, group_depth=1, block_depth=2,
                         callback=all_to_none, drop_p=0.0,
                         dropout_p=0.0, conv3d_args=None):
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
            in_channels (int): Number of channels in the input
            out_channels (int): Number of channels produced by the block
            kernel_size (int or tuple): Size of the convolving kernel. Default: 3
            stride (int or tuple, optional): Stride for the first convolution
            padding (int or tuple, optional): Padding for the first convolution
            input_shape (tuple): (N, C_{in}, H_{in}, W_{in})
            dilation: Spacing between kernel elements.
            groups: Number of blocked connections from input to output channels.
            bias: Add a learnable bias if True
            callback: function to call after the output in forward is calculated
            drop_p: Probability of vertical drop
            dropout_p: Probability of dropout in the blocks
            conv3d_args: extra conv arguments to be used in children
        """
        default = Blueprint(prefix, suffix, parent, False, ScopedResGroup)
        children = []
        default['input_shape'] = input_shape
        for i in range(group_depth):
            block_prefix = '%s/block' % prefix
            suffix = '%d_%d_%d_%d_%d_%d_%d_%d' % (in_channels, out_channels,
                                                  kernel_size, stride,
                                                  padding, dilation, groups, bias)

            block = ScopedResBlock.describe_default(block_prefix, suffix,
                                                    default, input_shape,
                                                    in_channels, out_channels,
                                                    kernel_size, stride, padding,
                                                    dilation, groups, bias,
                                                    act_module, bn_module, conv_module,
                                                    block_depth, callback,
                                                    dropout_p, conv3d_args)
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
