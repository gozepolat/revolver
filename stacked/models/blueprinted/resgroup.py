# -*- coding: utf-8 -*-
from stacked.meta.scope import ScopedMeta
from stacked.meta.sequential import Sequential
from stacked.meta.blueprint import Blueprint
from stacked.models.blueprinted.resblock import ScopedResBlock
from six import add_metaclass


@add_metaclass(ScopedMeta)
class ScopedResGroup(Sequential):
    """Group of residual blocks with the same number of output channels"""
    def __init__(self, scope, blueprint, *_, **__):
        self.scope = scope
        super(ScopedResGroup, self).__init__(blueprint)

    def forward(self, x):
        return self.function(self.container, x)

    @staticmethod
    def function(container, x):
        for module in container:
            x = module(x)
        return x

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
            suffix = '%d_%d_%d_%d_%d' % (ni, no, kernel_size, stride, padding)
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
