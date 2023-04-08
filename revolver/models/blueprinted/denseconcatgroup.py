# -*- coding: utf-8 -*-
import torch
from revolver.meta.scope import ScopedMeta
from revolver.meta.sequential import Sequential
from revolver.meta.blueprint import Blueprint
from revolver.models.blueprinted.resblock import ScopedResBlock
from revolver.models.blueprinted.convunit import ScopedConvUnit
from revolver.modules.scoped_nn import ScopedBatchNorm2d, \
    ScopedReLU, ScopedConv2d
from revolver.utils.transformer import all_to_none
from six import add_metaclass
from revolver.utils import common


@add_metaclass(ScopedMeta)
class ScopedDenseConcatGroup(Sequential):
    """Group of residual blocks with the same number of output channels"""

    def __init__(self, scope, blueprint, *_, **__):
        self.scope = scope
        self.blueprint = blueprint

        super(ScopedDenseConcatGroup, self).__init__(blueprint)
        self.callback = None
        self.depth = None
        self.scalar_weights = None
        self.transition = None
        self.update(True)

    def update(self, init=False):
        if not init:
            super(ScopedDenseConcatGroup, self).update()

        blueprint = self.blueprint

        self.callback = blueprint['callback']
        self.depth = len(self.container)
        self.transition = blueprint['input_shape'][2] != blueprint['output_shape'][2]

    def forward(self, x):
        return self.function(self.container,
                             self.callback,
                             self.depth,
                             self.transition,
                             self.scope,
                             id(self), x)

    @staticmethod
    def function(container, callback, depth,
                 transition, scope, module_id, x):
        i = 0
        if transition or depth == 1:
            x = container[0](x)
            i = 1

        for j in range(i, depth):
            o = container[j](x)
            x = torch.cat((x, o), dim=1)

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
                         mutation_p=0.8, toggle_p=common.UNIQUENESS_TOGGLE_P,
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
        if group_depth <= 1:
            group_depth = 2

        if stride > 1:
            block_prefix = '%s/block' % prefix
            suffix = '_'.join([str(s) for s in (in_channels, in_channels // 2,
                                                kernel_size, stride, padding,
                                                dilation, groups, bias)])
            module_order = ["bn", "act", "conv"]
            block = ScopedConvUnit.describe_default(block_prefix, suffix,
                                                    default, input_shape,
                                                    in_channels, in_channels // 2,
                                                    kernel_size, stride, padding, dilation,
                                                    groups, bias, act_module,
                                                    bn_module, conv_module,
                                                    callback, conv_kwargs,
                                                    bn_kwargs, act_kwargs,
                                                    module_order=module_order)

            input_shape = block['output_shape']
            children.append(block)
            in_channels = in_channels // 2
            stride = 1

        for i in range(group_depth):
            block_prefix = '%s/block' % prefix
            suffix = '_'.join([str(s) for s in (in_channels, out_channels,
                                                kernel_size, stride,
                                                padding, dilation, groups, bias)])

            block = block_module.describe_default(block_prefix, suffix,
                                                  default, input_shape,
                                                  in_channels, out_channels,
                                                  kernel_size, stride, padding,
                                                  dilation, groups, bias,
                                                  act_module, bn_module, conv_module,
                                                  callback, conv_kwargs,
                                                  bn_kwargs, act_kwargs,
                                                  unit_module=unit_module,
                                                  block_depth=block_depth,
                                                  dropout_p=dropout_p,
                                                  residual=residual)
            in_channels += out_channels

            input_shape = (block['output_shape'][0], in_channels,
                           block['output_shape'][2], block['output_shape'][3])
            children.append(block)

            # for the next blocks, stride and in_channels are changed
            stride = 1
            block_module = dense_unit_module

        output_shape = input_shape
        default['drop_p'] = drop_p
        default['dropout_p'] = dropout_p
        default['callback'] = callback
        default['children'] = children
        default['depth'] = len(children)
        default['output_shape'] = output_shape
        default['mutation_p'] = mutation_p
        default['toggle_p'] = toggle_p
        return default
