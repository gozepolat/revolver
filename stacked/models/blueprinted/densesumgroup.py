# -*- coding: utf-8 -*-
import torch
from stacked.meta.scope import ScopedMeta
from stacked.meta.sequential import Sequential
from stacked.meta.blueprint import Blueprint, make_module
from stacked.models.blueprinted.resblock import ScopedResBlock
from stacked.models.blueprinted.convunit import ScopedConvUnit
from stacked.modules.scoped_nn import ScopedBatchNorm2d, \
    ScopedReLU, ScopedConv2d, ScopedParameterList
from stacked.utils.transformer import all_to_none
from six import add_metaclass
from torch.nn import Parameter
from torch.nn.init import normal
from torch.nn.functional import softmax
from torch.nn.functional import dropout


@add_metaclass(ScopedMeta)
class ScopedDenseSumGroup(Sequential):
    """Group of residual blocks with the same number of output channels"""

    def __init__(self, scope, blueprint, *_, **__):
        self.scope = scope
        self.blueprint = blueprint

        super(ScopedDenseSumGroup, self).__init__(blueprint)
        self.callback = None
        self.drop_p = None
        self.weight_sum = None
        self.depth = None
        self.transition = None
        self.scalar_weights = None
        self.update(True)

    def update(self, init=False):
        if not init:
            super(ScopedDenseSumGroup, self).update()

        blueprint = self.blueprint

        self.callback = blueprint['callback']
        self.drop_p = blueprint['drop_p']
        self.weight_sum = blueprint['weight_sum']
        self.depth = len(self.container)
        self.scalar_weights = make_module(blueprint["scalars"])
        self.transition = blueprint['input_shape'][2] != blueprint['output_shape'][2]

        if self.weight_sum:
            for i in range(len(self.scalar_weights) + 2, self.depth + 1):
                self.scalar_weights.append(Parameter(normal(torch.ones(i)),
                                                     requires_grad=True))

    def forward(self, x):
        return self.function(self.container,
                             self.callback,
                             self.depth,
                             self.transition,
                             self.scalar_weights,
                             self.drop_p,
                             self.weight_sum,
                             self.training,
                             self.scope,
                             id(self), x)

    @staticmethod
    def weighted_sum(outputs, scalars, drop_p, weight_sum, training):
        index = len(outputs) - 2

        if index == -1:
            return outputs[0]

        assert (index >= 0)

        summed = 0.0
        if not weight_sum:
            for i, o in enumerate(outputs):
                summed = o + summed
            return summed

        weights = scalars[index]
        if drop_p > 0:
            weights = dropout(weights, training=training, p=drop_p)

        weights = softmax(weights, dim=0)
        for i, o in enumerate(outputs):
            summed = o * weights[i] + summed

        return summed

    @staticmethod
    def function(container, callback, depth,
                 transition, scalars, drop_p, weight_sum,
                 training, scope, module_id, x):
        # assert (depth > 1)
        i = 0

        t = x
        # adjust input resolution
        if transition or depth == 1:
            t = x = container[0](x)
            i = 1

        outputs = []
        for j in range(i, depth):
            x = container[j](x)
            outputs.append(x)
            x = ScopedDenseSumGroup.weighted_sum(outputs, scalars, drop_p,
                                                 weight_sum, training)

        # preserve previous input
        outputs.append(t)

        x = torch.cat(outputs, dim=1)
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
                         scalar_container=ScopedParameterList, weight_sum=False,
                         mutation_p=0.8,
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
            scalar_container: Sequential container of scalars for dense layers
            weight_sum (bool): Weight sum and then softmax the reused blocks or not
            mutation_p (float): How much mutation is allowed as default
        """
        default = Blueprint(prefix, suffix, parent, False, ScopedDenseSumGroup)
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

        module_order = ["bn", "act", "conv"]
        concat_out_channels = in_channels
        if stride > 1:
            block_prefix = '%s/block' % prefix
            suffix = '_'.join([str(s) for s in (in_channels, in_channels // 2,
                                                kernel_size, stride, padding,
                                                dilation, groups, bias)])
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
            concat_out_channels = in_channels
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
                                                  dropout_p=dropout_p, residual=residual,
                                                  module_order=module_order)
            input_shape = block['output_shape']
            concat_out_channels += input_shape[1]
            children.append(block)

            # for the next groups, stride and in_channels are changed
            stride = 1
            in_channels = block['output_shape'][1]
            block_module = dense_unit_module

        output_shape = (input_shape[0], concat_out_channels, input_shape[2], input_shape[3])
        default['scalars'] = Blueprint("%s/scalars" % prefix, "", default,
                                       False, scalar_container)
        default['drop_p'] = drop_p
        default['weight_sum'] = weight_sum
        default['callback'] = callback
        default['children'] = children
        default['depth'] = len(children)
        default['output_shape'] = output_shape
        default['mutation_p'] = mutation_p
        return default
