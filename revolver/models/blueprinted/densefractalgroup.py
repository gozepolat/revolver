# -*- coding: utf-8 -*-
from revolver.meta.scope import ScopedMeta
from revolver.meta.sequential import Sequential
from revolver.meta.blueprint import Blueprint
from revolver.models.blueprinted.convunit import ScopedConvUnit
from revolver.modules.scoped_nn import ScopedBatchNorm2d, \
    ScopedReLU, ScopedConv2d
from revolver.utils.transformer import all_to_none
from revolver.utils.common import time_to_drop
from six import add_metaclass
import torch


@add_metaclass(ScopedMeta)
class ScopedDenseFractalGroup(Sequential):
    """Group of dense fractals with the same number of output channels"""

    def __init__(self, scope, blueprint, *_, **__):
        self.scope = scope
        self.blueprint = blueprint

        super(ScopedDenseFractalGroup, self).__init__(blueprint)
        self.callback = None
        self.dropout_p = None
        self.drop_p = None
        self.depth = None
        self.squeeze = None
        self.scalar_weights = None
        self.update(True)

    def update(self, init=False):
        if not init:
            super(ScopedDenseFractalGroup, self).update()

        blueprint = self.blueprint

        self.callback = blueprint['callback']
        self.drop_p = blueprint['drop_p']
        self.squeeze = blueprint['squeeze']
        self.depth = len(self.container)

    def forward(self, x):
        return self.function(self.container,
                             self.callback,
                             self.depth,
                             self.scope,
                             self.training,
                             self.drop_p,
                             self.squeeze,
                             id(self), x)

    @staticmethod
    def function(container, callback, depth, scope,
                 training, drop_p, squeeze, module_id, x):
        assert (depth > 0)
        half = depth // 2

        if squeeze and time_to_drop(training, drop_p):
            half = 0

        dense_inputs = []
        fractal_out = 0.0
        for j in range(half):
            o = container[j](x)
            dense_inputs.append(o)
            o = container[j + half](o)
            fractal_out = o + fractal_out

        if half == 0:
            fractal_out = container[0](x)

        # concatenate with previous inputs
        if not squeeze and half > 0:
            dense_inputs.append(fractal_out / half)
            fractal_out = torch.cat(dense_inputs, dim=1)

        callback(scope, module_id, fractal_out)
        return fractal_out

    @staticmethod
    def describe_default(prefix, suffix, parent, input_shape,
                         in_channels, out_channels, kernel_size, stride, padding=1,
                         dilation=1, groups=1, bias=True,
                         act_module=ScopedReLU, bn_module=ScopedBatchNorm2d,
                         conv_module=ScopedConv2d, callback=all_to_none,
                         conv_kwargs=None, bn_kwargs=None, act_kwargs=None,
                         unit_module=ScopedConvUnit, block_depth=2,
                         dropout_p=0.0, residual=True, block_module=ScopedConvUnit,
                         group_depth=2, drop_p=0.0, fractal_depth=1,
                         squeeze=False, *_, **__):
        """Create a default DenseFractalGroup blueprint

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
            block_module: Children modules used as the output units or smallest fractal
            group_depth (int): Number of blocks in the group
            drop_p (float): Probability of vertical drop
            fractal_depth (int): Recursion depth for dense fractal group module
            squeeze (bool): Sum the outputs of smaller fractals if true, else concat
        """
        default = Blueprint(prefix, suffix, parent, False, ScopedDenseFractalGroup)
        children = []
        default['input_shape'] = input_shape
        default['kwargs'] = {'blueprint': default,
                             'kernel_size': kernel_size,
                             'stride': stride,
                             'padding': padding,
                             'dilation': dilation,
                             'groups': groups,
                             'bias': bias}

        suffix = '_'.join([str(s) for s in (in_channels, out_channels,
                                            kernel_size, stride,
                                            padding, dilation, groups,
                                            bias)])
        module_order = ["bn", "act", "conv"]
        unit_fractal = block_module.describe_default('%s/unit' % prefix, suffix,
                                                     default, input_shape,
                                                     in_channels, out_channels,
                                                     kernel_size, stride, padding,
                                                     dilation, groups, bias,
                                                     act_module, bn_module, conv_module,
                                                     callback, conv_kwargs,
                                                     bn_kwargs, act_kwargs,
                                                     dropout_p=dropout_p,
                                                     residual=residual,
                                                     module_order=module_order)
        children.append(unit_fractal)

        block_prefix = '%s/block' % prefix

        module = ScopedDenseFractalGroup
        for i in range(1, fractal_depth):
            block = module.describe_default(block_prefix, "%s_%d" % (suffix, i),
                                            default, input_shape,
                                            in_channels, out_channels,
                                            kernel_size, stride, padding,
                                            dilation, groups, bias,
                                            act_module, bn_module, conv_module,
                                            callback, conv_kwargs,
                                            bn_kwargs, act_kwargs, unit_module=unit_module,
                                            block_depth=block_depth, dropout_p=dropout_p,
                                            residual=residual,
                                            block_module=block_module,
                                            group_depth=group_depth, drop_p=drop_p,
                                            fractal_depth=fractal_depth - i,
                                            squeeze=True)
            children.append(block)

        for i in range(fractal_depth):
            input_shape = children[i]['output_shape']
            in_channels = input_shape[1]
            suffix = '_'.join([str(s) for s in (in_channels, out_channels,
                                                kernel_size, 1,
                                                padding, dilation, groups, bias)])
            unit = block_module.describe_default('%s/unit' % block_prefix, suffix,
                                                 default, input_shape,
                                                 in_channels, out_channels,
                                                 kernel_size, 1, padding,
                                                 dilation, groups, bias,
                                                 act_module, bn_module,
                                                 conv_module, callback,
                                                 conv_kwargs, bn_kwargs,
                                                 act_kwargs, dropout_p=dropout_p,
                                                 residual=residual,
                                                 module_order=module_order)
            children.append(unit)

        out_channels = out_channels * (fractal_depth + 1) if fractal_depth > 0 and not squeeze else out_channels
        output_shape = (input_shape[0], out_channels, input_shape[2], input_shape[3])
        default['squeeze'] = squeeze
        default['drop_p'] = drop_p
        default['dropout_p'] = dropout_p
        default['callback'] = callback
        default['children'] = children
        default['depth'] = len(children)
        default['fractal_depth'] = fractal_depth
        default['output_shape'] = output_shape
        return default
