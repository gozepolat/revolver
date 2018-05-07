# -*- coding: utf-8 -*-
from torch.nn import Module
from stacked.meta.scope import ScopedMeta
from stacked.meta.blueprint import Blueprint, make_module
from stacked.modules.scoped_nn import ScopedReLU, \
    ScopedConv2d, ScopedBatchNorm2d
from six import add_metaclass


@add_metaclass(ScopedMeta)
class ScopedConvUnit(Module):
    """BN-act-conv unit

    Args:
        scope: Scope for the self (ScopedConvUnit instance)
        blueprint: Description of bn, act, and conv
        Ignores the rest of the args
    """
    def __init__(self, scope, blueprint, *_, **__):
        super(ScopedConvUnit, self).__init__()
        self.scope = scope

        self.bn = make_module(blueprint['bn'])
        self.act = make_module(blueprint['act'])
        self.conv = make_module(blueprint['conv'])

    def forward(self, x):
        return self.function(self.bn, self.act, self.conv, x)

    @staticmethod
    def function(bn, act, conv, x):
        x = act(bn(x))
        x = conv(x)
        return x

    @staticmethod
    def set_unit_description(default, prefix, input_shape, ni, no, kernel_size,
                             stride, padding, conv_module, act_module, bn_module,
                             dilation=1, groups=1, bias=True, conv_args=None):
        """Set descriptions for act, bn, and conv"""
        suffix = '%d_%d_%d_%d_%d_%d_%d_%d' % (ni, no, kernel_size, stride,
                                              padding, dilation, groups, bias)

        default['act'] = Blueprint('%s/act' % prefix, suffix, default,
                                   False, act_module, kwargs={'inplace': True})

        default['bn'] = Blueprint('%s/bn' % prefix, suffix, default,
                                  False, bn_module, kwargs={'num_features': ni})

        if conv_args is None:
            conv_args = dict()
        if 'in_channels' not in conv_args:
            conv_args['in_channels'] = ni
        if 'out_channels' not in conv_args:
            conv_args['out_channels'] = no
        if 'kernel_size' not in conv_args:
            conv_args['kernel_size'] = kernel_size
        if 'stride' not in conv_args:
            conv_args['stride'] = stride
        if 'padding' not in conv_args:
            conv_args['padding'] = padding
        if 'dilation' not in conv_args:
            conv_args['dilation'] = dilation
        if 'groups' not in conv_args:
            conv_args['groups'] = groups
        if 'bias' not in conv_args:
            conv_args['bias'] = bias

        default['conv'] = conv_module.describe_default('%s/conv' % prefix, suffix,
                                                       default, input_shape,
                                                       conv_args['in_channels'],
                                                       conv_args['out_channels'],
                                                       conv_args['kernel_size'],
                                                       conv_args['stride'],
                                                       conv_args['padding'],
                                                       conv_args['dilation'],
                                                       conv_args['groups'],
                                                       conv_args['bias'])

    @staticmethod
    def describe_default(prefix, suffix, parent, input_shape,
                         in_channels, out_channels, kernel_size,
                         stride, padding, dilation=1, groups=1, bias=True,
                         act_module=ScopedReLU, bn_module=ScopedBatchNorm2d,
                         conv_module=ScopedConv2d, conv_args=None):
        """Create a default ScopedConvUnit blueprint"""
        default = Blueprint(prefix, suffix, parent, False, ScopedConvUnit)
        default['input_shape'] = input_shape

        ScopedConvUnit.set_unit_description(default, prefix, input_shape, in_channels, out_channels,
                                            kernel_size, stride, padding,
                                            conv_module, act_module, bn_module,
                                            dilation, groups, bias, conv_args)

        default['output_shape'] = default['conv']['output_shape']
        default['kwargs'] = {'blueprint': default, 'kernel_size': kernel_size,
                             'stride': stride, 'padding': padding,
                             'dilation': dilation, 'groups': groups, 'bias': bias}
        return default
