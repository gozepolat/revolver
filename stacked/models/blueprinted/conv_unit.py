# -*- coding: utf-8 -*-
from torch.nn import Module
from stacked.meta.scope import ScopedMeta
from stacked.meta.blueprint import Blueprint, make_module
from stacked.modules.conv import get_conv_out_shape
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
                             stride, padding, conv_module, act_module, bn_module):
        """Set descriptions for act, bn, and conv"""
        # describe act and bn
        suffix = '%d_%d_%d_%d_%d' % (ni, no, kernel_size, stride, padding)
        default['act'] = Blueprint('%s/act' % prefix, suffix, default,
                                   False, act_module, kwargs={'inplace': True})
        default['bn'] = Blueprint('%s/bn' % prefix, suffix, default,
                                  False, bn_module, kwargs={'num_features': ni})
        # describe conv
        kwargs = {'in_channels': ni, 'out_channels': no,
                  'kernel_size': kernel_size, 'stride': stride, 'padding': padding}
        conv = Blueprint('%s/conv' % prefix, suffix, default,
                         False, conv_module, kwargs=kwargs)
        conv['input_shape'] = input_shape
        conv['output_shape'] = get_conv_out_shape(input_shape, no,
                                                  kernel_size, stride, padding)
        default['conv'] = conv

    @staticmethod
    def describe_default(prefix, suffix, parent, input_shape, ni, no, kernel_size,
                         stride, padding, act_module, bn_module, conv_module):
        """Create a default ScopedConvUnit blueprint"""
        default = Blueprint(prefix, suffix, parent, False, ScopedConvUnit)
        default['input_shape'] = input_shape

        ScopedConvUnit.set_unit_description(default, prefix, input_shape, ni, no,
                                            kernel_size, stride, padding,
                                            conv_module, act_module, bn_module)

        default['output_shape'] = default['conv']['output_shape']
        default['kwargs'] = {'blueprint': default}
        return default
