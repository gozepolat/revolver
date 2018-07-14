# -*- coding: utf-8 -*-
from torch.nn import Module
from stacked.meta.scope import ScopedMeta
from stacked.meta.blueprint import Blueprint, make_module
from stacked.modules.scoped_nn import ScopedReLU, \
    ScopedConv2d, ScopedBatchNorm2d
from stacked.utils.transformer import all_to_none
from six import add_metaclass
from torch.nn.functional import dropout


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
        self.blueprint = blueprint

        self.bn = make_module(blueprint['bn'])
        self.act = make_module(blueprint['act'])
        self.conv = make_module(blueprint['conv'])
        self.callback = blueprint['callback']
        self.dropout_p = blueprint['dropout_p']

    def forward(self, x):
        return self.function(self.bn, self.act, self.conv,
                             self.callback, self.scope,
                             self.training, self.dropout_p,
                             id(self), x)

    @staticmethod
    def function(bn, act, conv, callback, scope,
                 training, dropout_p, module_id, x):
        if bn is not None:
            x = bn(x)
        x = act(x)

        if dropout_p > 0:
            x = dropout(x, training=training, p=dropout_p)

        x = conv(x)
        callback(scope, module_id, x)
        return x

    @staticmethod
    def set_unit_description(default, prefix, input_shape, ni, no, kernel_size,
                             stride, padding, conv_module, act_module,
                             bn_module=all_to_none, dilation=1, groups=1,
                             bias=True, callback=all_to_none, conv_kwargs=None,
                             bn_kwargs=None, act_kwargs=None, dropout_p=0.0):
        """Set descriptions for act, bn, and conv"""
        suffix = '%d_%d_%d_%d_%d_%d_%d_%d' % (ni, no, kernel_size, stride,
                                              padding, dilation, groups, bias)

        default['callback'] = callback

        if act_kwargs is None:
            if issubclass(act_module, ScopedReLU):
                act_kwargs = {'inplace': True}

        default['act'] = Blueprint('%s/act' % prefix, suffix, default,
                                   False, act_module, kwargs=act_kwargs)
        default['dropout_p'] = dropout_p

        if bn_kwargs is None:
            bn_kwargs = {'num_features': ni}
        default['bn'] = Blueprint('%s/bn' % prefix, suffix, default,
                                  False, bn_module, kwargs=bn_kwargs)

        default['conv'] = conv_module.describe_default('%s/conv' % prefix, suffix,
                                                       default, input_shape, ni,
                                                       no, kernel_size, stride,
                                                       padding, dilation, groups,
                                                       bias, conv_kwargs=conv_kwargs)

    @staticmethod
    def describe_default(prefix, suffix, parent, input_shape,
                         in_channels, out_channels, kernel_size,
                         stride, padding, dilation=1, groups=1, bias=True,
                         act_module=ScopedReLU, bn_module=ScopedBatchNorm2d,
                         conv_module=ScopedConv2d,
                         callback=all_to_none, conv_kwargs=None,
                         bn_kwargs=None, act_kwargs=None,
                         dropout_p=0.0, *_, **__):
        """Create a default ScopedConvUnit blueprint

        Args:
            prefix (str): Prefix from which the member scopes will be created
            suffix (str): Suffix to append the name of the scoped object
            parent (Blueprint): None or the instance of the parent blueprint
            in_channels (int): Number of channels in the input
            out_channels (int): Number of channels produced by the block
            kernel_size (int or tuple): Size of the convolving kernel. Default: 3
            stride (int or tuple, optional): Stride for the first convolution
            padding (int or tuple, optional): Padding for the first convolution
            input_shape (tuple): (N, C_{in}, H_{in}, W_{in})
            dilation (int): Spacing between kernel elements.
            groups (int): Number of blocked connections from input to output channels.
            bias (bool): Add a learnable bias if True
            conv_module: CNN module to use in forward. e.g. ScopedConv2d
            bn_module: Batch normalization module. e.g. ScopedBatchNorm2d
            act_module: Activation module e.g ScopedReLU
            callback: function to call after the output in forward is calculated
            conv_kwargs: extra conv arguments to be used in children
            bn_kwargs: extra bn args, if bn module requires other than 'num_features'
            act_kwargs: extra act args, if act module requires other than defaults
            dropout_p (float): Probability of dropout
        """
        default = Blueprint(prefix, suffix, parent, False, ScopedConvUnit)
        default['input_shape'] = input_shape

        ScopedConvUnit.set_unit_description(default, prefix, input_shape,
                                            in_channels, out_channels,
                                            kernel_size, stride, padding,
                                            conv_module, act_module, bn_module,
                                            dilation, groups, bias,
                                            callback, conv_kwargs,
                                            bn_kwargs, act_kwargs, dropout_p)

        default['output_shape'] = default['conv']['output_shape']
        default['kwargs'] = {'blueprint': default, 'kernel_size': kernel_size,
                             'stride': stride, 'padding': padding,
                             'dilation': dilation, 'groups': groups, 'bias': bias}
        return default
