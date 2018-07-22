# -*- coding: utf-8 -*-
from torch.nn import Module, Sequential
from stacked.meta.scope import ScopedMeta
from stacked.meta.blueprint import Blueprint, make_module
from stacked.modules.scoped_nn import ScopedReLU, \
    ScopedConv2d, ScopedBatchNorm2d, ScopedAvgPool2d, \
    ScopedDropout2d
from stacked.models.blueprinted.unit import set_conv, \
    set_batchnorm, set_pooling, set_activation, set_dropout
from stacked.utils.transformer import all_to_none
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

        self.blueprint = blueprint
        self.sequence = None
        self.callback = None
        self.update(True)

    def update(self, init=False):
        if not init:
            super(ScopedConvUnit, self).update()

        blueprint = self.blueprint

        self.callback = blueprint['callback']

        module_order = blueprint['module_order']

        self.sequence = None
        self.sequence = Sequential()

        for key in module_order:
            module = make_module(blueprint[key])
            if module is not None:
                self.sequence.add_module(key, module)

    def forward(self, x):
        return self.function(self.sequence, self.callback,
                             self.scope, id(self), x)

    @staticmethod
    def function(sequence, callback, scope,
                 module_id, x):
        x = sequence(x)
        callback(scope, module_id, x)
        return x

    @staticmethod
    def set_unit_description(default, prefix, input_shape, ni, no, kernel_size,
                             stride, padding, conv_module, act_module,
                             bn_module=all_to_none, dilation=1, groups=1,
                             bias=True, callback=all_to_none,
                             conv_kwargs=None, bn_kwargs=None, act_kwargs=None,
                             dropout_p=0.0, drop_module=ScopedDropout2d,
                             drop_kwargs=None, module_order=None,
                             pool_module=ScopedAvgPool2d, pool_kernel_size=2,
                             pool_stride=-1, pool_padding=0, pool_kwargs=None):
        """Set descriptions for act, bn, and conv"""

        suffix = '%d_%d_%d_%d_%d_%d_%d_%d' % (ni, no, kernel_size, stride,
                                              padding, dilation, groups, bias)
        if module_order is None:
            module_order = ['bn', 'act', 'conv']

        default['module_order'] = module_order

        default['callback'] = callback

        for key in module_order:
            if key == 'bn':
                set_batchnorm(default, prefix, suffix, ni, bn_module, bn_kwargs)
                default['bn']['output_shape'] = input_shape

            elif key == 'act':
                set_activation(default, prefix, suffix, True, act_module, act_kwargs)
                default['act']['output_shape'] = input_shape

            elif key == 'conv':
                conv_stride = stride
                if 'pool' in module_order:
                    conv_stride = 1

                set_conv(default, prefix, suffix, input_shape, ni, no, kernel_size,
                         conv_stride, padding, dilation, groups, bias, conv_module,
                         conv_kwargs)
                input_shape = default['conv']['output_shape']

            elif key == 'pool':
                if pool_stride == -1:
                    pool_stride = stride

                set_pooling(default, prefix, input_shape, pool_kernel_size,
                            pool_stride, pool_padding, pool_module, pool_kwargs)
                input_shape = default['pool']['output_shape']

            elif key == 'drop':
                set_dropout(default, prefix, dropout_p, drop_module, drop_kwargs)
                default['drop']['output_shape'] = input_shape

    @staticmethod
    def describe_default(prefix, suffix, parent, input_shape,
                         in_channels, out_channels, kernel_size,
                         stride, padding, dilation=1, groups=1, bias=True,
                         act_module=ScopedReLU, bn_module=ScopedBatchNorm2d,
                         conv_module=ScopedConv2d,
                         callback=all_to_none, conv_kwargs=None,
                         bn_kwargs=None, act_kwargs=None,
                         dropout_p=0.0, drop_module=ScopedDropout2d,
                         drop_kwargs=None, module_order=None,
                         pool_module=ScopedAvgPool2d, pool_kernel_size=2,
                         pool_stride=-1, pool_padding=0, pool_kwargs=None,
                         *_, **__):
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
            callback: Function to call after the output in forward is calculated
            conv_kwargs: Extra conv arguments to be used in children
            bn_kwargs: Extra bn args, if bn module requires other than 'num_features'
            act_kwargs: Extra act args, if act module requires other than defaults
            dropout_p (float): Probability of dropout
            drop_module: Dropout module, disabled if dropout_p == 0
            drop_kwargs: Extra dropout arguments
            module_order (iterable): Consecutive modules to be used e.g. ('act', 'conv')
            pool_module: Pooling module for changing the resolution
            pool_kernel_size (int or tuple): Size of pooling kernel. Default: 2
            pool_stride (int or tuple): Stride for resizing the resolution
            pool_padding (int or tuple): Padding for resizing the resolution
            pool_kwargs: Extra pool args for the pool module
        """
        default = Blueprint(prefix, suffix, parent, False, ScopedConvUnit)
        default['input_shape'] = input_shape

        ScopedConvUnit.set_unit_description(default, prefix, input_shape,
                                            in_channels, out_channels,
                                            kernel_size, stride, padding,
                                            conv_module, act_module, bn_module,
                                            dilation, groups, bias,
                                            callback, conv_kwargs,
                                            bn_kwargs, act_kwargs, dropout_p,
                                            drop_module, drop_kwargs,
                                            module_order, pool_module,
                                            pool_kernel_size, pool_stride,
                                            pool_padding, pool_kwargs)

        last_module = default['module_order'][-1]
        default['output_shape'] = default[last_module]['output_shape']

        default['kwargs'] = {'blueprint': default, 'kernel_size': kernel_size,
                             'stride': stride, 'padding': padding,
                             'dilation': dilation, 'groups': groups, 'bias': bias}
        return default
