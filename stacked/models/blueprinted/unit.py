# -*- coding: utf-8 -*-
from stacked.modules.conv import get_conv_out_shape
from stacked.modules.scoped_nn import ScopedReLU, \
    ScopedConv2d, ScopedBatchNorm2d, ScopedAvgPool2d, \
    ScopedDropout2d
from stacked.meta.blueprint import Blueprint
from stacked.utils.transformer import all_to_none


def set_pooling(default, prefix, input_shape,
                kernel_size=2, stride=2, padding=0,
                module=ScopedAvgPool2d, kwargs=None):
    """Add a pooling module to the blueprint description"""
    if kwargs is None:
        kwargs = {'kernel_size': kernel_size,
                  'stride': stride,
                  'padding': padding}

    default['pool'] = Blueprint("%s/pool" % prefix,
                                "%d_%d_%d" % (kernel_size, stride, padding),
                                default, False, module,
                                kwargs=kwargs)

    default['pool']['input_shape'] = input_shape

    shape = get_conv_out_shape(input_shape, input_shape[1], **kwargs)
    default['pool']['output_shape'] = shape


def set_activation(default, prefix, suffix, inplace=False,
                   module=ScopedReLU, kwargs=None):
    """Add an activation module to the blueprint description"""
    if kwargs is None:
        if issubclass(module, ScopedReLU):
            kwargs = {'inplace': inplace}

    default['act'] = Blueprint('%s/act' % prefix, suffix, default,
                               False, module, kwargs=kwargs)


def set_dropout(default, prefix, dropout_p=0.0,
                module=ScopedDropout2d, kwargs=None):
    """Add a dropout module to the blueprint description"""
    if kwargs is None:
        kwargs = {'p': dropout_p, 'inplace': False}

    if dropout_p == 0.0:
        module = all_to_none

    default['drop'] = Blueprint('%s/dropout' % prefix,
                                '%d' % dropout_p, default, False,
                                module, kwargs=kwargs)


def set_convdim(default, prefix, input_shape, ni, no,
                stride, dilation, groups, bias,
                conv_module=ScopedConv2d, residual=True):
    """Add a conv module blueprint for channel or resolution adjustment"""
    suffix = '%d_%d_%d_%d_%d_%d_%d_%d' % (ni, no, 1, stride,
                                          0, dilation, groups, bias)

    default['convdim'] = conv_module.describe_default('%s/convdim' % prefix,
                                                      suffix, default,
                                                      input_shape, ni, no, 1,
                                                      stride, 0, dilation,
                                                      groups, bias)

    if ni == no and stride == 1 or not residual:
        default['convdim']['type'] = all_to_none


def set_batchnorm(default, prefix, suffix, input_shape,
                  module=ScopedBatchNorm2d, kwargs=None, mutation_p=0.2):
    """Add a batch normalization module to the blueprint description"""
    default['bn'] = module.describe_default(prefix=prefix,
                                            suffix=suffix,
                                            parent=default,
                                            input_shape=input_shape,
                                            mutation_p=mutation_p,
                                            kwargs=kwargs)


def set_conv(default, prefix, suffix, input_shape, ni, no,
             kernel_size, stride, padding, dilation,
             groups, bias, module=ScopedConv2d, conv_kwargs=None):
    """Add a convolution module to the blueprint description"""
    default['conv'] = module.describe_default('%s/conv' % prefix, suffix,
                                              default, input_shape, ni,
                                              no, kernel_size, stride,
                                              padding, dilation, groups,
                                              bias, conv_kwargs=conv_kwargs)


def describe_from_blueprint(prefix, suffix, blueprint, parent,
                            kernel_size=None, stride=None, padding=None,
                            dilation=None, groups=None, bias=None,
                            conv_kwargs=None, module=ScopedConv2d):
    input_shape = blueprint['input_shape']
    output_shape = blueprint['output_shape']
    kwargs = blueprint['kwargs']
    args = ScopedConv2d.adjust_args(kwargs, kernel_size, stride,
                                    padding, dilation, groups, bias)

    _kernel, _stride, _padding, _dilation, _groups, _bias = args
    return module.describe_default(prefix=prefix,
                                   suffix=suffix,
                                   parent=parent,
                                   input_shape=input_shape,
                                   in_channels=input_shape[1],
                                   out_channels=output_shape[1],
                                   kernel_size=_kernel, stride=_stride,
                                   padding=_padding,
                                   dilation=_dilation, groups=_groups,
                                   bias=_bias,
                                   conv_module=blueprint['type'],
                                   conv_kwargs=conv_kwargs, )
