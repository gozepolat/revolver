# -*- coding: utf-8 -*-
from torch.nn import Module
from stacked.modules.scoped_nn import ScopedConv2d
from stacked.meta.scope import ScopedMeta
from stacked.meta.blueprint import Blueprint, make_module
from six import add_metaclass


@add_metaclass(ScopedMeta)
class ScopedDepthwiseSeparable(Module):
    def __init__(self, scope, blueprint, *_, **__):
        super(ScopedDepthwiseSeparable, self).__init__()
        self.scope = scope
        self.blueprint = blueprint

        self.conv = make_module(blueprint['conv'])
        self.convdim = make_module(blueprint['convdim'])

    def forward(self, x):
        return self.function(self.conv, self.convdim, x)

    @staticmethod
    def function(conv, convdim, x):
        out = conv(x)
        out = convdim(out)
        return out

    @staticmethod
    def describe_default(prefix='depthwise_separable', suffix='', parent=None,
                         input_shape=None, in_channels=3, out_channels=3,
                         kernel_size=3, stride=1, padding=1,
                         dilation=1, groups=-1, bias=False,
                         conv_module=ScopedConv2d,
                         conv_kwargs=None, *_, **__):
        """Describe depthwise separable conv"""
        # skip if this is just a 1x1 conv
        if kernel_size == 1:
            return conv_module.describe_default(prefix, suffix, parent,
                                                input_shape, in_channels, out_channels,
                                                kernel_size=kernel_size,
                                                stride=stride, padding=padding,
                                                dilation=dilation, groups=groups,
                                                bias=bias, conv_kwargs=conv_kwargs)
        if groups == -1:
            groups = in_channels

        kwargs = {'in_channels': in_channels,
                  'out_channels': out_channels,
                  'kernel_size': kernel_size, 'stride': stride,
                  'padding': padding, 'dilation': dilation,
                  'groups': groups, 'bias': bias}

        suffix = "%s_%d_%d_%d_%d_%d_%d_%d_%d" % (suffix, in_channels,
                                                 out_channels,
                                                 kernel_size, stride,
                                                 kernel_size, dilation,
                                                 groups, bias)

        bp = Blueprint(prefix, suffix, parent, True,
                       ScopedDepthwiseSeparable, kwargs=kwargs)

        # depthwise kxk conv
        bp['conv'] = conv_module.describe_default('%s/conv' % prefix, suffix,
                                                  bp, input_shape, in_channels,
                                                  in_channels, kernel_size,
                                                  stride, padding, dilation,
                                                  groups, bias, conv_kwargs=conv_kwargs)

        out_shape = bp['conv']['output_shape']

        # pointwise 1x1 conv
        bp['convdim'] = conv_module.describe_default('%s/convdim' % prefix, suffix,
                                                     bp, out_shape, in_channels,
                                                     out_channels, 1, 1, 0, dilation,
                                                     1, bias, conv_kwargs=conv_kwargs)
        bp['input_shape'] = input_shape
        bp['output_shape'] = bp['convdim']['output_shape']
        bp['kwargs']['blueprint'] = bp

        return bp

    @staticmethod
    def describe_from_blueprint(prefix, suffix, blueprint, parent,
                                kernel_size=None, stride=None, padding=None,
                                dilation=None, groups=None, bias=None,
                                conv_kwargs=None, ):
        input_shape = blueprint['input_shape']
        output_shape = blueprint['output_shape']
        kwargs = blueprint['kwargs']
        args = ScopedConv2d.adjust_args(kwargs, kernel_size, stride,
                                        padding, dilation, groups, bias)

        _kernel, _stride, _padding, _dilation, _groups, _bias = args
        return ScopedDepthwiseSeparable.describe_default(prefix=prefix,
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
