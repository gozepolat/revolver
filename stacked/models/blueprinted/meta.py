# -*- coding: utf-8 -*-
import torch
from torch.nn import Module, Parameter
from torch.nn.init import kaiming_normal
from stacked.modules.scoped_nn import ScopedConv2d, \
    ScopedBatchNorm2d, ScopedReLU, ScopedConv3d2d
from stacked.meta.scope import ScopedMeta
from stacked.models.blueprinted.resblock import ScopedResBlock
from stacked.meta.blueprint import Blueprint, make_module
from stacked.utils.transformer import get_cuda
from stacked.modules.fakes import MaskMultiplied, MaskScalarMultipliedSummed
from six import add_metaclass


@add_metaclass(ScopedMeta)
class PreConvMask(Module):
    def __init__(self, scope, blueprint, *_,  **__):
        super(PreConvMask, self).__init__()
        self.scope = scope
        self.scalar = Parameter(torch.FloatTensor([blueprint['scalar']]).cuda(),
                                requires_grad=True)

    def forward(self, module_out, mask):
        return self.function(module_out, mask, self.scalar)

    @staticmethod
    def function(module_out, mask, scalar):
        return module_out * scalar + mask


@add_metaclass(ScopedMeta)
class ScopedMetaMaskGenerator(Module):
    def __init__(self, scope, blueprint, *_, **__):
        super(ScopedMetaMaskGenerator, self).__init__()
        self.scope = scope
        self.conv = make_module(blueprint['conv'])
        self.mask = get_cuda(
            kaiming_normal(torch.ones(blueprint['input_shape'])))
        self.pre_conv = make_module(blueprint['pre_conv'])

    def forward(self, x):
        mask = self.function(x, self.conv, self.pre_conv, self.mask)
        self.mask = mask.data
        return mask

    @staticmethod
    def function(x, conv, pre_conv, mask):
        return conv(pre_conv(x, mask))

    @staticmethod
    def describe_default(prefix='gen', suffix='', parent=None,
                         shape=None,
                         bn_module=ScopedBatchNorm2d,
                         act_module=ScopedReLU,
                         conv_module=ScopedConv3d2d,
                         in_channels=3, out_channels=3,
                         kernel_size=7, stride=1,
                         padding=2, dilation=1, groups=1,
                         bias=True, pre_conv=MaskScalarMultipliedSummed,
                         **__):

        bp = Blueprint(prefix, suffix, parent, False,
                       ScopedMetaMaskGenerator)

        depth = 2
        bp['conv'] = ScopedResBlock.describe_default('%s/conv' % prefix,
                                                     suffix, bp, depth,
                                                     conv_module, bn_module,
                                                     act_module, in_channels,
                                                     out_channels, kernel_size,
                                                     stride, padding, shape,
                                                     dilation, groups, bias)
        bp['pre_conv'] = pre_conv
        bp['input_shape'] = shape
        bp['output_shape'] = shape
        assert(bp['conv']['output_shape'] == shape)
        return bp


@add_metaclass(ScopedMeta)
class ScopedMetaMasked(Module):
    def __init__(self, scope, blueprint, *_, **__):
        super(ScopedMeta, self).__init__()
        self.scope = scope
        self.generator = make_module(blueprint['generator'])
        self.conv = make_module(blueprint['conv'])
        self.mask_fn = make_module(blueprint['mask_fn'])

    def forward(self, x):
        return self.function(x, self.mask_fn,
                             self.generator, self.conv)

    @staticmethod
    def function(x, mask_fn, generator, conv):
        out = conv(x)
        return mask_fn(x, out, generator(out))

    @staticmethod
    def describe_default(prefix='meta_layer', suffix='', parent=None,
                         shape=None, in_channels=3, out_channels=3,
                         kernel_size=3, stride=1, padding=1,
                         dilation=1, groups=1, bias=True,
                         conv_module=ScopedConv2d,
                         generator=ScopedMetaMaskGenerator,
                         mask_fn=MaskMultiplied,
                         gen_bn_module=ScopedBatchNorm2d,
                         gen_act_module=ScopedReLU,
                         gen_conv=ScopedConv3d2d,
                         gen_in_channels=2, gen_out_channels=2,
                         gen_kernel_size=7, gen_stride=1,
                         gen_dilation=1, gen_groups=1,
                         gen_bias=True,
                         gen_pre_conv=PreConvMask,
                         **__):
        """Meta masks to model local rules"""

        kwargs = {'in_channels': in_channels,
                  'out_channels': out_channels,
                  'kernel_size': kernel_size, 'stride': stride,
                  'padding': padding, 'dilation': dilation,
                  'groups': groups, 'bias': bias}

        bp = Blueprint(prefix, suffix, parent, False,
                       ScopedMetaMasked, kwargs=kwargs)

        bp['mask_fn'] = Blueprint('%s/mask_fn' % prefix, suffix, bp,
                                  False, mask_fn)

        bp['conv'] = conv_module.describe_default('%s/conv' % prefix, suffix,
                                                  bp, shape, in_channels,
                                                  out_channels, kernel_size,
                                                  stride, padding, dilation,
                                                  groups, bias)
        out_shape = bp['conv']['output_shape']
        bp['generator'] = generator.describe_default('%s/gen' % prefix, suffix,
                                                     bp, out_shape, gen_bn_module,
                                                     gen_act_module, gen_conv,
                                                     gen_in_channels,
                                                     gen_out_channels, gen_kernel_size,
                                                     gen_stride, gen_kernel_size // 2,
                                                     gen_dilation, gen_groups, gen_bias,
                                                     gen_pre_conv)

        assert (shape is not None)
        bp['input_shape'] = shape
        bp['output_shape'] = out_shape
        return bp

