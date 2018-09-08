# -*- coding: utf-8 -*-
from torch.nn import Module, Conv3d, Conv2d, ConvTranspose2d
import torch.nn.functional as F
import torch
from stacked.utils import common
from logging import warning


def log(log_func, msg):
    if common.DEBUG_CONV:
        log_func("stacked.modules.conv: %s" % msg)


def _repeat_in_array(value, length):
    """Repeat int value in an array"""
    if not isinstance(value, int):
        return value
    return [value] * length


def _adjust_repeated_args(x, kernel_size, stride, padding, dilation):
    length = len(x)
    kernel_size = _repeat_in_array(kernel_size, length)
    stride = _repeat_in_array(stride, length)
    padding = _repeat_in_array(padding, length)
    dilation = _repeat_in_array(dilation, length)
    return kernel_size, stride, padding, dilation


def get_conv_out_res(x, kernel_size, stride, padding, dilation):
    """Given [D], H, W, get [D_out], H_out, W_out after conv"""
    args = _adjust_repeated_args(x, kernel_size, stride, padding, dilation)
    kernel_size, stride, padding, dilation = args
    return tuple(int((x_in + 2 * padding[i] - dilation[i] *
                      (kernel_size[i] - 1) - 1) / stride[i] + 1)
                 for i, x_in in enumerate(x))


def get_deconv_out_res(x, kernel_size, stride, padding, dilation):
    """Given [D], H, W, get [D_out], H_out, W_out after deconv"""
    args = _adjust_repeated_args(x, kernel_size, stride, padding, dilation)
    kernel_size, stride, padding, dilation = args
    return tuple(int((x_in - 1) * stride[i] - 2 * padding[i]
                     + kernel_size[i] * padding[i]) for i, x_in in enumerate(x))


def get_conv_out_shape(input_shape, c_out, kernel_size=3,
                       stride=1, padding=1, dilation=1):
    """Given input shape and conv arguments, get the output shape"""
    if input_shape is None:
        return None

    x = input_shape[2:]
    x_out = get_conv_out_res(x, kernel_size, stride, padding, dilation)

    return (input_shape[0], c_out) + x_out


def get_deconv_out_shape(input_shape, c_out, kernel_size=3,
                         stride=1, padding=1, dilation=1):
    """Given input shape and deconv arguments, get the output shape"""
    if input_shape is None:
        return None

    x = input_shape[2:]
    x_out = get_deconv_out_res(x, kernel_size, stride, padding, dilation)

    return (input_shape[0], c_out) + x_out


class Conv3d2d(Module):
    r""" Conv3d with 4d input mode

    When the input dimension == 4, converts the input into 5d and outputs 4d
    """

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(Conv3d2d, self).__init__()
        self.ni = in_channels
        self.no = out_channels
        self.conv = Conv3d(in_channels, out_channels, kernel_size,
                           stride, padding, dilation, groups, bias)

    def forward(self, x):
        if x.dim() == 4:
            # convert x into 5d input with depth channels = 1
            x = x.unsqueeze(0).permute(1, 0, 2, 3, 4)

            if self.ni > 1:
                # divide the original in_channels to increase the depth
                assert (x.size(2) % self.ni == 0)
                x = x.view(x.size(0),
                           x.size(1) * self.ni,
                           x.size(2) // self.ni,
                           x.size(3), x.size(4))

            x = self.conv(x)

            if self.no > 1:
                # merge the 3d channels
                assert (x.size(1) % self.no == 0)
                x = x.view(x.size(0),
                           x.size(1) // self.no,
                           x.size(2) * self.no,
                           x.size(3), x.size(4))

            return x.permute(1, 0, 2, 3, 4).squeeze(0)

        # default conv3d
        return self.conv(x)


class Conv2dDeconv2dConcat(Module):
    """Regular conv2d concatenated with transposed conv2d (deconv) output"""

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=0, dilation=1, groups=1, bias=False, *_, **__):
        super(Conv2dDeconv2dConcat, self).__init__()

        self.downsample = None
        self.convdim = None
        self.deconv = None
        self.vanilla_conv = None

        self.bias = bias
        self.dilation = dilation
        self.groups = groups
        conv_padding = kernel_size // 2  # input_size[2,3] == output_size[2,3]
        self.conv_padding = conv_padding

        if kernel_size == 1:
            self.vanilla_conv = Conv2d(in_channels, out_channels, kernel_size,
                                       stride=stride, padding=padding, dilation=dilation,
                                       groups=groups, bias=bias)
            return

        if stride > 1 or padding != kernel_size // 2:
            log(warning, "Conv2dDeconv2dConcat.__init__: downsampling k: %d, s: %d, p: %d" %
                (kernel_size, stride, padding))
            self.downsample = Conv2d(in_channels, in_channels, kernel_size,
                                     stride=stride, padding=padding, dilation=dilation,
                                     groups=in_channels, bias=bias)
        if out_channels % 2 != 0:
            log(warning, "Conv2dDeconv2dConcat.__init__: odd out_channels: %d" % out_channels)
            self.convdim = Conv2d(out_channels + 1, out_channels, 1, stride=1,
                                  padding=0, dilation=1, groups=groups, bias=False)
            out_channels += 1

        self.deconv = ConvTranspose2d(in_channels, out_channels // 2, kernel_size,
                                      stride=1, padding=conv_padding, dilation=dilation,
                                      groups=groups, bias=bias)

    def forward(self, x):
        if self.vanilla_conv is not None:
            o = self.vanilla_conv(x)
            return o

        if self.downsample is not None:
            x = self.downsample(x)

        o1 = F.conv2d(x, self.deconv.weight.transpose(0, 1), self.deconv.bias, 1,
                      self.conv_padding, self.dilation, self.groups)
        o2 = self.deconv(x, output_size=o1.size())
        o = torch.cat((o1, o2), dim=1)

        if self.convdim is None:
            return o

        o = self.convdim(o)
        return o
