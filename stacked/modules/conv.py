# -*- coding: utf-8 -*-
from torch.nn import Module, Conv3d
from math import floor


def _repeat_in_array(value, length):
    """Repeat int value in an array"""
    if not isinstance(value, int):
        return value
    return [value] * length


def get_conv_out_res(x, kernel_size, stride, padding, dilation):
    """Given [D], H, W, get [D_out], H_out, W_out after conv"""
    length = len(x)
    kernel_size = _repeat_in_array(kernel_size, length)
    stride = _repeat_in_array(stride, length)
    padding = _repeat_in_array(padding, length)
    dilation = _repeat_in_array(dilation, length)
    return tuple(floor((x_in + 2 * padding[i] - dilation[i] *
                        (kernel_size[i] - 1) - 1) / stride[i] + 1)
                 for i, x_in in enumerate(x))


def get_conv_out_shape(input_shape, c_out, kernel_size=3,
                       stride=1, padding=1, dilation=1):
    """Given input shape and conv arguments, get the output shape"""
    if input_shape is None:
        return None
    x = input_shape[2:]
    x_out = get_conv_out_res(x, kernel_size, stride, padding, dilation)
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
                assert(x.size(2) % self.ni == 0)
                x = x.view(x.size(0),
                           x.size(1) * self.ni,
                           x.size(2) // self.ni,
                           x.size(3), x.size(4))

            x = self.conv(x)

            if self.no > 1:
                # merge the 3d channels
                assert(x.size(1) % self.no == 0)
                x = x.view(x.size(0),
                           x.size(1) // self.no,
                           x.size(2) * self.no,
                           x.size(3), x.size(4))

            return x.permute(1, 0, 2, 3, 4).squeeze(0)

        # default conv3d
        return self.conv(x)
