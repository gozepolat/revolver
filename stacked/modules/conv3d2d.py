# -*- coding: utf-8 -*-
from torch.nn import Module, Conv3d


class Conv3d2d(Module):
    r""" Conv3d that can work on 4d input as well (more compatible with 2d convolutions)

    When the input dimension == 4, converts the input into 5d input that can work with Conv3d
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, groups=1):
        super(Conv3d2d, self).__init__()
        self.ni = in_channels
        self.no = out_channels
        self.conv = Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups)

    def forward(self, x):
        if x.dim() == 4:
            # convert x into a single channel 3d input
            x = x.unsqueeze(0).permute(1, 0, 2, 3, 4)

            if self.ni > 1:
                # divide the 2d channels
                assert(x.size(2) % self.ni == 0)
                x = x.view(x.size(0), x.size(1) * self.ni, x.size(2) // self.ni, x.size(3), x.size(4))

            x = self.conv(x)

            if self.no > 1:
                # merge the 3d channels
                assert(x.size(1) % self.no == 0)
                x = x.view(x.size(0), x.size(1) // self.no, x.size(2) * self.no, x.size(3), x.size(4))

            return x.permute(1, 0, 2, 3, 4).squeeze(0)

        # default conv3d
        return self.conv(x)
