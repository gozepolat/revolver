# -*- coding: utf-8 -*-
from torch.nn import Module, Conv3d


class Conv3d2d(Module):
    r""" Conv3d that can work on 2d input as well

    When the input dimension == 4, converts the input into a single channel 3d input (dimension 5)
    For such operations assumes ni == no == 1 (where ni = #input channels, no = #output channels)
    """
    def __init__(self, ni, no, kernel_size=3, stride=1, padding=0, dilation=1, groups=1):
        super(Conv3d2d, self).__init__()
        self.ni = ni
        self.no = no
        self.conv = Conv3d(ni, no, kernel_size, stride, padding, dilation, groups)

    def forward(self, x):
        if x.dim() == 4:  # convert x into a single channel 3d input
            x = x.unsqueeze(0).permute(1, 0, 2, 3, 4)
            assert(self.ni == 1 and self.no == 1)
            x = self.conv(x)
            return x.permute(1, 0, 2, 3, 4).squeeze(0)

        # default conv3d
        return self.conv(x)
