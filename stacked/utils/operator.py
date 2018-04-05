# -*- coding: utf-8 -*-
from torch.nn import Parameter
from torch.autograd import Variable
import numpy as np
from common import get_cuda
import torch


def get_np_operator(name, in_channels, out_channels):
    if name == 'laplace':
        return np.array([[[[0.0, 1.0, 0.0]
                         , [1.0, -4.0, 1.0]
                         , [0.0, 1.0, 0.0]]]
                         * in_channels] * out_channels)
    if name == 'sobel_x':
        return np.array([[[[1.0, 0.0, -1.0]
                         , [2.0, 0.0, -2.0]
                         , [1.0, 0.0, -1.0]]]
                         * in_channels] * out_channels)
    if name == 'sobel_y':
        return np.array([[[[1.0, 2.0, 1.0]
                         , [0.0, 0.0, 0.0]
                         , [-1.0, -2.0, -1.0]]]
                         * in_channels] * out_channels)
    if name == 'scharr_x':
        return np.array([[[[3.0, 0.0, -3.0]
                         , [10.0, 0.0, -10.0]
                         , [3.0, 0.0, -3.0]]]
                         * in_channels] * out_channels)
    if name == 'scharr_y':
        return np.array([[[[3.0, 10.0, 3.0]
                         , [0.0, 0.0, 0.0]
                         , [-3.0, -10.0, -3.0]]]
                         * in_channels] * out_channels)
    raise NotImplementedError("Name %s not found" % name)


def make_operator(name, in_channels=3, out_channels=3, requires_grad=False):
    """Create a gradient operator for edge detection and image regularization tasks

    :param name: laplace, scharr_x, or scharr_y
    :param in_channels: e.g. 3 for an RGB image
    :param out_channels: usually the same as the # input channels
    :param requires_grad: learnable operator
    :return: convolutional kernel initialized with one of the operator values
    """
    dtype = Variable
    if requires_grad:
        dtype = Parameter
    operator = get_np_operator(name, in_channels, out_channels)
    return dtype(get_cuda(torch.from_numpy(operator)), requires_grad=requires_grad)
