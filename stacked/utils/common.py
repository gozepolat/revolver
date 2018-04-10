# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import Parameter
import math


DEBUG_DOMAIN = True
DEBUG_SCOPE = True
DEBUG_SCOPED_RESNET = True
SCOPE_DICTIONARY = dict()


def get_cuda(param, dtype='float'):
    return getattr(param.cuda(), dtype)()


def imshow(img, duration=0.001):
    plt.imshow(np.rollaxis(img, 0, 3))
    plt.pause(duration)


def imsave(path, img):
    plt.imsave(path, np.rollaxis(img, 0, 3))


def make_weights(num_input_filters,
                 num_out_filters,
                 kernel_width,
                 kernel_height, requires_grad=True):
    dtype = Variable
    if requires_grad:
        dtype = Parameter

    fan_in = math.sqrt(num_input_filters * kernel_width * kernel_height)
    weights = torch.Tensor(num_out_filters,
                           num_input_filters,
                           kernel_width,
                           kernel_height).normal_(0, 2 / fan_in)
    return dtype(get_cuda(weights), requires_grad=requires_grad)
