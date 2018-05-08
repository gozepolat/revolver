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
DEBUG_HEURISTICS = True
DEBUG_BLUEPRINT = True
DEBUG_POPULATION = True
BLUEPRINT_GUI = True
GUI = None

SCOPE_DICTIONARY = dict()


def get_cuda(param, _type='float'):
    return getattr(param.cuda(), _type)()


def imshow(img, duration=0.001):
    plt.imshow(np.rollaxis(img, 0, 3))
    plt.pause(duration)


def imsave(path, img):
    plt.imsave(path, np.rollaxis(img, 0, 3))


def make_weights(num_input_filters,
                 num_out_filters,
                 kernel_width,
                 kernel_height, requires_grad=True):

    _type = Variable
    if requires_grad:
        _type = Parameter

    fan_in = math.sqrt(num_input_filters * kernel_width
                       * kernel_height)
    weights = torch.Tensor(num_out_filters,
                           num_input_filters,
                           kernel_width,
                           kernel_height).normal_(0, 2 / fan_in)

    return _type(get_cuda(weights), requires_grad=requires_grad)


def replace_key(container, key, value):
    if not isinstance(container, dict):
        return container
    new_dict = {k: v for k, v in container.items()}
    new_dict[key] = value
    return new_dict


def get_same_value_indices(container, key, ix=0):
    """Collect indices where value is the same for the given key"""
    indices = {str(c[key]): [] for c in container[ix:]
               if c[key] is not None}
    for i, c in enumerate(container[ix:]):
        indices[str(c[key])].append(i + ix)
    return indices


def swap_consecutive(container1, container2, index1, index2,
                     ix1=None, ix2=None):
    """Swap all the elements after the entry point (index1, index2)"""
    tmp = [k for k in container1]
    container1[index1:ix1] = container2[index2:ix2]
    container2[index2:ix2] = tmp[index1:ix1]

