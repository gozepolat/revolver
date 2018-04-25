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
BLUEPRINT_GUI = True
GUI = None

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


def replace_key(container, key, value):
    if not isinstance(container, dict):
        return container
    new_dict = {k: v for k, v in container.items()}
    new_dict[key] = value
    return new_dict


def get_same_value_indices(container, key, ix=0):
    """Collect indices where value is the same for the given key"""
    indices = {str(c[key]): [] for c in container[ix:]}
    for i, c in enumerate(container[ix:]):
        indices[str(c[key])].append(i + ix)
    return indices


def swap_elements(container1, container2, key1, key2):
    tmp = container1[key1]
    container1[key1] = container2[key2]
    container2[key2] = tmp
