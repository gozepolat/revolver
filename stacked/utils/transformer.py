# -*- coding: utf-8 -*-
from torchvision import transforms as T
import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
from torch.nn import Parameter
from common import get_cuda
import torchnet as tnt
from six import string_types
from math import ceil


def type_to_str(cls):
    """Convert the given type into a path like string form

    e.g. given the type <class 'nn.conv.Conv2d'>
    this will return: "class/nn/conv/Conv2d"
    """
    str_cls = str(cls)
    chars = []
    for c in str_cls:
        if c == "\'" or c == '<' or c == '>':
            continue
        if c == '.' or c == ' ':
            chars.append('/')
        else:
            chars.append(c)
    return ''.join(chars)


def normalize_index(index, length, mapper=None):
    """ Return the normalized index given an integer index

    e.g. num = 5 would behave similar to np.linspace(0.0, 1.0, 5)
    index:            0   1    2    3    4
    normalized index: 0.0 0.25 0.50 0.75 1.0   # =
    """
    if mapper is not None:
        return mapper.forward(index, length)
    assert (length > 0)
    if length == 1:
        return 0.0
    return float(index) / (length - 1)


def denormalize_index(normalized_float_index, length, mapper=None):
    """ Return the integer index, given a normalized index """
    if mapper is not None:
        return mapper.backward(normalized_float_index, length)
    return int(ceil(normalized_float_index * (length - 1)))


def normalize_float(value, low, high, mapper=None):
    """ Return the normalized value, given an interval

    e.g. low = 2, high = 6
    float value:            2.0   3.0   4.0  5.0   6.0
    normalized_float value: 0.0   0.25  0.5  0.75  1.0
    """
    if mapper is not None:
        return mapper.forward(value, low, high)
    assert (high > low and high >= value >= low)
    width = (high - low)
    return float(value - low) / width


def denormalize_float(normalized_float, low, high, mapper=None):
    """ Return the real value in an interval, given the normalized one """
    if mapper is None:
        value = normalized_float * (high - low) + low
    else:
        value = mapper.backward(normalized_float, low, high)
    assert (high > low and high >= value >= low)
    return value


def image_to_numpy(image):
    if isinstance(image, string_types):
        image = Image.open(image)
    return np.asarray(image)


def image_numpy_to_unsqueezed_cuda_tensor(data):
    return torch.unsqueeze(get_cuda(torch.from_numpy(data)), 0)


def image_to_unsqueezed_cuda_variable(image, requires_grad=False):
    if isinstance(image, string_types) or isinstance(image, Image.Image):
        image = image_to_numpy(image)
    if isinstance(image, np.ndarray):
        image = np.rollaxis(image, 2, 0)
        image = image_numpy_to_unsqueezed_cuda_tensor(image)
    return Variable(image, requires_grad=requires_grad)


def image_to_variable(image):
    if isinstance(image, string_types):
        image = Image.open(image)
    return T.ToTensor()(image)


def variable_to_image(tensor):
    return T.ToPILImage()(tensor)


def normalize(image, mean=None, std=None):
    if isinstance(image, string_types):
        image = image_to_variable(image)
    if mean is None:
        image = image - torch.min(image).expand_as(image)
        image = image / torch.max(image).expand_as(image)
        return image
    return T.Normalize(mean, std)(image)


def scalar_to_cuda_parameter(value, size, requires_grad=False, dtype=torch.FloatTensor):
    return Parameter(dtype([value]).cuda(),
                     requires_grad=requires_grad).expand(size)


def get_transformer(dataset="CIFAR10"):
    if dataset == "CIFAR10":
        transformer = T.Compose([
            T.ToTensor(),
            T.Normalize([125.3, 123.0, 113.9], [63.0, 62.1, 66.7]),
        ])
    elif dataset == "CIFAR100":
        transformer = tnt.transform.compose([
            T.ToTensor(),
            T.Normalize([129.3, 124.1, 112.4], [68.2, 65.4, 70.4]),
        ])
    else:
        raise NotImplementedError("The dataset {} is not supported".format(dataset))
    return transformer
