# -*- coding: utf-8 -*-
from torchvision import transforms as T
import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
from torch.nn import Parameter
from stacked.utils.common import get_cuda
from six import string_types
from math import ceil


def all_to_none(*_, **__):
    """Ignore everything and return None"""
    return None


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
    normalized index: 0.0 0.25 0.50 0.75 1.0
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
    assert high >= low and high >= value >= low, f"It is supposed to be {low} <= {value} <= {high}"
    width = (high - low)
    if width > 0:
        return float(value - low) / width
    return 0.0


def denormalize_float(normalized_float, low, high, mapper=None):
    """ Return the real value in an interval, given the normalized one """
    if mapper is None:
        value = normalized_float * (high - low) + low
    else:
        value = mapper.backward(normalized_float, low, high)
    assert high > low and high >= value >= low, f"It is supposed to be {low} <= {value} <= {high}"
    return value


def image_to_numpy(image):
    if isinstance(image, string_types):
        image = Image.open(image)
    return np.asarray(image)


def image_numpy_to_unsqueezed_cuda_tensor(data):
    return T.functional.to_tensor(data).reshape((1, *data.shape))
    #if torch.cuda.is_available():
    #    return torch.unsqueeze(get_cuda(torch.from_numpy(data.copy())), 0)
    #return torch.unsqueeze(torch.from_numpy(data.copy()), 0)


def image_to_unsqueezed_cuda_variable(image, requires_grad=False):
    if isinstance(image, string_types) or isinstance(image, Image.Image):
        image = image_to_numpy(image)
    if isinstance(image, np.ndarray):
        if image.ndim > 2:
            image = np.rollaxis(image, 2, 0)
        else:
            image = image.reshape((1, *image.shape))
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


def softmax(x):
    x = np.exp(x - np.max(x))
    return x / x.sum(axis=0)


def scalar_to_tensor(value, size):
    return torch.FloatTensor([float(value)]).expand(size)


def scalar_to_cuda_parameter(value, size, requires_grad=False,
                             dtype=torch.FloatTensor):
    return Parameter(dtype([value]).cuda(),
                     requires_grad=requires_grad).expand(size)


def list_to_pairs(iterable, condition=lambda x, y: x != y):
    """"""
    pairs = []
    length = len(iterable)
    for i, p1 in enumerate(iterable):
        for j in range(length - i):
            if condition(i, j):
                pairs.append((p1, iterable[j]))
    return pairs


def get_transformer(dataset="CIFAR10"):
    if dataset == "CIFAR10":
        transformer = T.Compose([
            T.ToTensor(),
            T.Normalize(np.array([125.3, 123.0, 113.9]) / 255.0,
                        np.array([63.0, 62.1, 66.7]) / 255.0),
        ])
    elif dataset == "CIFAR100":
        transformer = T.Compose([
            T.ToTensor(),
            T.Normalize(np.array([129.3, 124.1, 112.4]) / 255.0,
                        np.array([68.2, 65.4, 70.4]) / 255.0),
        ])
    elif dataset == "MNIST":
        transformer = T.Compose([
            T.ToTensor(),
            T.Normalize((0.1307,), (0.3081,))
        ])
    elif dataset == 'SVHN':
        transformer = T.Compose([
            T.ToTensor(),
            T.Normalize(np.array([109.9, 109.7, 113.8]) / 255.0,
                        np.array([50.1, 50.6, 50.8]) / 255.0),
        ])
    else:
        raise NotImplementedError(
            "The dataset {} is not supported".format(dataset))
    return transformer
