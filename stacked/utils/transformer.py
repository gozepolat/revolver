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
