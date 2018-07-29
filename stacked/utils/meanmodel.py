# -*- coding: utf-8 -*-
from stacked.utils import common
from logging import warning


def log(log_func, msg):
    if common.DEBUG_MEANMODEL:
        log_func("stacked.utils.meanmodel: %s" % msg)


def get_shape_dict(model):
    """Cluster model parameters according ot shape"""
    shape_dict = {}
    for k, w in model.named_parameters():
        shape = str(w.size())

        if shape in shape_dict:
            shape_dict[shape].append((k, w))
        else:
            shape_dict[shape] = [(k, w)]

    return shape_dict


def average_weights(group, exclude=lambda k: False):
    """Get the average weight, optionally excluding certain keys"""
    avg = 0.0
    exclude_keys = set()
    for k, w in group:
        if exclude(k):
            exclude_keys.add(k)
            continue
        avg = w + avg

    return avg / len(group), exclude_keys


def get_average_dict(shape_dict):
    """Get average weights for distinct shapes"""
    average_dict = {}
    for shape, group in shape_dict.items():
        average_dict[shape] = average_weights(group)
    return average_dict


def get_named_parameter(module, key):
    """Get the container parameter for the last key

    Args:
        module: Main container
        key: String in the form of k1.k2...kn

    :return immediate container (e.g. module.k1.k2..kn-1), last key (e.g. kn)
    """
    elements = key.split('.')
    last = elements.pop()
    for e in elements:
        module = getattr(module, e)
    return module, last


def set_named_parameter(module, key, value):
    """Set the parameter for the given key with the given value"""
    ref, last = get_named_parameter(module, key)
    setattr(ref, key, value)


def average_model(model, target=None):
    """Average the parameters of the same shape"""
    if target is None:  # in place avg
        target = model

    shape_dict = get_shape_dict(model)
    average_dict = get_average_dict(shape_dict)
    for shape, (avg, exclude_keys) in average_dict.items():
        for k, _ in shape_dict[shape]:
            if k not in exclude_keys:
                log(warning, "setting %s with avg" % k)
                set_named_parameter(target, k, avg)






