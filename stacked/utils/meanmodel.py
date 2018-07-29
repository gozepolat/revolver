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


def average_model(model, target):
    """Average the parameters of the same shape"""
    target_dict = target.state_dict()
    shape_dict = get_shape_dict(model)
    average_dict = get_average_dict(shape_dict)
    for shape, (avg, exclude_keys) in average_dict.items():
        for k, _ in shape_dict[shape]:
            if k not in exclude_keys:
                if k in target_dict:
                    log(warning, "setting %s with avg" % k)
                    target_dict[k].copy_(avg.data)






