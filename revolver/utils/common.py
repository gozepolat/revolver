# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import Parameter
from logging import warning
import math
import subprocess

DEBUG_DOMAIN = True
DEBUG_SCOPE = True
DEBUG_SCOPED_RESNET = True
DEBUG_HEURISTICS = True
DEBUG_BLUEPRINT = True
DEBUG_POPULATION = True
DEBUG_OPTIMIZER = True
DEBUG_OPTIMIZER_VERBOSE = False
DEBUG_ENGINE = True
DEBUG_MEANMODEL = True
DEBUG_CONV = True
DEBUG_CONVDECONV = True
DEBUG_COMMON = True
LAST_DEBUG_MESSAGE = ""
DEBUG_MODEL_DIAGNOSTIC = True

BLUEPRINT_GUI = True
GUI = None

SCOPE_DICTIONARY = dict()
AGGRESSIVELY_SHARE = True
CURRENT_EPOCH = 1

PREVIOUS_PARAMETERS = dict()
CURRENT_FEATURES = dict()
FEATURE_DEPTHS = dict()
FEATURE_DEPTH_CTR = 0
PREVIOUS_FEATURES = dict()
PREVIOUS_LABEL = None
CURRENT_LABEL = None
TRAIN = False

POPULATION_COST_ESTIMATION_SCALE = .25
POPULATION_COST_NUM_PARAMETER_SCALE = .0075
POPULATION_COST_SEED_INDIVIDUAL_SCALE = .25
POPULATION_GENOTYPE_COST_COEFFICIENT = 96.
POPULATION_MIN_GENOTYPE_COST_COEFFICIENT = 1e-3
POPULATION_MUTATION_COEFFICIENT = .8
POPULATION_NUM_BATCH_PER_INDIVIDUAL = 96
POPULATION_AVERAGE_SCORE = 1e7  # will be updated after the first iteration
MINIMUM_LEARNING_RATE = 0.00075
UNIQUENESS_TOGGLE_P = .2
POPULATION_CROSSOVER_COEFFICIENT = .95
POPULATION_IMMIGRATION_P = .05
POPULATION_CLEANUP_P = .8
POPULATION_RANDOM_PICK_P = .2
POPULATION_TOP_VALIDATION_SCORE = 1.e10
POPULATION_FOCUS_PICK_RATIO = 1.
POPULATION_RANDOM_SEARCH_ITERATIONS = 30
POPULATION_COMPONENT_SCORES_DICT = {}
YES_SET = {"y", "Y", "yes", "YES", "Yes", "true", "True", "TRUE", True}
NO_SET = {"n", "N", "no", "NO", "No", "None", "false", "False", "FALSE", False}


def log(log_func, msg):
    if DEBUG_COMMON:
        log_func("revolver.utils.common: %s" % msg)


def gcd(a, b):
    while b > 0:
        a, b = b, a % b

    return a


def get_process_output(cmd_array):
    log(warning, ' '.join(cmd_array))
    out = subprocess.check_output(cmd_array)
    return out


def get_gpu_memory_info():
    """Return the info in the form of {gpu_id: (used, total), ...}"""
    cmd = ['nvidia-smi', '--query-gpu=index,memory.used,memory.total',
           '--format=csv,nounits,noheader']

    out = subprocess.check_output(cmd)
    lines = out.decode("utf-8").strip().split('\n')
    gpu_memory_usage = {}

    for line in lines:
        (index, memory_used, memory_total) = (int(x) for x in line.split(','))
        gpu_memory_usage[index] = (memory_used, memory_total)

    return gpu_memory_usage


def get_cuda(param, _type='float'):
    if torch.cuda.is_available():
        return getattr(param.cuda(), _type)()
    return param


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
    """Return a shallow copy of the dictionary with the key changed"""
    if not isinstance(container, dict) or key not in container:
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


def time_to_drop(train_mode, drop_p):
    if train_mode and np.random.random() < drop_p:
        return True
    return False
