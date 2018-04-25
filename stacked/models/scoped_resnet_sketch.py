# -*- coding: utf-8 -*-
"""An early prototype"""
import torch.nn.functional as F
from torch.nn import Module
from stacked.modules.scoped_nn import ScopedConv2d, \
    ScopedBatchNorm2d, ScopedLinear, ScopedModuleList
from stacked.meta.scope import ScopedMeta, generate_random_scope
from six import add_metaclass
from stacked.utils import common
from logging import warning


def log(log_func, msg):
    if common.DEBUG_SCOPED_RESNET:
        log_func(msg)


def has_unique_in_blocks(blueprint):
    for _, block_blueprint in blueprint['block_elements']:
        if len(block_blueprint['uniques']) > 0:
            return True
    return False


def has_unique_in_block(blueprint, scope):
    for _, block_bp in blueprint['block_elements']:
        if scope == block_bp['name']:
            if len(block_bp['uniques']) > 0:
                return True
    return False


def has_unique_in_groups(blueprint):
    for _, group_blueprint in blueprint['group_elements']:
        if len(group_blueprint['uniques']) > 0:
            return True
        if has_unique_in_blocks(group_blueprint):
            return True


def has_unique_in_group(blueprint, scope):
    for _, group_blueprint in blueprint['group_elements']:
        if scope == group_blueprint['name']:
            if len(group_blueprint['uniques']) > 0:
                return True
            if has_unique_in_blocks(group_blueprint):
                return True
            return False
    return False


def has_unique(scope, blueprint):
    r"""Check whether the scope needs to be unique"""
    if scope in blueprint['uniques']:
        return True

    if scope == 'group_container':
        if has_unique_in_groups(blueprint):
            return True
    elif scope[:5] == 'group':
        if has_unique_in_group(blueprint, scope):
            return True
    elif scope == 'block_container':
        if has_unique_in_blocks(blueprint):
            return True
    elif scope[:5] == 'block':
        if has_unique_in_block(blueprint, scope):
            return True
    return False


def suffix(key, scopes):
    return "%s%s" % (scopes[key], scopes['%s_suffix' % key])


def get_scope_and_suffix(prefix, scope, blueprint):
    r"""Generate a full scope and suffix string

    Scope suffix enables creating unique scopes,
    defaults to '' when the scope is not in unique set
    and doesn't contain unique items
    Args:
        prefix: the global scope string so far
        scope: current keyword for the scoped object
        blueprint: dictionary that contains scope keywords
        and scoped object keys in the set uniques
    """
    _suffix = generate_random_scope() if has_unique(scope, blueprint) else ''
    if scope in blueprint:
        key = blueprint[scope]
    elif 'group_elements' in blueprint:
        key = dict(blueprint['group_elements'])[scope]['name']
    elif 'block_elements' in blueprint:
        key = dict(blueprint['block_elements'])[scope]['name']
    else:
        raise KeyError
    return "%s/%s" % (prefix, key), _suffix


@add_metaclass(ScopedMeta)
class ScopedResBlock(Module):
    r"""Residual block with scope

    Args:
        scope: Scope for the self (ScopedResBlock instance)
        prefix: Scope prefix for members / objects created in the constructor
        blueprint: A dictionary which stores the scope keywords for modules
        ni (int): Number of channels in the input
        no (int): Number of channels produced by the block
        kernel_size (int or tuple): Size of the convolving kernel. Default: 3
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        conv: Convolution module to use in forward. Default: ScopedConv2d
        act: Activation function to use in forward. Default: torch.nn.functional.relu
    """
    block_depth = 2

    def __init__(self, scope, prefix, blueprint, ni, no, kernel_size=3,
                 stride=1, conv=ScopedConv2d, act=F.relu):
        super(ScopedResBlock, self).__init__()
        self.scope = scope
        self.act = act

        if blueprint is None:
            log(warning, "Block %s has no blueprint, making a default one" % scope)
            blueprint = self.make_blueprint(prefix)
        self.blueprint = blueprint

        scopes = self.get_scopes_and_suffixes(prefix, blueprint)

        self.conv0 = conv(suffix('conv0', scopes), ni, no, kernel_size, stride, padding=1)
        self.bn0 = ScopedBatchNorm2d(suffix('bn0', scopes), ni)
        self.conv1 = conv(suffix('conv1', scopes), no, no, kernel_size, 1, 1)
        self.bn1 = ScopedBatchNorm2d(suffix('bn1', scopes), no)

        # 1x1 conv to correct the number of channels for summation
        self.convdim = ScopedConv2d(suffix('convdim', scopes),
                                    ni, no, 1, stride=stride) if ni != no else None

    def get_member_scopes(self):
        conv0 = self.conv0.scope
        bn0 = self.bn0.scope
        conv1 = self.conv1.scope
        bn1 = self.bn1.scope
        convdim = self.convdim.scope
        return {k: v for k, v in locals().items() if k != 'self'}

    def forward(self, x):
        o1 = self.act(self.bn0(x), inplace=True)
        y = self.conv0(o1)
        o2 = self.act(self.bn1(y), inplace=True)
        z = self.conv1(o2)
        if self.convdim is not None:
            return z + self.convdim(o1)
        else:
            return z + x

    @staticmethod
    def get_scopes_and_suffixes(prefix, blueprint):
        conv0, conv0_suffix = get_scope_and_suffix(prefix, 'conv0', blueprint)
        bn0, bn0_suffix = get_scope_and_suffix(prefix, 'bn0', blueprint)
        conv1, conv1_suffix = get_scope_and_suffix(prefix, 'conv1', blueprint)
        bn1, bn1_suffix = get_scope_and_suffix(prefix, 'bn1', blueprint)
        convdim, convdim_suffix = get_scope_and_suffix(prefix, 'convdim', blueprint)

        # include any local variable name that will be in the module scopes
        labels = ['conv0', 'conv0_suffix', 'bn0', 'bn0_suffix', 'conv1', 'conv1_suffix',
                  'bn1', 'bn1_suffix', 'convdim', 'convdim_suffix']
        return {k: v for k, v in locals().items() if k in labels}

    @staticmethod
    def make_blueprint(prefix):
        return {'name': prefix, 'conv0': 'conv0', 'bn0': 'bn0', 'conv1': 'conv1',
                'bn1': 'bn1', 'convdim': 'convdim', 'uniques': set()}


@add_metaclass(ScopedMeta)
class ScopedResGroup(Module):
    r"""Scoped group of residual blocks with the same number of output channels

        Args:
        scope: Global scope for self
        blueprint: A dictionary which stores the scope keywords for modules
        ni (int): Number of channels in the input
        no (int): Number of channels produced by the block
        kernel (int or tuple): Size of the convolving kernel.
        stride (int or tuple, optional): Stride of the convolution.
        num_blocks (int): Number of residual blocks per group
        conv (torch.nn.Module): Module to use in forward. Default: Conv2d
        act: Activation function to use in forward. Default: torch.nn.functional.relu
    """

    def __init__(self, scope, prefix, blueprint, ni, no, kernel, stride, num_blocks,
                 conv=ScopedConv2d, act=F.relu):
        super(ScopedResGroup, self).__init__()
        self.scope = scope

        if blueprint is None:
            log(warning, "Group %s has no blueprint, making a default one" % scope)
            blueprint = self.make_blueprint(prefix, num_blocks)
        self.blueprint = blueprint

        scopes = self.get_scopes_and_suffixes(prefix, blueprint)
        self.block_container = ScopedModuleList(suffix('block_container', scopes))
        if len(self.block_container) == 0:
            self._add_blocks(scopes, ni, no, kernel, stride, num_blocks, conv, act)

    def _add_blocks(self, scopes, ni, no, kernel_size, stride, num_blocks, conv, act):
        blocks = scopes['block_elements']
        for i in range(num_blocks):
            scope_prefix, _suffix = blocks[i][0]
            block_scope = "%s%s" % (scope_prefix, _suffix)
            block_blueprint = blocks[i][1]
            log(warning, "block_scope: {}".format(block_scope))
            self.block_container.append(
                ScopedResBlock(block_scope, scope_prefix, block_blueprint, ni, no,
                               kernel_size, stride, conv, act))
            ni = no
            stride = 1

    def get_member_scopes(self):
        named_elements = zip(self.blueprint['block_elements'], self.block_container)
        scope = {block_name: block.scope for block_name, block in named_elements}
        scope['block_container'] = self.block_container.scope
        return scope

    def forward(self, x):
        for block in self.block_container:
            x = block(x)
        return x

    @staticmethod
    def get_scopes_and_suffixes(prefix, blueprint):
        block_container, block_container_suffix = \
            get_scope_and_suffix(prefix, 'block_container', blueprint)

        block_elements = [(get_scope_and_suffix(prefix, key, blueprint), block_blueprint)
                          for key, block_blueprint in blueprint['block_elements']]

        # include any local variable name that will be in the module scopes
        labels = ['block_container', 'block_container_suffix', 'block_elements']
        return {k: v for k, v in locals().items() if k in labels}

    @staticmethod
    def make_blueprint(prefix, num_blocks):
        bp = dict()
        bp['name'] = prefix
        bp['uniques'] = set()
        bp['block_container'] = 'block_container'
        fn = ScopedResBlock.make_blueprint
        bp['block_elements'] = [('block%d' % i, fn('block%d' % i)) for i in range(num_blocks)]
        return bp


class ResNet(Module):
    r"""WRN inspired implementation of ResNet

    Args:
        scope: Global scope for self
        blueprint: A dictionary which stores the scope keywords for modules
        depth (int): Overall depth of the network
        width (int): scalar to get the width per group
        num_classes (int): Number of categories for supervised learning
        skeleton (iterable): the smallest possible widths per group
        conv (torch.nn.Module): Module to use in forward. Default: Conv2d
        act: Activation function to use in forward. Default: torch.nn.functional.relu
    """

    def __init__(self, scope, prefix, blueprint, depth, width, num_classes, skeleton=(16, 32, 64),
                 conv=ScopedConv2d, act=F.relu):
        super(ResNet, self).__init__()
        self.scope = scope
        if prefix is None:
            prefix = scope
        self.skeleton = skeleton
        self.depth = depth
        self.num_groups = len(skeleton)
        self.num_blocks = self.get_num_blocks_per_group(depth, self.num_groups)

        if blueprint is None:
            log(warning, "ResNet %s has no blueprint, making a default one" % blueprint)
            blueprint = self.make_blueprint(prefix, self.num_groups, self.num_blocks)

        self.blueprint = blueprint

        scopes = self.get_scopes_and_suffixes(prefix, blueprint)
        self.widths = [i * width for i in skeleton]
        self.act = act
        self.conv0 = ScopedConv2d(suffix('conv0', scopes), 3, skeleton[0], 3, 1, padding=1)
        self.group_container = ScopedModuleList(suffix('group_container', scopes))
        self.bn = ScopedBatchNorm2d(suffix('bn', scopes), self.widths[2])
        self.linear = ScopedLinear(suffix('linear', scopes), self.widths[2], num_classes)

        if len(self.group_container) == 0:
            self._add_groups(scopes, conv, act)

    def _add_groups(self, scopes, conv, act):
        ni = self.skeleton[0]
        stride = 1
        k = 0
        groups = scopes['group_elements']
        for width in self.widths:
            no = width
            scope_prefix, _suffix = groups[k][0]
            group_scope = "%s%s" % (scope_prefix, _suffix)
            group_blueprint = groups[k][1]
            self.group_container.append(
                ScopedResGroup(group_scope, scope_prefix, group_blueprint, ni, no, 3,
                               stride, self.num_blocks, conv, act))
            ni = no
            stride = 2
            k += 1

    def get_member_scopes(self):
        named_elements = zip(self.blueprint['group_elements'], self.group_container)
        scope = {name: element.scope for name, element in named_elements}
        scope['group_container'] = self.group_container.scope
        scope['conv0'] = self.conv0.scope
        scope['bn'] = self.bn.scope
        scope['linear'] = self.linear.scope
        return scope

    def forward(self, x):
        x = self.conv0(x)
        for group in self.group_container:
            x = group(x)
        o = self.act(self.bn(x))
        o = F.avg_pool2d(o, o.size()[2], 1, 0)
        o = o.view(o.size(0), -1)
        o = self.linear(o)
        return o

    @staticmethod
    def get_num_blocks_per_group(depth, num_groups):
        return (depth - 4) // (num_groups * ScopedResBlock.block_depth)

    @staticmethod
    def get_scopes_and_suffixes(prefix, blueprint):
        conv0, conv0_suffix = get_scope_and_suffix(prefix, 'conv0', blueprint)
        bn, bn_suffix = get_scope_and_suffix(prefix, 'bn', blueprint)
        linear, linear_suffix = get_scope_and_suffix(prefix, 'linear', blueprint)
        group_container, group_container_suffix = \
            get_scope_and_suffix(prefix, 'group_container', blueprint)

        group_elements = [(get_scope_and_suffix(prefix, key, blueprint), group_blueprint)
                          for key, group_blueprint in blueprint['group_elements']]
        # include any local variable name that will be in the module scopes
        labels = ['conv0', 'conv0_suffix', 'bn', 'bn_suffix', 'linear', 'linear_suffix',
                  'group_container', 'group_container_suffix', 'group_elements']
        return {k: v for k, v in locals().items() if k in labels}

    @staticmethod
    def make_blueprint(prefix, num_groups, num_blocks):
        bp = dict()
        bp['name'] = prefix
        bp['conv0'] = 'conv0'
        bp['uniques'] = set()
        bp['bn'] = 'bn'
        bp['linear'] = 'linear'
        bp['group_container'] = 'group_container'
        fn = ScopedResGroup.make_blueprint
        bp['group_elements'] = [('group%d' % i, fn('group%d' % i, num_blocks)) for i in range(num_groups)]
        return bp

