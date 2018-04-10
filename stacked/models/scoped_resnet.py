# -*- coding: utf-8 -*-
import torch.nn.functional as F
from torch.nn import Module
from stacked.modules.scoped_nn import ScopedConv2d, \
    ScopedBatchNorm2d, ScopedLinear, ScopedModuleList
from stacked.meta.scoped import ScopedMeta
from six import add_metaclass
from stacked.utils import common
from logging import warning


def log(log_func, msg):
    if common.DEBUG_SCOPED_RESNET:
        log_func(msg)


@add_metaclass(ScopedMeta)
class ScopedResBlock(Module):
    r"""Residual relu block

    Args:
        scope: Global scope for self
        blueprint: A dictionary which stores the scope keywords for modules
        ni (int): Number of channels in the input
        no (int): Number of channels produced by the block
        kernel_size (int or tuple): Size of the convolving kernel. Default: 3
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        conv: Convolution module to use in forward. Default: ScopedConv2d
        act: Activation function to use in forward. Default: torch.nn.functional.relu
    """

    def __init__(self, scope, blueprint, ni, no, kernel_size=3, stride=1,
                 conv=ScopedConv2d, act=F.relu):
        super(ScopedResBlock, self).__init__()
        self.scope = scope
        if blueprint is None:
            log(warning, "Block %s has no blueprint, making a default one" % scope)
            blueprint = self.make_blueprint()
        self.blueprint = blueprint
        scopes = self.get_module_scopes(scope, blueprint)
        self.conv0 = conv(scopes['conv0'], ni, no, kernel_size, stride, padding=1)
        self.bn0 = ScopedBatchNorm2d(scopes['bn0'], ni)
        self.conv1 = conv(scopes['conv1'], no, no, kernel_size, 1, 1)
        self.bn1 = ScopedBatchNorm2d(scopes['bn1'], no)

        # 1x1 conv to correct the number of channels for summation
        self.convdim = ScopedConv2d(scopes['convdim'],
                                    ni, no, 1, stride=stride) if ni != no else None
        self.act = act

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
    def get_module_scopes(scope, blueprint):
        conv0 = "%s/%s" % (scope, blueprint['conv0'])
        bn0 = "%s/%s" % (scope, blueprint['bn0'])
        conv1 = "%s/%s" % (scope, blueprint['conv1'])
        bn1 = "%s/%s" % (scope, blueprint['bn1'])
        convdim = "%s/%s" % (scope, blueprint['convdim'])
        return {'conv0': conv0, 'bn0': bn0, 'conv1': conv1, 'bn1': bn1,
                'convdim': convdim}

    @staticmethod
    def make_blueprint():
        return {'conv0': 'conv0', 'bn0': 'bn0', 'conv1': 'conv1',
                'bn1': 'bn1', 'convdim': 'convdim'}


@add_metaclass(ScopedMeta)
class ScopedResGroup(Module):
    r"""Group of residual blocks with the same number of output channels

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

    def __init__(self, scope, blueprint, ni, no, kernel, stride, num_blocks,
                 conv=ScopedConv2d, act=F.relu):
        super(ScopedResGroup, self).__init__()
        self.scope = scope
        if blueprint is None:
            log(warning, "Group %s has no blueprint, making a default one" % scope)
            blueprint = self.make_blueprint(num_blocks)
        self.blueprint = blueprint
        scopes = self.get_module_scopes(scope, blueprint)
        self.block_container = ScopedModuleList(scopes['block_container'])
        if len(self.block_container) == 0:
            self._add_blocks(scopes, ni, no, kernel, stride, num_blocks, conv, act)

    def _add_blocks(self, scopes, ni, no, kernel_size, stride, num_blocks, conv, act):
        blocks = scopes['block_elements']
        for i in range(num_blocks):
            block_scope = blocks[i][0]
            block_blueprint = blocks[i][1]
            self.block_container.append(
                ScopedResBlock(block_scope, block_blueprint, ni, no,
                               kernel_size, stride, conv, act))
            ni = no
            stride = 1

    def forward(self, x):
        for block in self.block_container:
            x = block(x)
        return x

    @staticmethod
    def get_module_scopes(scope, blueprint):
        block_container = "%s/%s" % (scope, blueprint['block_container'])
        blocks = [("%s/%s" % (scope, key), block_blueprint)
                  for key, block_blueprint in blueprint['block_elements']]
        return {'block_container': block_container, 'block_elements': blocks}

    @staticmethod
    def make_blueprint(num_blocks):
        bp = dict()
        bp['block_container'] = 'block_container'
        fn = ScopedResBlock.make_blueprint
        bp['block_elements'] = [('block%d'%i, fn()) for i in range(num_blocks)]
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

    def __init__(self, scope, blueprint, depth, width, num_classes, skeleton=(16, 32, 64),
                 conv=ScopedConv2d, act=F.relu):
        super(ResNet, self).__init__()
        self.scope = scope
        self.skeleton = skeleton
        self.depth = depth
        self.num_groups = len(skeleton)
        self.num_blocks = self.get_num_blocks_per_group(depth, self.num_groups)

        if blueprint is None:
            log(warning, "ResNet %s has no blueprint, making a default one" % blueprint)
            blueprint = self.make_blueprint(self.num_groups, self.num_blocks)

        self.blueprint = blueprint

        scopes = self.get_module_scopes(scope, blueprint)

        self.widths = [i * width for i in skeleton]
        self.act = act
        self.conv0 = ScopedConv2d(scopes['conv0'], 3, skeleton[0], 3, 1, padding=1)
        self.group_container = ScopedModuleList(scopes['group_container'])
        self.bn = ScopedBatchNorm2d(scopes['bn'], self.widths[2])
        self.linear = ScopedLinear(scopes['linear'], self.widths[2], num_classes)

        if len(self.group_container) == 0:
            self._add_groups(scopes, conv, act)

    def _add_groups(self, scopes, conv, act):
        ni = self.skeleton[0]
        stride = 1
        k = 0
        groups = scopes['group_elements']
        for width in self.widths:
            no = width
            group_scope = groups[k][0]
            group_blueprint = groups[k][1]
            self.group_container.append(
                ScopedResGroup(group_scope, group_blueprint, ni, no, 3,
                               stride, self.num_blocks, conv, act))
            ni = no
            stride = 2
            k += 1

    def forward(self, x):
        x = self.conv0(x)
        for i, _ in enumerate(self.widths):
            x = self.group_container[i](x)
        o = self.act(self.bn(x))
        o = F.avg_pool2d(o, o.size()[2], 1, 0)
        o = o.view(o.size(0), -1)
        o = self.linear(o)
        return o

    @staticmethod
    def get_num_blocks_per_group(depth, num_groups):
        block_depth = 2
        return (depth - 4) // (num_groups * block_depth)

    @staticmethod
    def get_module_scopes(scope, blueprint):
        conv0 = "%s/%s" % (scope, blueprint['conv0'])
        bn = "%s/%s" % (scope, blueprint['bn'])
        linear = "%s/%s" % (scope, blueprint['linear'])
        group_container = "%s/%s" % (scope, blueprint['group_container'])

        groups = [("%s/%s" % (scope, key), group_blueprint)
                  for key, group_blueprint in blueprint['group_elements']]

        return {'conv0': conv0, 'bn': bn, 'linear': linear,
                'group_container': group_container, 'group_elements': groups}

    @staticmethod
    def make_blueprint(num_groups, num_blocks):
        bp = dict()
        bp['conv0'] = 'conv0'
        bp['bn'] = 'bn'
        bp['linear'] = 'linear'
        bp['group_container'] = 'group_container'
        fn = ScopedResGroup.make_blueprint
        bp['group_elements'] = [('group%d'%i, fn(num_blocks)) for i in range(num_groups)]
        return bp

