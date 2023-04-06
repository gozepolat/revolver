# -*- coding: utf-8 -*-
import torch.nn.functional as F
from revolver.modules.scoped_nn import ScopedConv2d, \
    ScopedBatchNorm2d, ScopedLinear, ScopedReLU, \
    ScopedMaxPool2d, ScopedAvgPool2d
from revolver.meta.scope import ScopedMeta
from revolver.meta.sequential import Sequential
from revolver.meta.blueprint import Blueprint, make_module
from revolver.models.blueprinted.resgroup import ScopedResGroup
from revolver.models.blueprinted.resblock import ScopedResBlock
from revolver.models.blueprinted.bottleneckblock import ScopedBottleneckBlock
from revolver.models.blueprinted.densesumgroup import ScopedDenseSumGroup
from revolver.models.blueprinted.convunit import ScopedConvUnit, is_conv_simple
from revolver.utils.transformer import all_to_none
from six import add_metaclass
import inspect


@add_metaclass(ScopedMeta)
class ScopedDenseNet(Sequential):
    """DenseNet with blueprint

    Args:
        scope (string): Scope for the self
        blueprint: Description of the scopes and member module types
    """

    def __init__(self, scope, blueprint, *_, **__):
        super(ScopedDenseNet, self).__init__(blueprint)
        self.scope = scope
        self.blueprint = blueprint
        self.conv = None
        self.bn = None
        self.act = None
        self.linear = None
        self.callback = None
        self.update(True)

    def update(self, init=False):
        if not init:
            super(ScopedDenseNet, self).update()
        blueprint = self.blueprint
        self.conv = make_module(blueprint['conv'])
        # if is_conv_simple(blueprint['conv']['type']):
        self.bn = make_module(blueprint['bn'])
        self.act = make_module(blueprint['act'])
        self.linear = make_module(blueprint['linear'])
        self.callback = blueprint['callback']

    def forward(self, x):
        return self.function(self.conv, self.container, self.bn,
                             self.act, self.linear, self.callback,
                             self.scope, id(self), x)

    @staticmethod
    def function(conv, container, bn, act, linear,
                 callback, scope, module_id, x):
        o = conv(x)
        for module in container:
            o = module(o)
        o = bn(o)
        o = act(o)
        o = F.avg_pool2d(o, o.size()[2], 1, 0).view(o.size(0), -1)
        o = linear(o)
        callback(scope, module_id, o)
        return o

    @staticmethod
    def get_num_blocks_per_group(depth, num_groups, block_depth):
        return (depth - 4) // (num_groups * block_depth)

    @staticmethod
    def __set_head(prefix, default, input_shape, out_features,
                   kernel_size=3, stride=1, padding=1, dilation=1,
                   pool_kernel_size=3, pool_stride=2, pool_padding=1,
                   groups=1, bias=False, unit_module=ScopedConvUnit,
                   act_module=ScopedReLU, conv_module=ScopedConv2d,
                   conv_kwargs=None, bn_module=ScopedBatchNorm2d,
                   callback=all_to_none, module_order=None):
        """Set initial layers for the default DenseNet blueprint"""
        if module_order is None:
            module_order = ('conv', 'bn', 'act', 'pool')

        suffix = '_'.join([str(s) for s in (input_shape[1], out_features,
                                            kernel_size, stride, padding,
                                            dilation, groups, bias,
                                            pool_kernel_size, pool_stride,
                                            pool_padding)])

        head = unit_module.describe_default("%s/conv" % prefix,
                                            suffix, default, input_shape,
                                            input_shape[1], out_features,
                                            kernel_size=kernel_size, stride=stride,
                                            padding=padding, dilation=1,
                                            groups=groups, bias=bias,
                                            act_module=act_module, bn_module=bn_module,
                                            conv_module=conv_module,
                                            callback=callback, conv_kwargs=conv_kwargs,
                                            bn_kwargs=None, act_kwargs=None,
                                            dropout_p=0.0, drop_kwargs=None,
                                            module_order=module_order,
                                            pool_module=ScopedMaxPool2d,
                                            pool_kernel_size=pool_kernel_size,
                                            pool_stride=pool_stride,
                                            pool_padding=pool_padding)

        default['conv'] = head

    @staticmethod
    def __set_children(prefix, default, widths, group_depths, depth, kernel_size=3, stride=1,
                       padding=1, dilation=1, groups=1, bias=False, act_module=ScopedReLU,
                       bn_module=ScopedBatchNorm2d, conv_module=ScopedConv2d, callback=all_to_none,
                       conv_kwargs=None, bn_kwargs=None, act_kwargs=None, unit_module=ScopedConvUnit,
                       block_depth=2, dropout_p=0.0, residual=False, block_module=ScopedBottleneckBlock,
                       drop_p=0.0, dense_unit_module=ScopedBottleneckBlock, fractal_depth=1,
                       group_module=ScopedDenseSumGroup, weight_sum=False):
        """Set dense groups for the default DenseNet blueprint"""
        num_groups = len(widths)
        if group_depths is None:
            group_depth = ScopedDenseNet.get_num_blocks_per_group(depth, num_groups,
                                                                  block_depth)
            group_depths = []
            for _ in widths:
                group_depths.append(group_depth)

        block = default['conv']
        children = []
        pool_module_type = ScopedAvgPool2d
        for i, (w, d) in enumerate(zip(widths, group_depths)):
            input_shape = block['output_shape']
            ni = input_shape[1]
            no = w
            suffix = 'dense_%d_%d_%d_%d_%d_%d_%d_%d' % (ni, no, kernel_size, stride,
                                                        padding, dilation, groups, bias)
            # dense group
            block = group_module.describe_default("%s/group" % prefix,
                                                  suffix, default, input_shape,
                                                  ni, no, kernel_size, stride,
                                                  padding=padding, dilation=dilation,
                                                  groups=groups, bias=bias,
                                                  act_module=act_module,
                                                  bn_module=bn_module,
                                                  conv_module=conv_module,
                                                  callback=callback,
                                                  conv_kwargs=conv_kwargs,
                                                  bn_kwargs=bn_kwargs,
                                                  act_kwargs=act_kwargs,
                                                  unit_module=unit_module,
                                                  block_depth=block_depth,
                                                  dropout_p=dropout_p, residual=residual,
                                                  block_module=block_module,
                                                  group_depth=d, drop_p=drop_p,
                                                  dense_unit_module=dense_unit_module,
                                                  fractal_depth=fractal_depth,
                                                  weight_sum=weight_sum)
            children.append(block)

            # transition
            if i < num_groups - 1:
                input_shape = block['output_shape']
                ni = input_shape[1]
                no = ni // 2
                suffix = 'trans_%d_%d_%d_%d_%d_%d_%d_%d' % (ni, no, 1, 2,
                                                            0, 1, groups, bias)
                block = unit_module.describe_default("%s/block" % prefix,
                                                     suffix, default, input_shape,
                                                     ni, no, kernel_size=1, stride=1,
                                                     padding=0, dilation=1, groups=groups,
                                                     bias=bias, act_module=act_module,
                                                     bn_module=bn_module,
                                                     conv_module=conv_module,
                                                     callback=callback,
                                                     conv_kwargs=conv_kwargs,
                                                     bn_kwargs=bn_kwargs, act_kwargs=act_kwargs,
                                                     dropout_p=dropout_p, drop_kwargs=None,
                                                     module_order=('bn', 'act', 'conv', 'pool'),
                                                     pool_module=pool_module_type,
                                                     pool_kernel_size=2, pool_stride=2)
                children.append(block)
                # do not allow resolution smaller than 4
                if block['output_shape'][-2] < 9:
                    pool_module_type = None

        default['children'] = children
        default['depth'] = len(children)

    @staticmethod
    def __set_tail(prefix, default, bn_module, act_module,
                   act_kwargs, num_classes, linear_module, batch_size):
        """Set final layers for the default DenseNet blueprint"""
        block = default['conv']
        if len(default['children']) > 0:
            block = default['children'][-1]

        w = block['output_shape'][1]
        default['bn'] = Blueprint('%s/bn' % prefix, 'tail_%d' % w, default, True,
                                  bn_module, kwargs={'num_features': w})

        if act_kwargs is None:
            if inspect.isclass(act_module) and issubclass(act_module, ScopedReLU):
                act_kwargs = {'inplace': True}
        default['act'] = Blueprint('%s/act' % prefix, '%d' % w, default, False,
                                   act_module, kwargs=act_kwargs)

        kwargs = {'in_features': w, 'out_features': num_classes}
        default['linear'] = Blueprint('%s/linear' % prefix, '%d_%d' % (w, num_classes),
                                      default, True, linear_module, kwargs=kwargs)
        default['linear']['input_shape'] = (batch_size, w)
        default['linear']['output_shape'] = (batch_size, num_classes)
        default['output_shape'] = (batch_size, num_classes)

    @staticmethod
    def describe_default(prefix='DenseNet', suffix='', parent=None,
                         skeleton=(16, 16, 16), group_depths=None,
                         num_classes=10, depth=28, width=1, block_depth=2,
                         block_module=ScopedBottleneckBlock, conv_module=ScopedConv2d,
                         bn_module=ScopedBatchNorm2d, linear_module=ScopedLinear,
                         act_module=ScopedReLU, kernel_size=3, padding=1,
                         input_shape=None, dilation=1, groups=1, bias=False,
                         callback=all_to_none, drop_p=0.0, dropout_p=0.0,
                         residual=True, conv_kwargs=None, bn_kwargs=None,
                         act_kwargs=None, unit_module=ScopedConvUnit,
                         group_module=ScopedResGroup, fractal_depth=1,
                         dense_unit_module=ScopedConvUnit, weight_sum=False,
                         head_kernel=3, head_stride=1, head_padding=1,
                         head_pool_kernel=3, head_pool_stride=2,
                         head_pool_padding=1, head_modules=('conv', 'bn', 'act', 'pool'),
                         mutation_p=0.8,
                         *_, **__):
        """Create a default DenseNet blueprint

        Args:
            prefix (str): Prefix from which the member scopes will be created
            suffix (str): Suffix to append the name of the scoped object
            parent (Blueprint): None or the instance of the parent blueprint
            skeleton (iterable): Smallest possible widths per group
            group_depths (iterable): Finer grained group depth description
            num_classes (int): Number of categories for supervised learning
            depth (int): Overall depth of the network
            width (int): Scalar to get the scaled width per group
            block_depth (int): Number of [conv/act/bn] units in the block
            block_module: Children modules used as block modules
            conv_module: CNN module to use in forward. e.g. ScopedConv2d
            bn_module: Batch normalization module. e.g. ScopedBatchNorm2d
            linear_module: Linear module for classification e.g. ScopedLinear
            act_module: Activation module e.g ScopedReLU
            kernel_size (int or tuple): Size of the convolving kernel.
            padding (int or tuple, optional): Padding for the first convolution
            input_shape (tuple): (N, C_{in}, H_{in}, W_{in})
            dilation (int): Spacing between kernel elements.
            groups (int): Number of blocked connections from input to output channels.
            bias (bool): Add a learnable bias if True
            callback: Function to call after the output in forward is calculated
            drop_p (float): Probability of vertical drop
            dropout_p (float): Probability of dropout in the blocks
            residual (bool): True if a shortcut connection will be used
            conv_kwargs: Extra conv arguments to be used in children
            bn_kwargs: Extra bn args, if bn module requires other than 'num_features'
            act_kwargs: Extra act args, if act module requires other than defaults
            unit_module: Basic building unit of resblock
            group_module: Basic building group of resnet
            fractal_depth (int): Recursion depth for fractal group module
            dense_unit_module: Children modules that will be used in dense connections
            weight_sum (bool): Weight sum and softmax the reused blocks or not
            head_kernel (int or tuple): Size of the kernel for head conv
            head_stride (int or tuple): Size of the stride for head conv
            head_padding (int or tuple): Size of the padding for head conv
            head_pool_kernel (int or tuple): Size of the first pool kernel
            head_pool_stride (int or tuple): Size of the first pool stride
            head_pool_padding (int or tuple): Size of the first pool padding
            head_modules (iterable): Key list of head modules
            mutation_p (float): How much mutation is allowed as default
        """
        if input_shape is None:
            # assume batch_size = 1, in_channels: 3, h: 32, and w : 32
            input_shape = (1, 3, 32, 32)

        batch_size = input_shape[0]
        widths = [i * width for i in skeleton]

        default = Blueprint(prefix, suffix, parent, False, ScopedDenseNet)
        default['input_shape'] = input_shape
        default['callback'] = callback

        ScopedDenseNet.__set_head(prefix, default, input_shape, 2 * widths[0],
                                  kernel_size=head_kernel, stride=head_stride,
                                  padding=head_padding, pool_kernel_size=head_pool_kernel,
                                  pool_stride=head_pool_stride, pool_padding=head_pool_padding,
                                  groups=1, bias=False, unit_module=unit_module,
                                  act_module=act_module, conv_module=conv_module,
                                  conv_kwargs=conv_kwargs, bn_module=bn_module,
                                  callback=callback, module_order=head_modules)

        ScopedDenseNet.__set_children(prefix, default, widths, group_depths, depth,
                                      kernel_size=kernel_size, stride=1, padding=padding,
                                      dilation=dilation, groups=groups, bias=bias,
                                      act_module=act_module, bn_module=bn_module,
                                      conv_module=conv_module, callback=callback,
                                      conv_kwargs=conv_kwargs, bn_kwargs=bn_kwargs,
                                      act_kwargs=act_kwargs, unit_module=unit_module,
                                      block_depth=block_depth, dropout_p=dropout_p,
                                      residual=residual, block_module=block_module,
                                      drop_p=drop_p, dense_unit_module=dense_unit_module,
                                      fractal_depth=fractal_depth, group_module=group_module,
                                      weight_sum=weight_sum)

        ScopedDenseNet.__set_tail(prefix, default, bn_module, act_module,
                                  act_kwargs, num_classes, linear_module, batch_size)
        default['kwargs'] = {'blueprint': default, 'kernel_size': kernel_size,
                             'stride': head_stride, 'padding': padding,
                             'dilation': dilation, 'groups': groups, 'bias': bias}
        default['mutation_p'] = mutation_p
        return default
