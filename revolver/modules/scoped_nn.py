from six import add_metaclass
from revolver.meta.scope import ScopedMeta
from revolver.meta.blueprint import Blueprint
from torch.nn import Conv2d, Conv3d, BatchNorm2d, \
    BatchNorm3d, Linear, Module, ModuleList, Parameter, \
    ParameterList, ReLU, ReLU6, Tanh, Hardtanh, Sigmoid, \
    CrossEntropyLoss, AvgPool2d, Dropout2d, MaxPool2d, ConvTranspose2d
from revolver.modules.conv import Conv3d2d, get_conv_out_shape, \
    get_deconv_out_shape, Conv2dDeconv2dConcat
from revolver.modules.loss import FeatureSimilarityLoss, \
    ParameterSimilarityLoss, FeatureConvergenceLoss
import copy
from revolver.utils import common
from logging import warning
import inspect


def log(log_func, msg):
    if common.DEBUG_BLUEPRINT:
        log_func("revolver.modules.scoped_nn: %s" % msg)


def adjust_padding(padding, kernel_size):
    if padding != kernel_size // 2:
        warn = "padding {} would reduce the resolution, instead using {}".format(padding, kernel_size // 2)
        log(warning, warn)
        return kernel_size // 2
    return padding


@add_metaclass(ScopedMeta)
class ScopedConv2d(Conv2d):
    def __init__(self, scope, in_channels, out_channels,
                 kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, *_, **__):
        super(ScopedConv2d, self).__init__(in_channels, out_channels,
                                           kernel_size, stride, padding,
                                           dilation, groups, bias)
        self.scope = scope

    @staticmethod
    def describe_default(prefix='conv', suffix='', parent=None,
                         input_shape=None, in_channels=3, out_channels=3,
                         kernel_size=3, stride=1, padding=1,
                         dilation=1, groups=1, bias=True,
                         mutation_p=0.8, toggle_p=common.UNIQUENESS_TOGGLE_P, *_, **__):
        """Create a default ScopedConv2d blueprint

        Args:
            prefix (str): Prefix from which the member scopes will be created
            suffix (str): Suffix to append the name of the scoped object
            parent (Blueprint): None or the instance of the parent blueprint
            in_channels (int): Number of channels in the input
            out_channels (int): Number of channels produced by the block
            kernel_size (int or tuple): Size of the convolving kernel. Default: 3
            stride (int or tuple, optional): Stride for the first convolution
            padding (int or tuple, optional): Padding for the first convolution
            input_shape (tuple): (N, C_{in}, H_{in}, W_{in})
            dilation (int): Spacing between kernel elements.
            groups (int): Number of blocked connections from input to output channels.
            bias (bool): Add a learnable bias if True
        """
        padding = adjust_padding(padding, kernel_size)
        kwargs = {'in_channels': in_channels,
                  'out_channels': out_channels,
                  'kernel_size': kernel_size, 'stride': stride,
                  'padding': padding, 'dilation': dilation,
                  'groups': groups, 'bias': bias}

        suffix = f"{suffix}, {input_shape[1]}, {out_channels},{kwargs['kernel_size']},{kwargs['stride']}, " \
                 f"{kwargs['padding']}, {kwargs['dilation']},{kwargs['groups']},{kwargs['bias']},"

        # prefix is conv for aggressive parameter sharing
        if common.AGGRESSIVELY_SHARE:
            prefix = "conv"

        bp = Blueprint(prefix, suffix, parent, False,
                       ScopedConv2d, kwargs=kwargs)

        assert (input_shape is not None)
        bp['input_shape'] = input_shape
        bp['output_shape'] = get_conv_out_shape(input_shape, out_channels,
                                                kernel_size, stride,
                                                padding, dilation)
        assert (in_channels == bp['input_shape'][1])
        bp['mutation_p'] = mutation_p
        bp['toggle_p'] = toggle_p
        return bp

    @staticmethod
    def adjust_args(conv_kwargs, kernel_size, stride, padding, dilation, groups, bias):
        def override_default(key, default):
            return conv_kwargs[key] if default is None else default

        _kernel = override_default('kernel_size', kernel_size)
        _stride = override_default('stride', stride)
        _padding = override_default('padding', padding)
        _dilation = override_default('dilation', dilation)
        _groups = override_default('groups', groups)
        _bias = override_default('bias', bias)
        return _kernel, _stride, _padding, _dilation, _groups, _bias


@add_metaclass(ScopedMeta)
class ScopedConvTranspose2d(ConvTranspose2d):
    def __init__(self, scope, blueprint, *_, **__):
        input_shape = blueprint['input_shape']
        output_shape = blueprint['output_shape']
        in_channels = input_shape[1]
        out_channels = output_shape[1]
        kwargs = blueprint['kwargs']
        kernel_size = kwargs['kernel_size']
        stride = kwargs['stride']
        padding = kwargs['padding']
        dilation = kwargs['dilation']
        groups = kwargs['groups']
        bias = kwargs['bias']
        super(ScopedConvTranspose2d, self).__init__(in_channels, out_channels,
                                                    kernel_size, stride=stride,
                                                    padding=padding, dilation=dilation,
                                                    groups=groups, bias=bias)
        self.scope = scope

    @staticmethod
    def describe_default(prefix='conv', suffix='', parent=None,
                         input_shape=None, in_channels=3, out_channels=3,
                         kernel_size=3, stride=1, padding=1,
                         dilation=1, groups=1, bias=True, *_, **__):
        padding = adjust_padding(padding, kernel_size)
        bp = ScopedConv2d.describe_default(prefix, suffix, parent,
                                           input_shape, in_channels, out_channels,
                                           kernel_size, stride, padding,
                                           dilation, groups, bias)
        bp['output_shape'] = get_deconv_out_shape(input_shape, out_channels,
                                                  kernel_size, stride,
                                                  padding, dilation)
        bp['type'] = ScopedConvTranspose2d
        bp['kwargs']['blueprint'] = bp
        return bp

    @staticmethod
    def describe_from_blueprint(prefix, suffix, blueprint, parent, *_, **__):
        input_shape = blueprint['input_shape']
        output_shape = blueprint['output_shape']
        kwargs = blueprint['kwargs']
        suffix = f"{suffix}_{input_shape[1]}_{output_shape[1]}_{kwargs['kernel_size']}_{kwargs['stride']}_" \
                 f"{kwargs['padding']}_{kwargs['dilation']}_{kwargs['groups']}_{kwargs['bias']},"
        if parent is None:
            parent = blueprint['parent']

        bp = copy.deepcopy(blueprint)

        # if bp['input_shape'] != bp['output_shape']:
        #    bp.refresh_name()
        #    bp.refresh_unique_suffixes()
        #    return bp

        bp['type'] = ScopedConvTranspose2d

        # aggressive parameter sharing where prefix has no scope
        prefix = f"{prefix}/deconv"
        if common.AGGRESSIVELY_SHARE:
            prefix = 'deconv'
        bp['prefix'] = prefix
        bp['suffix'] = suffix
        bp['parent'] = parent
        bp['output_shape'] = get_deconv_out_shape(input_shape, output_shape[1],
                                                  kwargs['kernel_size'], kwargs['stride'],
                                                  kwargs['padding'], kwargs['dilation'])
        bp['kwargs']['blueprint'] = bp
        bp.refresh_name()
        bp.refresh_unique_suffixes()
        return bp


@add_metaclass(ScopedMeta)
class ScopedConv2dDeconv2dConcat(Conv2dDeconv2dConcat):
    def __init__(self, scope, *args, **kwargs):
        super(ScopedConv2dDeconv2dConcat, self).__init__(*args, **kwargs)
        self.scope = scope

    @staticmethod
    def describe_default(*args, **kwargs):
        bp = ScopedConv2d.describe_default(*args, **kwargs)
        bp['type'] = ScopedConv2dDeconv2dConcat
        return bp

    @staticmethod
    def describe_from_blueprint(prefix, suffix, blueprint, parent, *_, **__):
        input_shape = blueprint['input_shape']
        output_shape = blueprint['output_shape']
        kwargs = blueprint['kwargs']
        suffix = "%s_%d_%d_%d_%d_%d_%d_%d_%d" % (suffix, input_shape[1],
                                                 output_shape[1],
                                                 kwargs['kernel_size'],
                                                 kwargs['stride'],
                                                 kwargs['padding'],
                                                 kwargs['dilation'],
                                                 kwargs['groups'],
                                                 kwargs['bias'],)
        if parent is None:
            parent = blueprint['parent']
        bp = copy.deepcopy(blueprint)
        bp['type'] = ScopedConv2dDeconv2dConcat
        bp['prefix'] = '%s/conv' % prefix
        bp['suffix'] = suffix
        bp['parent'] = parent
        bp.refresh_name()
        bp.refresh_unique_suffixes()
        return bp


class Conv3d2dMeta(ScopedMeta, Conv3d2d.__class__):
    pass


@add_metaclass(ScopedMeta)
class ScopedConv3d2d(Conv3d2d):
    def __init__(self, scope, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=0, dilation=1, groups=1, bias=True, *_, **__):

        super(ScopedConv3d2d, self).__init__(in_channels, out_channels,
                                             kernel_size, stride, padding,
                                             dilation, groups, bias)
        self.scope = scope

    @staticmethod
    def describe_default(prefix='conv3d2d', suffix='', parent=None,
                         input_shape=None, in_channels=3, out_channels=3,
                         kernel_size=3, stride=1, padding=1,
                         dilation=1, groups=1, bias=True,
                         conv_kwargs=None, *_, **__):
        """Create a default ScopedConv3d2d blueprint

        Args:
            prefix (str): Prefix from which the member scopes will be created
            suffix (str): Suffix to append the name of the scoped object
            parent (Blueprint): None or the instance of the parent blueprint
            in_channels (int): Number of channels in the input
            out_channels (int): Number of channels produced by the block
            kernel_size (int or tuple): Size of the convolving kernel. Default: 3
            stride (int or tuple, optional): Stride for the first convolution
            padding (int or tuple, optional): Padding for the first convolution
            input_shape (tuple): (N, C_{in}, H_{in}, W_{in})
            dilation: Spacing between kernel elements.
            groups: Number of blocked connections from input to output channels.
            bias: Add a learnable bias if True
            conv_kwargs: extra conv kwargs for Conv3d2d
        """
        conv_kwargs = ScopedConv3d2d.adjust_args(conv_kwargs, ScopedConv3d2d,
                                                 in_channels, out_channels,
                                                 kernel_size, stride, padding,
                                                 dilation, groups, bias)

        bp = Blueprint(prefix, suffix, parent, False,
                       ScopedConv3d2d, kwargs=conv_kwargs)

        assert (input_shape is not None)
        bp['input_shape'] = input_shape

        # override with conv3d_args
        in_channels = conv_kwargs['in_channels']
        out_channels = conv_kwargs['out_channels']
        kernel_size = conv_kwargs['kernel_size']
        stride = conv_kwargs['stride']
        padding = conv_kwargs['padding']
        dilation = conv_kwargs['dilation']
        padding = adjust_padding(padding, kernel_size)

        if isinstance(stride, int):
            conv_kwargs['stride'] = stride = (1, stride, stride)

        if len(input_shape) == 4:
            input_shape = (input_shape[0], in_channels,
                           input_shape[1] // in_channels,
                           input_shape[2], input_shape[3])

            output_shape = get_conv_out_shape(input_shape, out_channels,
                                              kernel_size, stride,
                                              padding, dilation)

            bp['output_shape'] = (output_shape[0],
                                  output_shape[2] * out_channels,
                                  output_shape[3], output_shape[4])

            return bp

        bp['output_shape'] = get_conv_out_shape(input_shape, out_channels,
                                                kernel_size, stride,
                                                padding, dilation)
        return bp

    @staticmethod
    def describe_from_blueprint(prefix, suffix, blueprint, parent, conv_kwargs=None):
        input_shape = blueprint['input_shape']
        output_shape = blueprint['output_shape']
        kwargs = blueprint['kwargs']
        suffix = "{}_{}_{}_{}_{}_{}_{}_{}_{}".format(suffix, input_shape[1],
                                                     output_shape[1],
                                                     kwargs['kernel_size'],
                                                     kwargs['stride'],
                                                     kwargs['padding'],
                                                     kwargs['dilation'],
                                                     kwargs['groups'],
                                                     kwargs['bias'], )
        if parent is None:
            parent = blueprint['parent']

        prefix = '%s/conv' % prefix
        in_channels = input_shape[1]
        out_channels = output_shape[1]
        kernel_size = kwargs['kernel_size']
        stride = kwargs['stride']
        padding = kwargs['padding']
        dilation = kwargs['dilation']
        groups = kwargs['groups']
        bias = kwargs['bias']

        return ScopedConv3d2d.describe_default(prefix, suffix, parent, input_shape,
                                               in_channels, out_channels, kernel_size,
                                               stride, padding, dilation, groups,
                                               bias, conv_kwargs)

    @staticmethod
    def adjust_args(conv_kwargs, conv_module, in_channels, out_channels,
                    kernel_size, stride, padding, dilation, groups, bias):
        """Update the conv_args if the given conv module has missing keys"""
        if conv_kwargs is None:
            conv_kwargs = dict()
        else:
            conv_kwargs = conv_kwargs.copy()

        def need_to_set(key):
            if key not in conv_kwargs or (inspect.isclass(conv_module) and issubclass(conv_module, ScopedConv2d)):
                return True
            return False

        if need_to_set('in_channels'):
            conv_kwargs['in_channels'] = in_channels
        if need_to_set('out_channels'):
            conv_kwargs['out_channels'] = out_channels
        if need_to_set('kernel_size'):
            conv_kwargs['kernel_size'] = kernel_size
        if need_to_set('stride'):
            conv_kwargs['stride'] = stride
        if need_to_set('padding'):
            conv_kwargs['padding'] = padding
        if need_to_set('dilation'):
            conv_kwargs['dilation'] = dilation
        if need_to_set('groups'):
            conv_kwargs['groups'] = groups
        if need_to_set('bias'):
            conv_kwargs['bias'] = bias

        return conv_kwargs


@add_metaclass(ScopedMeta)
class ScopedBatchNorm2d(BatchNorm2d):
    def __init__(self, scope, *args, **kwargs):
        super(ScopedBatchNorm2d, self).__init__(*args, **kwargs)
        self.scope = scope

    @staticmethod
    def describe_default(prefix='', suffix='', parent=None,
                         input_shape=None,
                         mutation_p=0.8, kwargs=None, *_, **__):
        bn_kwargs = {
            'num_features': input_shape[1],
            'eps': 1e-05,
            'momentum': 0.1,
            'affine': True
        }

        suffix = f"{suffix}_{input_shape[1]}"
        if kwargs is not None:
            bn_kwargs.update(kwargs)
        bp = Blueprint(f"{prefix}/bn" if prefix else "bn", suffix, parent, True,
                       ScopedBatchNorm2d, kwargs=bn_kwargs)

        assert (input_shape is not None)
        bp['input_shape'] = input_shape
        bp['output_shape'] = input_shape
        bp['mutation_p'] = mutation_p
        bp.refresh_name()
        return bp


@add_metaclass(ScopedMeta)
class ScopedConv3d(Conv3d):
    def __init__(self, scope, *args, **kwargs):
        super(ScopedConv3d, self).__init__(*args, **kwargs)
        self.scope = scope


@add_metaclass(ScopedMeta)
class ScopedBatchNorm3d(BatchNorm3d):
    def __init__(self, scope, *args, **kwargs):
        super(ScopedBatchNorm3d, self).__init__(*args, **kwargs)
        self.scope = scope


@add_metaclass(ScopedMeta)
class ScopedLinear(Linear):
    def __init__(self, scope, *args, **kwargs):
        super(ScopedLinear, self).__init__(*args, **kwargs)
        self.scope = scope


@add_metaclass(ScopedMeta)
class ScopedModule(Module):
    def __init__(self, scope, *args, **kwargs):
        super(ScopedModule, self).__init__(*args, **kwargs)
        self.scope = scope

    def forward(self, *input):
        raise NotImplementedError


@add_metaclass(ScopedMeta)
class ScopedModuleList(ModuleList):
    def __init__(self, scope, *args, **kwargs):
        super(ScopedModuleList, self).__init__(*args, **kwargs)
        self.scope = scope


@add_metaclass(ScopedMeta)
class ScopedParameterList(ParameterList):
    def __init__(self, scope, *args, **kwargs):
        super(ScopedParameterList, self).__init__(*args, **kwargs)
        self.scope = scope


# Fix for PyTorch Parameter's own Meta class
class ParameterMeta(ScopedMeta, Parameter.__class__):
    pass


@add_metaclass(ParameterMeta)
class ScopedParameter(Parameter):
    def __init__(self, scope, *args, **kwargs):
        super(ScopedParameter, self).__init__(*args, **kwargs)
        self.scope = scope


@add_metaclass(ScopedMeta)
class ScopedReLU(ReLU):
    def __init__(self, scope, *args, **kwargs):
        super(ScopedReLU, self).__init__(*args, **kwargs)
        self.scope = scope


@add_metaclass(ScopedMeta)
class ScopedReLU6(ReLU6):
    def __init__(self, scope, *args, **kwargs):
        super(ScopedReLU6, self).__init__(*args, **kwargs)
        self.scope = scope


@add_metaclass(ScopedMeta)
class ScopedSigmoid(Sigmoid):
    def __init__(self, scope, *args, **kwargs):
        super(ScopedSigmoid, self).__init__(*args, **kwargs)
        self.scope = scope


@add_metaclass(ScopedMeta)
class ScopedTanh(Tanh):
    def __init__(self, scope, *args, **kwargs):
        super(ScopedTanh, self).__init__(*args, **kwargs)
        self.scope = scope


@add_metaclass(ScopedMeta)
class ScopedHardTanh(Hardtanh):
    def __init__(self, scope, *args, **kwargs):
        super(ScopedHardTanh, self).__init__(*args, **kwargs)
        self.scope = scope


@add_metaclass(ScopedMeta)
class ScopedCrossEntropyLoss(CrossEntropyLoss):
    def __init__(self, scope, *args, **kwargs):
        super(ScopedCrossEntropyLoss, self).__init__(*args, **kwargs)
        self.scope = scope


@add_metaclass(ScopedMeta)
class ScopedFeatureSimilarityLoss(FeatureSimilarityLoss):
    def __init__(self, scope, *args, **kwargs):
        super(ScopedFeatureSimilarityLoss, self).__init__(*args, **kwargs)
        self.scope = scope


@add_metaclass(ScopedMeta)
class ScopedParameterSimilarityLoss(ParameterSimilarityLoss):
    def __init__(self, scope, *args, **kwargs):
        super(ScopedParameterSimilarityLoss, self).__init__(*args, **kwargs)
        self.scope = scope


@add_metaclass(ScopedMeta)
class ScopedFeatureConvergenceLoss(FeatureConvergenceLoss):
    def __init__(self, scope, *args, **kwargs):
        super(ScopedFeatureConvergenceLoss, self).__init__(*args, **kwargs)
        self.scope = scope


@add_metaclass(ScopedMeta)
class ScopedAvgPool2d(AvgPool2d):
    def __init__(self, scope, *args, **kwargs):
        super(ScopedAvgPool2d, self).__init__(*args, **kwargs)
        self.scope = scope


@add_metaclass(ScopedMeta)
class ScopedDropout2d(Dropout2d):
    def __init__(self, scope, *args, **kwargs):
        super(ScopedDropout2d, self).__init__(*args, **kwargs)
        self.scope = scope


@add_metaclass(ScopedMeta)
class ScopedMaxPool2d(MaxPool2d):
    def __init__(self, scope, *args, **kwargs):
        super(ScopedMaxPool2d, self).__init__(*args, **kwargs)
        self.scope = scope
