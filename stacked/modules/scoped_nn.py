from six import add_metaclass
from stacked.meta.scope import ScopedMeta
from stacked.utils.transformer import scalar_to_tensor
from stacked.meta.blueprint import Blueprint
from torch.nn import Conv2d, Conv3d, BatchNorm2d, \
    BatchNorm3d, Linear, Module, ModuleList, Parameter, \
    ParameterList, ReLU, Tanh, Hardtanh, Sigmoid, CrossEntropyLoss
from stacked.modules.conv import Conv3d2d, get_conv_out_shape
from stacked.modules.loss import FeatureSimilarityLoss, ParameterSimilarityLoss


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
                         dilation=1, groups=1, bias=True, *_, **__):
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
            dilation: Spacing between kernel elements.
            groups: Number of blocked connections from input to output channels.
            bias: Add a learnable bias if True
        """
        kwargs = {'in_channels': in_channels,
                  'out_channels': out_channels,
                  'kernel_size': kernel_size, 'stride': stride,
                  'padding': padding, 'dilation': dilation,
                  'groups': groups, 'bias': bias}

        bp = Blueprint(prefix, suffix, parent, False,
                       ScopedConv2d, kwargs=kwargs)

        assert(input_shape is not None)
        bp['input_shape'] = input_shape
        bp['output_shape'] = get_conv_out_shape(input_shape, out_channels,
                                                kernel_size, stride,
                                                padding, dilation)
        assert(in_channels == bp['input_shape'][1])
        return bp


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
                         conv3d_args=None, *_, **__):
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
            conv3d_args: direct conv arguments to be used as kwargs
        """
        conv3d_args = ScopedConv3d2d.adjust_args(conv3d_args, ScopedConv3d2d,
                                                 in_channels, out_channels,
                                                 kernel_size, stride, padding,
                                                 dilation, groups, bias)

        bp = Blueprint(prefix, suffix, parent, False,
                       ScopedConv3d2d, kwargs=conv3d_args)

        assert (input_shape is not None)
        bp['input_shape'] = input_shape

        # override with conv3d_args
        in_channels = conv3d_args['in_channels']
        out_channels = conv3d_args['out_channels']
        kernel_size = conv3d_args['kernel_size']
        stride = conv3d_args['stride']
        padding = conv3d_args['padding']
        dilation = conv3d_args['dilation']

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
    def adjust_args(conv_args, conv_module, in_channels, out_channels,
                    kernel_size, stride, padding, dilation, groups, bias):
        """Update the conv_args if the given conv module has missing keys"""
        if conv_args is None:
            conv_args = dict()
        else:
            conv_args = conv_args.copy()

        def need_to_set(key):
            if key not in conv_args or issubclass(conv_module, ScopedConv2d):
                return True
            return False

        if need_to_set('in_channels'):
            conv_args['in_channels'] = in_channels
        if need_to_set('out_channels'):
            conv_args['out_channels'] = out_channels
        if need_to_set('kernel_size'):
            conv_args['kernel_size'] = kernel_size
        if need_to_set('stride'):
            conv_args['stride'] = stride
        if need_to_set('padding'):
            conv_args['padding'] = padding
        if need_to_set('dilation'):
            conv_args['dilation'] = dilation
        if need_to_set('groups'):
            conv_args['groups'] = groups
        if need_to_set('bias'):
            conv_args['bias'] = bias

        return conv_args


@add_metaclass(ScopedMeta)
class ScopedBatchNorm2d(BatchNorm2d):
    def __init__(self, scope, *args, **kwargs):
        super(ScopedBatchNorm2d, self).__init__(*args, **kwargs)
        self.scope = scope


@add_metaclass(ScopedMeta)
class ScopedConv3d(Conv3d):
    def __init__(self, scope, *args, **kwargs):
        super(ScopedConv3d, self).__init__(*args, **kwargs)
        self.scope = scope
        self.eval()


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


@add_metaclass(ScopedMeta)
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
class ParameterModule(Module):
    def __init__(self, scope, value, size, *args, **kwargs):
        super(ParameterModule, self).__init__(*args, **kwargs)
        self.scope = scope
        self.parameter = Parameter(scalar_to_tensor(value, size))

    def forward(self, *_):
        return self.parameter


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
