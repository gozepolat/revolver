from six import add_metaclass
from stacked.meta.scope import ScopedMeta
from torch.nn import Module, Sequential
from stacked.modules.scoped_nn import ScopedConv2d, \
    ScopedConv2dDeconv2dConcat, ScopedConvTranspose2d
from stacked.meta.blueprint import make_module
from stacked.models.blueprinted.ensemble import ScopedEnsembleConcat
from stacked.utils import common
from logging import warning


def log(log_func, msg):
    if common.DEBUG_CONVDECONV:
        log_func("stacked.models.blueprinted.convdeconv: %s" % msg)


@add_metaclass(ScopedMeta)
class ScopedConv2dDeconv2dSum(Module):
    """Regular conv2d convdim summed with transposed conv2d (deconv) output"""

    def __init__(self, scope, blueprint, *_, **__):
        super(ScopedConv2dDeconv2dSum, self).__init__()
        self.scope = scope

        self.blueprint = blueprint
        self.sequence = None
        self.update(True)

    def update(self, init=False):
        if not init:
            super(ScopedConv2dDeconv2dSum, self).update()

        blueprint = self.blueprint

        module_order = blueprint['module_order']

        self.sequence = None
        self.sequence = Sequential()

        for key in module_order:
            module = make_module(blueprint[key])
            if module is not None:
                self.sequence.add_module(key, module)

    def forward(self, x):
        return self.function(self.sequence, x)

    @staticmethod
    def function(sequence, x):
        x = sequence(x)
        return x

    @staticmethod
    def describe_default(prefix='conv', suffix='', parent=None,
                         input_shape=None, in_channels=3, out_channels=3,
                         kernel_size=3, stride=1, padding=1,
                         dilation=1, groups=1, bias=True, separable_deconv=True, *_, **__):
        """Create a default ScopedConv2dDeconv2dSum blueprint

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
            separable_deconv (bool): Conv and deconv with independent and separable weights or not
        """
        # modify the description from vanilla Conv2d
        bp = ScopedConv2d.describe_default(prefix=prefix, suffix=suffix, parent=parent,
                                           input_shape=input_shape, in_channels=in_channels,
                                           out_channels=out_channels, kernel_size=kernel_size,
                                           stride=stride, padding=padding, dilation=dilation,
                                           groups=groups, bias=bias)

        if kernel_size == 1:
            return bp

        bp['type'] = ScopedConv2dDeconv2dSum
        bp['kwargs']['blueprint'] = bp
        bp['module_order'] = []

        if bp['input_shape'] != bp['output_shape']:
            bp['module_order'].append('downsample')
            log(warning, "Conv2dDeconv2d.describe_default: downsampling k: %d, s: %d, p: %d" %
                (kernel_size, stride, padding))
            bp['downsample'] = ScopedConv2d.describe_default(prefix='%s/downsample' % prefix,
                                                             suffix=suffix, parent=bp,
                                                             input_shape=input_shape,
                                                             in_channels=in_channels,
                                                             out_channels=out_channels,
                                                             kernel_size=kernel_size,
                                                             stride=stride, padding=padding,
                                                             dilation=dilation,
                                                             groups=groups, bias=bias)

        bp['module_order'].extend(['convdeconv', 'convdim'])

        if separable_deconv:
            conv = ScopedConv2d.describe_default(prefix='%s/convdeconv/conv' % prefix,
                                                 suffix=suffix,
                                                 input_shape=bp['output_shape'],
                                                 in_channels=out_channels,
                                                 out_channels=out_channels,
                                                 kernel_size=kernel_size,
                                                 stride=1, padding=kernel_size // 2,
                                                 dilation=dilation,
                                                 groups=out_channels, bias=bias)

            deconv = ScopedConvTranspose2d.describe_default(prefix='%s/convdeconv/deconv' % prefix,
                                                            suffix=suffix,
                                                            input_shape=bp['output_shape'],
                                                            in_channels=out_channels,
                                                            out_channels=out_channels,
                                                            kernel_size=kernel_size,
                                                            stride=1, padding=kernel_size // 2,
                                                            dilation=dilation,
                                                            groups=out_channels, bias=bias)
            children = [conv, deconv]
            bp['convdeconv'] = ScopedEnsembleConcat.describe_default(prefix='%s/convdeconv' % prefix,
                                                                     suffix=suffix, parent=bp,
                                                                     children=children)
        else:
            bp['convdeconv'] = ScopedConv2dDeconv2dConcat.describe_default(prefix='%s/convdeconv' % prefix,
                                                                           suffix=suffix, parent=bp,
                                                                           input_shape=bp['output_shape'],
                                                                           in_channels=out_channels,
                                                                           out_channels=out_channels * 2,
                                                                           kernel_size=kernel_size,
                                                                           stride=1, padding=kernel_size // 2,
                                                                           dilation=dilation,
                                                                           groups=groups, bias=bias)

        bp['convdim'] = ScopedConv2d.describe_default(prefix='%s/convdim' % prefix,
                                                      suffix=suffix, parent=bp,
                                                      input_shape=bp['convdeconv']['output_shape'],
                                                      in_channels=out_channels * 2,
                                                      out_channels=out_channels, kernel_size=1,
                                                      stride=1, padding=0, dilation=dilation,
                                                      groups=groups, bias=bias)
        return bp
