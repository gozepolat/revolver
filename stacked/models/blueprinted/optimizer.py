# -*- coding: utf-8 -*-
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
from stacked.models.blueprinted.resnet import ScopedResNet
from stacked.meta.scope import ScopedMeta
from stacked.meta.blueprint import Blueprint, make_module
from stacked.modules.scoped_nn import ScopedBatchNorm2d, \
    ScopedReLU, ScopedConv2d, ScopedLinear
from stacked.utils.engine import EpochEngine, EngineEventHooks
from stacked.utils.dataset import create_dataset
from six import add_metaclass
from torch.optim import SGD


@add_metaclass(ScopedMeta)
class ScopedOptimizerMaker:
    def __init__(self, scope, blueprint, *_, **__):
        self.scope = scope
        self.optimizer_type = blueprint['optimizer_type']

    def __call__(self, model, *args, **kwargs):
        return self.optimizer_type(model.parameters(), *args, **kwargs)

    @staticmethod
    def describe_default(prefix, suffix, parent, optimizer_type):
        default = Blueprint(prefix, suffix, parent, False, ScopedOptimizerMaker)

        default['optimizer']['optimizer_type'] = optimizer_type

        default['kwargs'] = {'blueprint': default}
        return default


@add_metaclass(ScopedMeta)
class ScopedDataLoader(DataLoader):
    def __init__(self, scope, blueprint, *_, **__):
        dataset = create_dataset(blueprint['dataset'],
                                 '.', blueprint['train_mode'],
                                 blueprint['crop_size'])
        super(ScopedDataLoader, self).__init__(dataset,
                                               blueprint['batch_size'],
                                               blueprint['train_mode'],
                                               blueprint['num_thread'],
                                               torch.cuda.is_available())
        self.scope = scope

    @staticmethod
    def describe_default(prefix, suffix, parent,
                         dataset, train_mode, batch_size, num_thread, crop_size):
        default = Blueprint(prefix, suffix, parent, False, ScopedDataLoader)

        default['dataset'] = dataset
        default['batch_size'] = batch_size
        default['train_mode'] = train_mode
        default['num_thread'] = num_thread
        default['crop_size'] = crop_size

        default['kwargs'] = {'blueprint': default}
        return default


@add_metaclass(ScopedMeta)
class ScopedCriterion:
    def __init__(self, scope, blueprint, *_, **__):
        self.scope = scope
        self.criterion = blueprint['criterion']

    def __call__(self, out, target, *args, **kwargs):
        return self.criterion(out, target)

    @staticmethod
    def describe_default(prefix, suffix, parent,
                         criterion=F.cross_entropy):
        default = Blueprint(prefix, suffix, parent, False, ScopedCriterion)
        default['criterion'] = criterion
        default['kwargs'] = {'blueprint': default}
        return default


@add_metaclass(ScopedMeta)
class ScopedNetRunner:
    def __init__(self, scope, blueprint, *_, **__):
        self.scope = scope
        self.loss_func = make_module(blueprint['loss_func'])
        self.net = None

    def __call__(self, sample):
        x_input = Variable(getattr(sample[0].cuda(), 'float'))
        y_targets = Variable(getattr(sample[1].cuda(), 'long'))
        y_out = self.net(x_input)
        return self.loss_func(y_out, y_targets), y_out

    @staticmethod
    def describe_default(prefix, suffix, parent,
                         criterion=F.cross_entropy,
                         loss_func=ScopedCriterion):
        default = Blueprint(prefix, suffix, parent, True, ScopedNetRunner)
        default['loss_func'] = loss_func.describe_default("%s/loss_func" % prefix,
                                                          suffix, default, criterion)
        default['kwargs'] = {'blueprint': default}
        return default


@add_metaclass(ScopedMeta)
class ScopedEpochEngine(EpochEngine):
    """Training engine with blueprint"""
    def __init__(self, scope, blueprint, *_, **__):
        super(ScopedEpochEngine, self).__init__()
        self.scope = scope

        engine = self

        train_loader = make_module(blueprint['train_loader'])
        test_loader = make_module(blueprint['test_loader'])

        net = make_module(blueprint['net'])
        net_runner = make_module(blueprint['net_runner'])
        net_runner.net = net

        optimizer_maker = make_module(blueprint['optimizer_maker'])

        logger = make_module(blueprint['logger'])

        lr = blueprint['learning_rate']
        lr_decay_ratio = blueprint['lr_decay_ratio']
        lr_drop_epochs = blueprint['lr_drop_epochs']

        train_id = id(blueprint)
        hooks = EngineEventHooks(engine, train_loader, test_loader, net,
                                 net_runner, optimizer_maker, lr,
                                 lr_decay_ratio, lr_drop_epochs,
                                 logger, train_id)

        self.hooks['on_sample'] = hooks.on_sample
        self.hooks['on_forward'] = hooks.on_forward
        self.hooks['on_start_epoch'] = hooks.on_start_epoch
        self.hooks['on_end_epoch'] = hooks.on_end_epoch
        self.hooks['on_start'] = hooks.on_start

        self.set_state(net_runner, train_loader,
                       blueprint(['max_epoch']), optimizer_maker(net, lr),
                       epoch=0, t=0, train=True)

        self.hook('on_start', self.state)

    @staticmethod
    def describe_default(prefix='EpochEngine', suffix='', parent=None,
                         net_blueprint=None, skeleton=(16, 32, 64),
                         num_classes=10, depth=28, width=1,
                         block_depth=2, conv_module=ScopedConv2d,
                         bn_module=ScopedBatchNorm2d, linear_module=ScopedLinear,
                         act_module=ScopedReLU, kernel_size=3, padding=1,
                         input_shape=None, dilation=1, groups=1, bias=True,
                         conv3d_args=None, optimizer_maker=ScopedOptimizerMaker,
                         optimizer_type=SGD, max_epoch=200, batch_size=128,
                         learning_rate=0.1, lr_decay_ratio=0.2,
                         lr_drop_epochs=(60, 120, 160), logger=None,
                         data_loader=ScopedDataLoader, dataset="CIFAR10",
                         crop_size=32, num_thread=4, net_runner=ScopedNetRunner,
                         criterion=F.cross_entropy, loss_func=ScopedCriterion):
        """Create a default ResBlock blueprint

        Args:
            prefix (str): Prefix from which the member scopes will be created
            suffix (str): Suffix to append the name of the scoped object
            parent (Blueprint): None or the instance of the parent blueprint
            net_blueprint (Blueprint): None or the blueprint of the network to be used
            skeleton (iterable): Smallest possible widths per group
            num_classes (int): Number of categories for supervised learning
            depth (int): Overall depth of the network
            width (int): Scalar to get the scaled width per group
            block_depth (int): Number of [conv/act/bn] units in the block
            conv_module (type): CNN module to use in forward. e.g. ScopedConv2d
            bn_module (type): Batch normalization module. e.g. ScopedBatchNorm2d
            linear_module (type): Linear module for classification e.g. ScopedLinear
            act_module (type): Activation module e.g ScopedReLU
            kernel_size (int or tuple): Size of the convolving kernel.
            padding (int or tuple, optional): Padding for the first convolution
            input_shape (tuple): (N, C_{in}, H_{in}, W_{in})
            dilation (int): Spacing between kernel elements.
            groups: Number of blocked connections from input to output channels.
            bias: Add a learnable bias if True
            conv3d_args: extra conv arguments to be used in children
            optimizer_maker: Functor that will return an optimizer
            optimizer_type: Type of the optimizer that will be returned
            max_epoch: Maximum number of epochs for training
            batch_size: Batch size for training
            learning_rate: Initial learning rate for training
            lr_decay_ratio: Scalar that will be multiplied with the learning rate
            lr_drop_epochs: Epoch numbers where where lr *= lr_decay_ratio will occur
            logger: Functor that will print the training progress (None: only to stdout)
            data_loader: Loader that will load a dataset (and pad / augment it)
            dataset: Name of the dataset the loader can use
            crop_size: Size of the image samples for cropping after padding for data loader
            num_thread: Number of subprocesses for the data loader
            net_runner: Functor that runs the network and returns loss, output
            criterion: Loss criterion function to be used in net_runner
            loss_func: Module that can customize the loss criterion or use it as is
        """
        default = Blueprint(prefix, suffix, parent, False, ScopedEpochEngine)

        default['learning_rate'] = learning_rate
        default['lr_decay_ratio'] = lr_decay_ratio
        default['lr_drop_epochs'] = lr_drop_epochs
        default['max_epoch'] = max_epoch

        default['net_runner'] = net_runner.describe_default("%s/net_runner" % prefix,
                                                            suffix, default, criterion,
                                                            loss_func)

        default['train_loader'] = data_loader.describe_default("%s/train_loader" % prefix,
                                                               suffix, default, dataset,
                                                               True, batch_size, num_thread,
                                                               crop_size)

        default['test_loader'] = data_loader.describe_default("%s/test_loader" % prefix,
                                                              suffix, default, dataset,
                                                              False, batch_size, num_thread,
                                                              crop_size)
        if logger is not None:
            default['logger'] = Blueprint("%s/optimizer" % prefix, suffix, default)
        else:
            # make_module will return None
            default['logger'] = logger.describe_default("%s/optimizer" % prefix, suffix,
                                                        default)

        default['optimizer_maker'] = optimizer_maker.describe_default("%s/optimizer" % prefix,
                                                                      suffix, default, optimizer_type)

        if net_blueprint is not None:
            default['net'] = net_blueprint
            default['net']['parent'] = default
        else:
            default['net'] = ScopedResNet.describe_default("%s/ResNet" % prefix, suffix,
                                                           default, skeleton, num_classes,
                                                           depth, width, block_depth,
                                                           conv_module, bn_module,
                                                           linear_module, act_module,
                                                           kernel_size, padding,
                                                           input_shape, dilation, groups,
                                                           bias, conv3d_args)

