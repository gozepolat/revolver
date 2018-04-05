# -*- coding: utf-8 -*-
import torch
from torch import nn


class Ensemble(nn.Module):
    r"""An ensemble of modules with masked outputs

        Note that this assumes all modules have the same output size, given the input size
    Arguments:
        iterable_args: an iterable that consists of elements in the form: (module, module_args, module_kwargs)
        in_size: the input size that will be used for deducing the mask size
        mask_with_grad: True if mask is learnable
        mask_func: lambda function to run with conv_out and mask"""

    def __init__(self, iterable_args, inp_shape, mask_with_grad=True,
                 mask_func=lambda out, module_out, mask: out + module_out * mask):
        super(Ensemble, self).__init__()
        assert(len(iterable_args) > 0)
        self.stack = torch.nn.ModuleList(
            [module(*args, **kwargs) for module, args, kwargs in iterable_args])
        self.size = len(iterable_args)
        self.mask = torch.nn.ParameterList()
        self.mask_func = mask_func

        # get the output size using the first module in the iterable args
        fake_input = torch.FloatTensor([1]).expand(inp_shape)
        self.out_size = self.stack[0](fake_input).size()

        # create masks using the output size
        for i in range(self.size):
            self.mask.append(nn.Parameter(torch.FloatTensor([1]).expand(self.out_size),
                                          requires_grad=mask_with_grad))

    def forward(self, inp):
        out = 0.0
        for i in range(self.size):
            out = self.mask_func(out, self.stack[i](inp), self.mask[i])
        return out
