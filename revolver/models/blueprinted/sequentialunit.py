from torch.nn import Module, Sequential
from revolver.meta.blueprint import make_module


class SequentialUnit(Module):
    """Regular conv2d convdim summed with transposed conv2d (deconv) output"""

    def __init__(self, blueprint, *_, **__):
        super(SequentialUnit, self).__init__()

        self.blueprint = blueprint
        self.sequence = None
        self.update(True)

    def update(self, init=False):
        if not init:
            super(SequentialUnit, self).update()

        blueprint = self.blueprint

        module_order = blueprint['module_order']
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
