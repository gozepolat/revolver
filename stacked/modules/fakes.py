class Mask:
    def __init__(self, *_, **__):
        return

    def __call__(self, module_out, mask, *_):
        raise NotImplementedError("Mask operator not implemented")


class MaskMultiplied(Mask):
    def __call__(self, module_out, mask, *_):
        return module_out * mask


class MaskIdentity(Mask):
    def __call__(self, module_out, mask, *_):
        return module_out


class MaskHarmonicMean(Mask):
    def __call__(self, module_out, mask, *_):
        return module_out * mask * 2 / (module_out + mask)


class MaskMultipliedSummed(Mask):
    def __call__(self, module_out, mask, *args):
        # x, module_out and mask should have the same size
        x = args[0]
        return x + module_out * mask


class MaskScalarMultipliedSummed(Mask):
    def __call__(self, module_out, mask, *args):
        return module_out * args[0] + mask * (1 - args[0])


class MaskPowered(Mask):
    def __call__(self, module_out, mask, *_):
        return module_out ** mask


class MaskSummed(Mask):
    def __call__(self, module_out, mask, *_):
        return module_out + mask





