import torchvision.transforms as T
from torchvision import datasets
import numpy as np
from stacked.utils.transformer import get_transformer


def create_dataset(dataset="CIFAR10", data_root=".",
                   train_mode=True, crop_size=32, padding=4):
    convert = get_transformer(dataset)

    if train_mode:
        convert = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomCrop(crop_size),
            convert,
        ])

    ds = getattr(datasets, dataset)(data_root,
                                    train=train_mode,
                                    download=True,
                                    transform=convert)
    if train_mode:
        ds.train_data = np.pad(ds.train_data,
                               ((0, 0), (padding, padding),
                                (padding, padding), (0, 0)),
                               mode='reflect')
    return ds
