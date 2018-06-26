import torchvision.transforms as T
from torchvision import datasets
import numpy as np
from stacked.utils.transformer import get_transformer
import os


def create_imagenet_dataset(dataset="ILSVRC2012", data_root=".",
                            train_mode=True, crop_size=224):
    datadir = os.path.join(dataset, 'val')
    if train_mode:
        datadir = os.path.join(data_root, dataset, 'train')

    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])

    if train_mode:
        return datasets.ImageFolder(datadir,
                                    T.Compose([
                                        T.RandomResizedCrop(crop_size),
                                        T.RandomHorizontalFlip(),
                                        T.ToTensor(),
                                        normalize,
                                    ]))

    return datasets.ImageFolder(datadir,
                                T.Compose([
                                    T.Resize(256),
                                    T.CenterCrop(crop_size),
                                    T.ToTensor(),
                                    normalize,
                                ]))


def create_tiny_imagenet_dataset(dataset="tiny-imagenet-200",
                                 data_root=".",
                                 train_mode=True, crop_size=56):
    datadir = os.path.join(dataset, 'val/images')

    if train_mode:
        datadir = os.path.join(data_root, dataset, 'train/images')

    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])

    if train_mode:
        return datasets.ImageFolder(datadir,
                                    T.Compose([
                                        T.RandomResizedCrop(crop_size),
                                        T.RandomHorizontalFlip(),
                                        T.ToTensor(),
                                        normalize,
                                    ]))

    return datasets.ImageFolder(datadir,
                                T.Compose([
                                    T.Resize(64),
                                    T.CenterCrop(crop_size),
                                    T.ToTensor(),
                                    normalize,
                                ]))


def create_dataset(dataset="CIFAR10", data_root=".",
                   train_mode=True, crop_size=32, padding=4):

    if dataset == 'ILSVRC2012':
        return create_imagenet_dataset(dataset, data_root,
                                       train_mode, crop_size)

    if dataset == 'tiny-imagenet-200':
        return create_tiny_imagenet_dataset(dataset, data_root,
                                            train_mode, crop_size)

    if dataset == 'MNIST':
        crop_size = 28
        padding = 4

    convert = get_transformer(dataset)

    if train_mode:
        ops = [T.RandomCrop(crop_size), convert]

        if dataset != 'MNIST':
            ops = [T.RandomHorizontalFlip()] + ops

        convert = T.Compose(ops)

    ds = getattr(datasets, dataset)(data_root,
                                    train=train_mode,
                                    download=True,
                                    transform=convert)

    if train_mode:
        if dataset != 'MNIST':
            ds.train_data = np.pad(ds.train_data,
                                   ((0, 0), (padding, padding),
                                    (padding, padding), (0, 0)),
                                   mode='reflect')
    return ds
