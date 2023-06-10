import torchvision.transforms as T
from torchvision import datasets
import numpy as np
from revolver.utils.transformer import get_transformer
import os


def create_imagenet_dataset(dataset="ILSVRC2012", data_root=".",
                            train_mode=True, crop_size=224):
    datadir = os.path.join(dataset, 'val')

    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])

    if train_mode:
        datadir = os.path.join(data_root, dataset, 'train')
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

    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])

    if train_mode:
        datadir = os.path.join(data_root, dataset, 'train/images')
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
                   train_mode=True, is_validation=False,
                   crop_size=32, padding=4, horizontal_flip=False):
    if dataset == 'ILSVRC2012':
        # TODO handle validation set as well
        return create_imagenet_dataset(dataset, data_root,
                                       train_mode, crop_size)

    if dataset == 'tiny-imagenet-200':
        return create_tiny_imagenet_dataset(dataset, data_root,
                                            train_mode, crop_size)

    if dataset == 'MNIST':
        crop_size = 28

    convert = get_transformer(dataset)

    if train_mode:
        ops = [T.RandomCrop(crop_size), convert]

        if horizontal_flip and dataset != 'MNIST':
            ops = [T.RandomHorizontalFlip()] + ops

        convert = T.Compose(ops)

    if dataset == 'SVHN':
        ds = datasets.SVHN(data_root, split='train' if train_mode or is_validation else 'test',
                           download=True, transform=convert)
    else:
        ds = getattr(datasets, dataset)(data_root,
                                        train=train_mode or is_validation,
                                        download=True,
                                        transform=convert)
        if train_mode:
            if dataset != 'MNIST':
                ds.data = np.pad(ds.data,
                                 ((0, 0), (padding, padding),
                                  (padding, padding), (0, 0)),
                                 mode='reflect')
    return ds
