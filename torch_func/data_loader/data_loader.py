import os
import sys

import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data.sampler import Sampler
from torch.utils.data import TensorDataset, DataLoader


def split_data(train_data_set, sizes=(15000, 20000)):
    """
    split training data set into several subsets.
    :param train_data_set:
    :param sizes: [0.5, 0.8], [15000, 20000], []
    :return: a list of subsets of data
    """
    x, y = train_data_set
    size = x.shape[0]
    if not isinstance(sizes, (list, tuple)):
        print("sizes are not list or tuple")
        sys.exit(-1)

    sizes = [0] + sizes + [1]
    split_sets = []
    index = 0
    for i in range(1, len(sizes)):
        if sizes[i] < 1:
            next_index = int(size * sizes[i])
        elif sizes[i] == 1:
            next_index = size
        else:
            next_index = sizes[i]
        split_sets.append([x[index:next_index], y[index:next_index]])
        index = next_index
    return split_sets


def split_val_pool_data(train_data_set, sizes=(15000, 20000)):
    """
    split validate and pool data set
    :param train_data_set:
    :param sizes: (15000, 20000)
    :return: 3 data sets
    """
    return split_data(train_data_set, sizes)


def data_set_name(name, norm=1):
    func_map = {
        "mnist": mnist_set,
        "cifar10": cifar10_set,
        "cifar100": cifar100_set,
        "fashion": fashion_set,
        "svhn": svhn_set,
    }
    return func_map[name](norm)


def load_data_set(name, norm=True):
    func_map = {
        "mnist": load_mnist,
        "cifar10": load_cifar10,
        "cifar100": load_cifar100,
        "fashion": load_fashion,
        "svhn": load_svhn,
    }
    return func_map[name](norm)


def load_npz_as_dict(path):
    data = np.load(path)
    return {key: data[key] for key in data}


def load_dataset_semi(dataset, valid=False, dataset_seed=1):
    dirpath = os.path.join(os.environ['HOME'], 'project/data/dataset/', dataset)
    if valid:
        train_l = load_npz_as_dict(os.path.join(dirpath, 'seed' + str(dataset_seed), 'labeled_train_valid.npz'))
        train_ul = load_npz_as_dict(os.path.join(dirpath, 'seed' + str(dataset_seed), 'unlabeled_train_valid.npz'))
        test = load_npz_as_dict(os.path.join(dirpath, 'seed' + str(dataset_seed), 'test_valid.npz'))
    else:
        train_l = load_npz_as_dict(os.path.join(dirpath, 'seed' + str(dataset_seed), 'labeled_train.npz'))
        train_ul = load_npz_as_dict(os.path.join(dirpath, 'seed' + str(dataset_seed), 'unlabeled_train.npz'))
        test = load_npz_as_dict(os.path.join(dirpath, 'seed' + str(dataset_seed), 'test.npz'))

    c, w, h = (3, 32, 32)
    if dataset in ['mnist', 'fashion']:
        c, w, h = (1, 28, 28)

    train_l['images'] = torch.from_numpy(train_l['images'].reshape(train_l['images'].shape[0], c, w, h).astype(np.float32))
    train_ul['images'] = torch.from_numpy(train_ul['images'].reshape(train_ul['images'].shape[0], c, w, h).astype(np.float32))
    test['images'] = torch.from_numpy(test['images'].reshape(test['images'].shape[0], c, w, h).astype(np.float32))
    train_l['labels'] = torch.LongTensor(train_l['labels'])
    train_ul['labels'] = torch.LongTensor(train_ul['labels'])
    test['labels'] = torch.LongTensor(test['labels'])
    t_l_set = TensorDataset(train_l['images'], train_l['labels'])
    t_ul_set = TensorDataset(train_ul['images'], train_ul['labels'])
    test_set = TensorDataset(test['images'], test['labels'])
    num_classes = 10
    if dataset == 'cifar100':
        num_classes = 100
    if dataset == 'imagenet':
        num_classes = 1000

    return t_l_set, t_ul_set, test_set, (c, w, h), num_classes


def mnist_set(norm=1):
    if norm == 1:
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    elif norm == 2:
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x / 256)
        ])
    else:
        trans = transforms.Compose([transforms.ToTensor()])

    dir_path = os.path.join(os.environ['HOME'], 'project/data/mnist')
    train_set = datasets.MNIST(dir_path, train=True, download=True, transform=trans)
    test_set = datasets.MNIST(dir_path, train=False, transform=trans)
    return train_set, test_set, (1, 28, 28), 10


def fashion_set(norm=1):
    if norm == 1:
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    elif norm == 2:
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x / 256)
        ])
    else:
        trans = transforms.Compose([transforms.ToTensor()])
    dir_path = os.path.join(os.environ['HOME'], 'project/data/fashion')
    train_set = datasets.FashionMNIST(dir_path, train=True, download=True, transform=trans)
    test_set = datasets.FashionMNIST(dir_path, train=False, transform=trans)
    return train_set, test_set, (1, 28, 28), 10


def cifar10_set(norm=1):
    if norm == 1:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    elif norm == 2:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x / 256)
        ])
        transform_test = transform_train
    else:
        transform_train = transforms.Compose([transforms.ToTensor()])
        transform_test = transforms.Compose([transforms.ToTensor()])
    dir_path = os.path.join(os.environ['HOME'], 'project/data/cifar10')
    train_set = datasets.CIFAR10(dir_path, train=True, download=True, transform=transform_train)
    test_set = datasets.CIFAR10(dir_path, train=False, transform=transform_test)
    return train_set, test_set, (3, 32, 32), 10


def cifar100_set(norm=1):
    if norm == 1:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    else:
        transform = transforms.Compose([transforms.ToTensor()])
    dir_path = os.path.join(os.environ['HOME'], 'project/data/cifar100')
    train_set = datasets.CIFAR100(dir_path, train=True, download=True, transform=transform)
    test_set = datasets.CIFAR100(dir_path, train=False, transform=transform)
    return train_set, test_set, (3, 32, 32), 100


def svhn_set(norm=1):
    if norm == 1:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    elif norm == 2:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x / 255 - 127.5)
        ])
    else:
        transform = transforms.Compose([transforms.ToTensor()])

    dir_path = os.path.join(os.environ['HOME'], 'project/data/svhn')
    train_set = datasets.SVHN(dir_path, split="train", download=True, transform=transform)
    test_set = datasets.SVHN(dir_path, split="test", transform=transform)
    return train_set, test_set, (3, 32, 32), 10


def load_mnist(norm=True):
    c, img_rows, img_cols = 1, 28, 28
    if norm:
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        trans = transforms.Compose([transforms.ToTensor()])

    dir_path = os.path.join(os.environ['HOME'], 'project/data/mnist')
    train_set = datasets.MNIST(dir_path, train=True, download=True, transform=trans)
    test_set = datasets.MNIST(dir_path, train=False, transform=trans)
    x_train_all = []
    y_train_all = []
    for e in train_set:
        x, y = e
        x_train_all.append(x.numpy())
        y_train_all.append(y)
    x_train_all = np.array(x_train_all)
    y_train_all = np.array(y_train_all)
    x_test = []
    y_test = []
    for e in test_set:
        x, y = e
        x_test.append(x.numpy())
        y_test.append(y)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    if norm == 2:
        x_train_all = x_train_all/255
        x_test = x_test/255
    return (x_train_all, y_train_all), (x_test, y_test), (1, 28, 28), 10


def load_fashion(norm=True):
    img_rows, img_cols = 28, 28

    if norm:
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        trans = transforms.Compose([transforms.ToTensor()])
    dir_path = os.path.join(os.environ['HOME'], 'project/data/fashion')
    train_set = datasets.FashionMNIST(dir_path, train=True, download=True, transform=trans)
    test_set = datasets.FashionMNIST(dir_path, train=False, transform=trans)
    x_train_all = []
    y_train_all = []
    for e in train_set:
        x, y = e
        x_train_all.append(x.numpy())
        y_train_all.append(y)
    x_train_all = np.array(x_train_all)
    y_train_all = np.array(y_train_all)
    x_test = []
    y_test = []
    for e in test_set:
        x, y = e
        x_test.append(x.numpy())
        y_test.append(y)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    if norm == 2:
        x_train_all = x_train_all/255
        x_test = x_test/255
    return (x_train_all, y_train_all), (x_test, y_test), (1, 28, 28), 10


def load_cifar10(norm=True):
    if norm:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    else:
        transform_train = transforms.Compose([transforms.ToTensor()])
        transform_test = transforms.Compose([transforms.ToTensor()])

    dir_path = os.path.join(os.environ['HOME'], 'project/data/cifar10')
    train_set = datasets.CIFAR10(dir_path, train=True, download=True, transform=transform_train)
    test_set = datasets.CIFAR10(dir_path, train=False, transform=transform_test)
    x_train_all = []
    y_train_all = []
    for e in train_set:
        x, y = e
        x_train_all.append(x.numpy())
        y_train_all.append(y)
    x_train_all = np.array(x_train_all)
    y_train_all = np.array(y_train_all)
    x_test = []
    y_test = []
    for e in test_set:
        x, y = e
        x_test.append(x.numpy())
        y_test.append(y)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    return (x_train_all, y_train_all), (x_test, y_test), (3, 32, 32), 10


def load_cifar100(norm=True):
    if norm:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    else:
        transform = transforms.Compose([transforms.ToTensor()])

    dir_path = os.path.join(os.environ['HOME'], 'project/data/cifar100')
    train_set = datasets.CIFAR100(dir_path, train=True, download=True, transform=transform)
    test_set = datasets.CIFAR100(dir_path, train=False, transform=transform)
    x_train_all = []
    y_train_all = []
    for e in train_set:
        x, y = e
        x_train_all.append(x.numpy())
        y_train_all.append(y)
    x_train_all = np.array(x_train_all)
    y_train_all = np.array(y_train_all)
    x_test = []
    y_test = []
    for e in test_set:
        x, y = e
        x_test.append(x.numpy())
        y_test.append(y)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    return (x_train_all, y_train_all), (x_test, y_test), (3, 32, 32), 10


def load_svhn(norm=True):
    if norm:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    else:
        transform = transforms.Compose([transforms.ToTensor()])

    dir_path = os.path.join(os.environ['HOME'], 'project/data/svhn')
    train_set = datasets.SVHN(dir_path, split="train", download=True, transform=transform)
    test_set = datasets.SVHN(dir_path, split="test", transform=transform)
    x_train_all = []
    y_train_all = []
    for e in train_set:
        x, y = e
        x_train_all.append(x.numpy())
        y_train_all.append(y)
    x_train_all = np.array(x_train_all)
    y_train_all = np.array(y_train_all)
    x_test = []
    y_test = []
    for e in test_set:
        x, y = e
        x_test.append(x.numpy())
        y_test.append(y)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    return (x_train_all, y_train_all), (x_test, y_test), (3, 32, 32), 10


def prepare_uniform_data(train_data, train_target, uniform_size=2):
    train_data_set = []
    train_target_set = []
    index = []
    for i in range(0, 10):
        arr = np.array(np.where(train_target == i))
        idx = np.random.permutation(arr[0])[:uniform_size]
        index.append(idx)
        # pick the first size elements of the shuffled idx array
        # for uniform distribution
        data_i = train_data[idx]
        target_i = train_target[idx]
        train_data_set.append(data_i)
        train_target_set.append(target_i)
    index = np.concatenate(index)
    train_data_set = np.concatenate(train_data_set, axis=0).astype("float32")
    train_target_set = np.concatenate(train_target_set, axis=0)
    return train_data_set, train_target_set, index


def get_loader_from_tensor(data, target, batch_size=128, shuffle=True, **kwargs):
    data_set = TensorDataset(data, target)
    return DataLoader(data_set, batch_size=batch_size, shuffle=shuffle, **kwargs)


def get_loader_from_np(x, y, batch_size=128, shuffle=True, **kwargs):
    data_set = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(data_set, batch_size=batch_size, shuffle=shuffle, **kwargs)


class InfiniteSampler(Sampler):
    def __init__(self, num_samples):
        super(InfiniteSampler, self).__init__(num_samples)
        self.num_samples = num_samples

    def __iter__(self):
        return iter(self.loop())

    def __len__(self):
        return 2 ** 31

    def loop(self):
        i = 0
        order = np.arange(self.num_samples)
        while True:
            yield order[i]
            i += 1
            if i >= self.num_samples:
                order = np.random.permutation(self.num_samples)
                i = 0


def init_infinite_iter(x, y, batch_size=128, **kwargs):
    data_set = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return iter(DataLoader(data_set, batch_size, sampler=InfiniteSampler(len(x)), **kwargs))
