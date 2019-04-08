import os
import sys

import pickle
import numpy as np


def load_npz_as_dict(path):
    data = np.load(path)
    return {key: data[key] for key in data}


def augmentation(images, random_crop=True, random_flip=True):
    # random crop and random flip
    h, w = images.shape[2], images.shape[3]
    pad_size = 2
    aug_images = []
    padded_images = np.pad(images, ((0, 0), (0, 0), (pad_size, pad_size), (pad_size, pad_size)), 'reflect')
    for image in padded_images:
        if random_flip:
            image = image[:, :, ::-1] if np.random.uniform() > 0.5 else image
        if random_crop:
            offset_h = np.random.randint(0, 2 * pad_size)
            offset_w = np.random.randint(0, 2 * pad_size)
            image = image[:, offset_h:offset_h + h, offset_w:offset_w + w]
        else:
            image = image[:, pad_size:pad_size + h, pad_size:pad_size + w]
        aug_images.append(image)
    ret = np.stack(aug_images)
    assert ret.shape == images.shape
    return ret


def _augmentation(images, trans=True, flip=True):
    # shape of `image' [N, K, W, H]
    assert images.ndim == 4
    return augmentation(images, trans, flip)


class Data2:
    def __init__(self, data, label):
        self.data = data
        self.label = label
        self.index = np.arange(self.N)

    @property
    def N(self):
        return len(self.data)

    def get(self, n=None, shuffle=True, aug_trans=False, aug_flip=False):
        if shuffle:
            ind = np.random.permutation(self.data.shape[0])
        else:
            ind = np.arange(self.data.shape[0])
        if n is None:
            n = self.data.shape[0]
        index = ind[:n]
        batch_data = self.data[index]
        batch_label = self.label[index]
        if aug_trans or aug_flip:
            batch_data = _augmentation(batch_data, aug_trans, aug_flip)
        return batch_data, batch_label


class Data:
    def __init__(self, data, label):
        self.data = data
        self.label = label
        self.ind = np.arange(self.data.shape[0])
        self.index = 0

    @property
    def N(self):
        return len(self.data)

    def get(self, n=None, aug_trans=False, aug_flip=False):
        if n is None:
            n = self.data.shape[0]
        end = self.index + n
        if end >= len(self.label):
            index = self.ind[self.index:]
            self.ind = np.random.permutation(self.data.shape[0])
            remain = end % len(self.label)
            if remain:
                index = np.concatenate([index, self.ind[:remain]])
        else:
            index = self.ind[self.index:end]
        batch_data = self.data[index]
        batch_label = self.label[index]
        self.index = end % self.data.shape[0]
        if aug_trans or aug_flip:
            batch_data = _augmentation(batch_data, aug_trans, aug_flip)

        return batch_data, batch_label


def load_mnist_dataset():
    if sys.version_info.major == 3:
        dataset = pickle.load(open('dataset/mnist.pkl', 'rb'), encoding="bytes")
    else:
        dataset = pickle.load(open('dataset/mnist.pkl', 'rb'))
    train_set_x = np.concatenate((dataset[0][0], dataset[1][0]), axis=0).astype("float32")
    train_set_y = np.concatenate((dataset[0][1], dataset[1][1]), axis=0)
    return (train_set_x, train_set_y), (dataset[2][0], dataset[2][1])


def load_mnist_for_semi_sup(n_l=1000, n_v=1000):
    dataset = load_mnist_dataset()

    _train_set_x, _train_set_y = dataset[0]
    test_set_x, test_set_y = dataset[1]

    rand_ind = np.random.permutation(_train_set_x.shape[0])
    _train_set_x = _train_set_x[rand_ind]
    _train_set_y = _train_set_y[rand_ind]

    s_c = int(n_l / 10.0)
    train_set_x = np.zeros((n_l, 28 ** 2))
    train_set_y = np.zeros(n_l)
    for i in range(10):
        ind = np.where(_train_set_y == i)[0]
        train_set_x[i * s_c:(i + 1) * s_c, :] = _train_set_x[ind[0:s_c], :]
        train_set_y[i * s_c:(i + 1) * s_c] = _train_set_y[ind[0:s_c]]
        _train_set_x = np.delete(_train_set_x, ind[0:s_c], 0)
        _train_set_y = np.delete(_train_set_y, ind[0:s_c])

    l_rand_ind = np.random.permutation(train_set_x.shape[0])
    train_set_x = train_set_x[l_rand_ind].astype("float32")
    train_set_y = train_set_y[l_rand_ind]
    train_set_ul_x = np.concatenate((train_set_x, _train_set_x[n_v:]), axis=0)
    train_set_ul_x = train_set_ul_x[np.random.permutation(train_set_ul_x.shape[0])]
    train_set_ul_y = np.concatenate((train_set_y, _train_set_y[n_v:]), axis=0)

    return (train_set_x, train_set_y), (train_set_ul_x, train_set_ul_y), (test_set_x, test_set_y)


def load_dataset(dirpath, valid=False, dataset_seed=1, size=0):
    if 'mnist' in dirpath:
        train_l, train_ul, test_set = load_mnist_for_semi_sup(n_l=size)
        train_l = {'images': train_l[0], 'labels': train_l[1]}
        train_ul = {'images': train_ul[0], 'labels': train_ul[1]}
        test = {'images': test_set[0], 'labels': test_set[1]}
    else:
        if valid:
            train_l = load_npz_as_dict(os.path.join(dirpath, 'seed' + str(dataset_seed), 'labeled_train_valid.npz'))
            train_ul = load_npz_as_dict(os.path.join(dirpath, 'seed' + str(dataset_seed), 'unlabeled_train_valid.npz'))
            test = load_npz_as_dict(os.path.join(dirpath, 'seed' + str(dataset_seed), 'test_valid.npz'))
        else:
            train_l = load_npz_as_dict(os.path.join(dirpath, 'seed' + str(dataset_seed), 'labeled_train.npz'))
            train_ul = load_npz_as_dict(os.path.join(dirpath, 'seed' + str(dataset_seed), 'unlabeled_train.npz'))
            test = load_npz_as_dict(os.path.join(dirpath, 'seed' + str(dataset_seed), 'test.npz'))
        train_l['images'] = train_l['images'].reshape(train_l['images'].shape[0], 3, 32, 32).astype(np.float32)
        train_ul['images'] = train_ul['images'].reshape(train_ul['images'].shape[0], 3, 32, 32).astype(np.float32)
        test['images'] = test['images'].reshape(test['images'].shape[0], 3, 32, 32).astype(np.float32)
    # mnist 100 use data
    # larger data set should use Data2 as original one
    return Data(train_l['images'], train_l['labels'].astype(np.int32)), \
           Data(train_ul['images'], train_ul['labels'].astype(np.int32)), \
           Data(test['images'], test['labels'].astype(np.int32))
