import os
import sys

import pickle
import numpy as np
from .source.data import Data
from .source.utils import mkdir_p, load_npz_as_dict


def load_mnist_dataset():
    path = os.path.join(os.environ['HOME'], 'project/data/dataset/mnist.pkl')
    if sys.version_info.major == 3:
        dataset = pickle.load(open(path, 'rb'), encoding="bytes")
    else:
        dataset = pickle.load(open(path, 'rb'))
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
        # _train_set_x = np.delete(_train_set_x, ind[0:s_c], 0)
        # _train_set_y = np.delete(_train_set_y, ind[0:s_c])

    l_rand_ind = np.random.permutation(train_set_x.shape[0])
    train_set_x = train_set_x[l_rand_ind].reshape((n_l, 1, 28, 28)).astype("float32")
    train_set_y = train_set_y[l_rand_ind]
    train_set_ul_x = _train_set_x[rand_ind].reshape((rand_ind.shape[0], 1, 28, 28))
    train_set_ul_y = _train_set_y[rand_ind]
    test_set_x = test_set_x.reshape((test_set_x.shape[0], 1, 28, 28))

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
    return Data(train_l['images'], train_l['labels'].astype(np.int32)), \
           Data(train_ul['images'], train_ul['labels'].astype(np.int32)), \
           Data(test['images'], test['labels'].astype(np.int32))


