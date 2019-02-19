import os
import sys

import numpy as np
from .source.data import Data
from .source.utils import mkdir_p, load_npz_as_dict


def load_mnist_dataset():
    if sys.version_info.major == 3:
        dataset = pickle.load(open('dataset/mnist.pkl', 'rb'), encoding="bytes")
    else:
        dataset = pickle.load(open('dataset/mnist.pkl', 'rb'))
    train_set_x = np.concatenate((dataset[0][0], dataset[1][0]), axis=0)
    train_set_y = np.concatenate((dataset[0][1], dataset[1][1]), axis=0)
    return (train_set_x, train_set_y), (dataset[2][0], dataset[2][1])


def load_dataset(dirpath, valid=False, dataset_seed=1):
    if valid:
        train_l = load_npz_as_dict(os.path.join(dirpath, 'seed' + str(dataset_seed), 'labeled_train_valid.npz'))
        train_ul = load_npz_as_dict(os.path.join(dirpath, 'seed' + str(dataset_seed), 'unlabeled_train_valid.npz'))
        test = load_npz_as_dict(os.path.join(dirpath, 'seed' + str(dataset_seed), 'test_valid.npz'))
    else:
        train_l = load_npz_as_dict(os.path.join(dirpath, 'seed' + str(dataset_seed), 'labeled_train.npz'))
        train_ul = load_npz_as_dict(os.path.join(dirpath, 'seed' + str(dataset_seed), 'unlabeled_train.npz'))
        test = load_npz_as_dict(os.path.join(dirpath, 'seed' + str(dataset_seed), 'test.npz'))
    if 'mnist' in dirpath:
        train_set, test_set = load_mnist_dataset()

    train_l['images'] = train_l['images'].reshape(train_l['images'].shape[0], 3, 32, 32).astype(np.float32)
    train_ul['images'] = train_ul['images'].reshape(train_ul['images'].shape[0], 3, 32, 32).astype(np.float32)
    test['images'] = test['images'].reshape(test['images'].shape[0], 3, 32, 32).astype(np.float32)
    return Data(train_l['images'], train_l['labels'].astype(np.int32)), \
           Data(train_ul['images'], train_ul['labels'].astype(np.int32)), \
           Data(test['images'], test['labels'].astype(np.int32))


