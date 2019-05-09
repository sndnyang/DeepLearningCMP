import sys

import numpy as np


def load_mnist_dataset():
    if sys.version_info.major == 3:
        import cPickle as pickle
        dataset = pickle.load(open('dataset/mnist.pkl', 'rb'), encoding="bytes")
    else:
        import pickle
        dataset = pickle.load(open('dataset/mnist.pkl', 'rb'))
    train_set_x = np.concatenate((dataset[0][0], dataset[1][0]), axis=0).astype("float32")
    train_set_y = np.concatenate((dataset[0][1], dataset[1][1]), axis=0)
    return (train_set_x, train_set_y), (dataset[2][0], dataset[2][1])


def load_mnist_for_semi_sup(n_l=1000, n_v=1000):
    dataset = load_mnist_dataset()

    _train_set_x, _train_set_y = dataset[0]
    test_set_x, test_set_y = dataset[1]
    test_set_x = test_set_x.reshape(test_set_x.shape[0], 1, 28, 28)

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
        # remove them from the set
        _train_set_x = np.delete(_train_set_x, ind[0:s_c], 0)
        _train_set_y = np.delete(_train_set_y, ind[0:s_c])

    l_rand_ind = np.random.permutation(train_set_x.shape[0])  # shuffle from uniform sequence to random permutation
    train_set_x = train_set_x[l_rand_ind].astype("float32").reshape(l_rand_ind.shape[0], 1, 28, 28)
    train_set_y = train_set_y[l_rand_ind]

    valid_set_x = _train_set_x[:n_v]
    valid_set_x = valid_set_x.reshape(n_v, 1, 28, 28)
    valid_set_y = _train_set_y[:n_v]
    train_set_ul_x = _train_set_x[n_v:]
    train_set_ul_x = train_set_ul_x.reshape(train_set_ul_x.shape[0], 1, 28, 28)
    train_set_ul_y = _train_set_y[n_v:]
    # Will unlabeled set contain labeled points?
    # train_set_ul_x = np.concatenate((train_set_x, _train_set_x[n_v:]), axis=0)
    # train_set_ul_y = np.concatenate((train_set_y, _train_set_y[n_v:]), axis=0)
    # train_set_ul_x = train_set_ul_x[np.random.permutation(train_set_ul_x.shape[0])]
    # ul_y is useless
    # train_set_ul_y = train_set_ul_y[np.random.permutation(train_set_ul_x.shape[0])]

    return (train_set_x, train_set_y), (train_set_ul_x, train_set_ul_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)


def load_dataset(dir_path, size=0, valid_size=0):
    if 'mnist' in dir_path:
        train_l, train_ul, val_set, test_set = load_mnist_for_semi_sup(n_l=size, n_v=valid_size)
        train_l = {'images': train_l[0], 'labels': train_l[1]}
        train_ul = {'images': train_ul[0], 'labels': train_ul[1]}
        # use val_set or test_set
        test = {'images': test_set[0], 'labels': test_set[1]}
    else:
        raise NotImplementedError
    return Data(train_l['images'], train_l['labels'].astype(np.int32)), \
        Data(train_ul['images'], train_ul['labels'].astype(np.int32)), \
        Data(test['images'], test['labels'].astype(np.int32))
