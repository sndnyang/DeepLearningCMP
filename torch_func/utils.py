import torch
import numpy as np
import torch.nn.functional as nfunc
from torch.nn.parameter import Parameter


def call_bn(bn, x, update_batch_stats=True):
    if bn.training is False:
        return bn(x)
    elif not update_batch_stats:
        return nfunc.batch_norm(x, None, None, bn.weight, bn.bias, True, bn.momentum, bn.eps)
    else:
        return bn(x)


def set_framework_seed(seed, debug=False):
    if debug:
        # torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    _ = torch.manual_seed(seed)
    if torch.cuda.is_available():
        _ = torch.cuda.manual_seed(seed)


def weights_init_uniform(m):
    """
    initialize normal distribution weight matrix
    and set bias to 0
    :param m:
    :return:
    """
    class_name = m.__class__.__name__
    fan_in = 0
    if class_name.find('Conv') != -1:
        shape = m.weight.data.shape
        fan_in = shape[1] * shape[2] * shape[3]
    if class_name.find('Linear') != -1:
        shape = m.weight.data.shape
        fan_in = shape[1]
    if fan_in:
        s = 1.0 * np.sqrt(6.0 / fan_in)
        transpose = np.random.uniform(-s, s, m.weight.data.shape).astype("float32")
        tensor = torch.from_numpy(transpose)
        m.weight = Parameter(tensor, requires_grad=True)
        if m.bias is not None:
            m.bias.data.zero_()


def weights_init_normal(m):
    """
    initialize normal distribution weight matrix
    and set bias to 0
    :param m:
    :return:
    """
    class_name = m.__class__.__name__
    fan_in = 0
    if class_name.find('Conv') != -1:
        shape = m.weight.data.shape
        fan_in = shape[1] * shape[2] * shape[3]
    if class_name.find('Linear') != -1:
        shape = m.weight.data.shape
        fan_in = shape[1]
    if fan_in:
        s = 1.0 * np.sqrt(1.0 / fan_in)
        # in PyTorch default shape is [1200, 784]
        # compare to theano, shape is [784, 1200], I do transpose in theano for getting same outputs
        transpose = np.random.normal(0, s, m.weight.data.shape[::-1]).astype("float32").T
        tensor = torch.from_numpy(transpose)
        # print(shape, transpose.sum())
        m.weight = Parameter(tensor, requires_grad=True)
        if m.bias is not None:
            m.bias.data.zero_()
