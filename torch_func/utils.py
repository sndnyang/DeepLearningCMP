import os
import sys
import random

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as nfunc


def call_bn(bn, x, update_batch_stats=True):
    if bn.training is False:
        return bn(x)
    elif not update_batch_stats:
        return nfunc.batch_norm(x, None, None, bn.weight, bn.bias, True, bn.momentum, bn.eps)
    else:
        return bn(x)


def set_framework_seed(seed, debug=False):
    if debug:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    _ = torch.manual_seed(seed)
    if torch.cuda.is_available():
        _ = torch.cuda.manual_seed(seed)
        # if use multi-GPUs, maybe it's required
        # torch.cuda.manual_seed_all(seed)


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
        m.weight = torch.nn.parameter.Parameter(tensor, requires_grad=True)
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
        m.weight = torch.nn.parameter.Parameter(tensor, requires_grad=True)
        if m.bias is not None:
            m.bias.data.zero_()


def adjust_learning_rate(optimizer, epoch, args, group=1):
    """Sets the learning rate from start_epoch linearly to zero at the end"""
    if epoch < args.epoch_decay_start:
        return args.lr
    lr = float(args.num_epochs - epoch) / (args.num_epochs - args.epoch_decay_start) * args.lr
    if args.dataset == "cifar10":
        lr *= 0.2
    for i, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = lr
        if group > 1 and i > 0:
            continue
        param_group['betas'] = (0.5, 0.999)
    return lr


def load_checkpoint_by_marker(args, exp_marker):
    base_dir = os.path.join(os.environ['HOME'], 'project/runs') if not args.log_dir else args.log_dir
    dir_path = os.path.join(base_dir, exp_marker)
    c = 0
    file_name = ""
    # example 060708/80 -> dir path contains 060708 and model_80.pth
    parts = args.resume.split("@")
    for p in os.listdir(dir_path):
        if parts[0] in p:
            c += 1
            if c == 2:
                print("can't resume, find 2")
                sys.exit(-1)
            file_name = os.path.join(dir_path, p, "model.pth" if len(parts) == 1 else "model_%s.pth" % parts[1])
    if file_name == "":
        print("can't resume, find 0")
        sys.exit(-1)
    checkpoint = torch.load(file_name)
    return checkpoint


def l2_normalize(d):
    t = d.clone()    # remove from the computing graph
    # norm = torch.sqrt(torch.sum(t.view(d.shape[0], -1) ** 2, dim=1))
    norm = torch.norm(t.view(d.shape[0], -1), p=2, dim=1)
    if len(t.shape) < 1:
        raise NotImplementedError
    # shape [128] -> [128, 1, 1, 1] to match the images' shape [128, c, h, w]
    target_shape = [1] * len(t.shape)
    target_shape[0] = -1
    normed_d = t / (norm.view(target_shape) + 1e-10)
    return normed_d


def entropy(logits):
    p = nfunc.softmax(logits, dim=1)
    return torch.mean(torch.sum(p * nfunc.log_softmax(logits, dim=1), dim=1))


def show_image(image, args):
    if isinstance(image, torch.Tensor):
        if image.device.type == "cuda":
            image = image.cpu().detach()
        image = image.numpy()

    if args.dataset == "mnist":
        image = image[0]
    elif args.dataset == "svhn":
        image = image.transpose((1, 2, 0))
        if args.data_dir != "data1.0":
            image += 0.5
    else:
        image = image.transpose((1, 2, 0))

    c_bar = plt.imshow(image)
    plt.colorbar(c_bar)
    plt.show()
    plt.close()


def stat_of_model(model):
    l = 0
    for p in model.parameters():
        print("layer weight", p.sum())
        print("layer grad ", p.grad.sum())
        l += 1
        if l > 3:
            break
