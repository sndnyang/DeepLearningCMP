import torch
import numpy as np
from torch.nn.parameter import Parameter


def grad_update(ensemble, grad_theta):
    for i in range(len(ensemble)):
        m = ensemble[i].model
        theta_one = grad_theta[i]
        start = 0
        for e in m.modules():
            class_name = e.__class__.__name__
            if class_name.find('Conv') != -1:
                t = np.prod(e.weight.data.shape)
                e.weight.grad.data = theta_one[start: start+t].reshape(e.weight.data.shape)
                if e.bias is not None:
                    e.bias.grad.data = theta_one[start+t: start + t + e.weight.data.shape[0]]
                start += t + e.weight.data.shape[0]
            if class_name.find('Linear') != -1:
                t = np.prod(e.weight.data.shape)
                e.weight.grad.data = theta_one[start: start+t].reshape(e.weight.data.shape)
                if e.bias is not None:
                    e.bias.grad.data = theta_one[start+t: start + t + e.weight.data.shape[0]]
                start += t + e.weight.data.shape[0]


def model_pack_weight(m):
    paras = []
    t = m.model.parameters()
    for p in t:
        layer_parameter = p.view(-1)
        paras.append(layer_parameter)
    var = m.vars
    if var:
        paras.append(var)
    paras = torch.cat(paras)
    return paras


def pack_weight_var(ensemble):
    theta = []
    for m in ensemble:
        paras = model_pack_weight(m)
        theta.append(paras)
    theta = torch.stack(theta)
    return theta


def pack_grad_var(ensemble):
    theta = []
    for m in ensemble:
        t = m.model.parameters()
        paras = []
        for p in t:
            layer_parameter = p.grad.view(-1)
            paras.append(layer_parameter)
        if m.vars:
            paras.append(m.vars.grad)
        paras = torch.cat(paras)
        theta.append(paras)
    theta = torch.stack(theta)
    return theta


def weight_update(model, theta):
    start = 0
    for e in model.modules():
        class_name = e.__class__.__name__
        if class_name.find('Conv') != -1:
            t = np.prod(e.weight.data.shape)
            e.weight.data = theta[start: start + t].reshape(e.weight.shape)
            if e.bias is not None:
                e.bias.data = theta[start + t: start + t + e.weight.shape[0]]
            start += t + e.weight.data.shape[0]
        if class_name.find('Linear') != -1:
            t = np.prod(e.weight.data.shape)
            e.weight.data = theta[start: start + t].reshape(e.weight.shape)
            if e.bias is not None:
                e.bias.data = theta[start + t: start + t + e.weight.shape[0]]
            start += t + e.weight.data.shape[0]


def weights_init_xavier(m):
    """
    initialize normal distribution weight matrix
    and set bias to 0
    :param m:
    :return:
    """
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    if class_name.find('Linear') != -1:
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()


def weights_init(m):
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
    if class_name.find('BatchNorm') != -1:
        transpose = np.ones(m.weight.data.shape).astype("float32")
        tensor = torch.from_numpy(transpose)
        m.weight = Parameter(tensor, requires_grad=True)
        if m.bias is not None:
            m.bias.data.zero_()


def cumulative_batch(cum_loss, loss, cum_metrics, a_metrics, size, final=False):
    if isinstance(loss, float):
        cum_loss += loss
    else:
        cum_loss += loss.item()
    status = "Loss: {:.6f}".format(cum_loss / size)
    for k in a_metrics:
        cum_metrics[k] += a_metrics[k]
        status += "  {}: {:.5f}".format(k, cum_metrics[k] / size)
        if final:
            cum_metrics[k] = cum_metrics[k] / size
    return status, cum_loss, cum_metrics
