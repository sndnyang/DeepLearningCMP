import chainer
import chainer.functions as F
import numpy as np
from chainer import Variable, optimizers, cuda, serializers

from .utils import get_normalized_vector


def kl_div_batch(p_logit, q_logit):
    p = F.softmax(p_logit)
    _kl = F.sum(p * (F.log_softmax(p_logit) - F.log_softmax(q_logit)), 1)
    # return F.sum(_kl) / xp.prod(xp.array(_kl.shape))
    return _kl


def vat_plus(forward, distance, x, t=None, train=True, epsilon=8.0, xi=1e-6, Ip=1, p_logit=None, dif=False):
    if p_logit is None:
        p_logit = forward(x, train=train, update_batch_stats=False).data  # unchain
    else:
        assert not isinstance(p_logit, Variable)

    y = F.argmax(p_logit, axis=1)
    xp = cuda.get_array_module(x.data)
    d = xp.random.normal(size=x.shape)
    d = get_normalized_vector(d, xp) 
    for ip in range(Ip):
        x_d = Variable(x.data + xi * d.astype(xp.float32))
        p_d_logit = forward(x_d, train=train, update_batch_stats=False)
        kl_loss = distance(p_logit, p_d_logit)
        kl_loss.backward()
        d = x_d.grad
        d = d / xp.sqrt(xp.sum(d ** 2, axis=tuple(range(1, len(d.shape))), keepdims=True))
    x_adv = x + epsilon * d
    p_adv_logit = forward(x_adv, train=train, update_batch_stats=False)
    if t is None:
        t = y

    eq = np.equal(y.data, t.data) * 1
    correct_num = eq.sum()
    vadv_cost = F.sum(eq * kl_div_batch(p_logit, p_adv_logit)) / correct_num if correct_num else 0

    return vadv_cost


def vat_pp(forward, distance, x, t=None, train=True, epsilon=8.0, neg_eps=8.0, correct_lamb=1.0, xi=1e-6, Ip=1, p_logit=None, dif=False):
    if p_logit is None:
        p_logit = forward(x, train=train, update_batch_stats=False).data  # unchain
    else:
        assert not isinstance(p_logit, Variable)

    y = F.argmax(p_logit, axis=1)
    xp = cuda.get_array_module(x.data)
    d = xp.random.normal(size=x.shape)
    d = get_normalized_vector(d, xp)
    for ip in range(Ip):
        x_d = Variable(x.data + xi * d.astype(xp.float32))
        p_d_logit = forward(x_d, train=train, update_batch_stats=False)
        kl_loss = distance(p_logit, p_d_logit)
        kl_loss.backward()
        d = x_d.grad
        d = d / xp.sqrt(xp.sum(d ** 2, axis=tuple(range(1, len(d.shape))), keepdims=True))
    x_adv = x + epsilon * d
    p_adv_logit = forward(x_adv, train=train, update_batch_stats=False)
    neg_x_adv = x - neg_eps * d
    p_neg_logit = forward(neg_x_adv, train=train, update_batch_stats=False)
    t_neg = t
    if t is None:
        t = y
        t_neg = y
    eq = np.equal(y.data, t.data) * 1
    vadv_cost = F.sum(eq * kl_div_batch(p_logit, p_adv_logit)) / eq.shape[0]

    neg_eq = np.equal(y.data, t_neg.data)
    neg_adv_cost = F.sum(neg_eq * kl_div_batch(p_logit, p_neg_logit)) / neg_eq.shape[0]

    # return distance(p_logit, p_adv_logit)
    return correct_lamb * (vadv_cost + neg_adv_cost)


def vat_double(forward, distance, x, y=None, train=True, epsilon=8.0, neg_eps=0.8, extra_lamb=1.0, xi=1e-6, Ip=1, p_logit=None, rev=False):
    if p_logit is None:
        p_logit = forward(x, train=train, update_batch_stats=False).data  # unchain
    else:
        assert not isinstance(p_logit, Variable)

    xp = cuda.get_array_module(x.data)
    d = xp.random.normal(size=x.shape)
    d = get_normalized_vector(d, xp)
    for ip in range(Ip):
        x_d = Variable(x.data + xi * d.astype(xp.float32))
        p_d_logit = forward(x_d, train=train, update_batch_stats=False)
        kl_loss = distance(p_logit, p_d_logit)
        kl_loss.backward()
        d = x_d.grad
        d = d / xp.sqrt(xp.sum(d ** 2, axis=tuple(range(1, len(d.shape))), keepdims=True))
    x_adv = x + epsilon * d
    p_adv_logit = forward(x_adv, train=train, update_batch_stats=False)
    neg_x_adv = x - neg_eps * d
    p_neg_logit = forward(neg_x_adv, train=train, update_batch_stats=False)
    if rev:
        pos_cost = distance(p_adv_logit, p_logit)
        neg_cost = distance(p_neg_logit, p_logit)
    else:
        pos_cost = distance(p_logit, p_adv_logit)
        neg_cost = distance(p_logit, p_neg_logit)
    return extra_lamb * (pos_cost + neg_cost)


def vat_sharp(forward, distance, x, y=None, train=True, epsilon=8.0, neg_eps=0.8, extra_lamb=1.0, xi=1e-6, Ip=1, p_logit=None):
    if p_logit is None:
        p_logit = forward(x, train=train, update_batch_stats=False).data  # unchain
    else:
        assert not isinstance(p_logit, Variable)

    xp = cuda.get_array_module(x.data)
    d = xp.random.normal(size=x.shape)
    d = get_normalized_vector(d, xp)
    for ip in range(Ip):
        x_d = Variable(x.data + xi * d.astype(xp.float32))
        p_d_logit = forward(x_d, train=train, update_batch_stats=False)
        kl_loss = distance(p_logit, p_d_logit)
        kl_loss.backward()
        d = x_d.grad
        d = d / xp.sqrt(xp.sum(d ** 2, axis=tuple(range(1, len(d.shape))), keepdims=True))
    x_adv = x + epsilon * d
    p_adv_logit = forward(x_adv, train=train, update_batch_stats=False)
    neg_x_adv = x - neg_eps * d
    p_neg_logit = forward(neg_x_adv, train=train, update_batch_stats=False)
    pos_cost = distance(p_logit, p_adv_logit)
    neg_cost = distance(p_logit, p_neg_logit)
    rev_pos_cost = distance(p_adv_logit, p_logit)
    rev_neg_cost = distance(p_neg_logit, p_logit)
    return extra_lamb * (pos_cost + neg_cost + rev_pos_cost + rev_neg_cost)


def vat_noise(forward, distance, x, y=None, train=True, epsilon=8.0, noise_eps=1.0, extra_lamb=1.0, xi=1e-6, Ip=1, p_logit=None):
    if p_logit is None:
        p_logit = forward(x, train=train, update_batch_stats=False).data  # unchain
    else:
        assert not isinstance(p_logit, Variable)

    xp = cuda.get_array_module(x.data)
    rand_pert = xp.random.normal(size=x.shape)
    d = get_normalized_vector(rand_pert, xp)
    for ip in range(Ip):
        x_d = Variable(x.data + xi * d.astype(xp.float32))
        p_d_logit = forward(x_d, train=train, update_batch_stats=False)
        kl_loss = distance(p_logit, p_d_logit)
        kl_loss.backward()
        d = x_d.grad
        d = d / xp.sqrt(xp.sum(d ** 2, axis=tuple(range(1, len(d.shape))), keepdims=True))
    x_adv = x + epsilon * d
    p_adv_logit = forward(x_adv, train=train, update_batch_stats=False)
    # noise = xp.random.normal(size=x.shape)
    noise_d = get_normalized_vector(rand_pert, xp)
    noise_adv = x + noise_eps * noise_d
    p_noise_logit = forward(noise_adv, train=train, update_batch_stats=False)
    pos_cost = distance(p_logit, p_adv_logit)
    neg_cost = extra_lamb * distance(p_logit, p_noise_logit)
    return extra_lamb * (pos_cost + neg_cost)
