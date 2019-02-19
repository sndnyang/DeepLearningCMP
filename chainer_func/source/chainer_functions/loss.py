from chainer import Variable, optimizers, cuda, serializers

from .vat_p import vat_plus, vat_double, vat_noise, vat_pp, vat_sharp
from .utils import *


def at_loss(forward, x, y, train=True, epsilon=8.0):
    ce = cross_entropy(forward(x, train=train, update_batch_stats=False), y)
    ce.backward()
    d = x.grad
    xp = cuda.get_array_module(x.data)
    d = get_normalized_vector(d, xp) 
    x_adv = x + epsilon * d 
    return cross_entropy(forward(x_adv, train=train, update_batch_stats=False), y)


def vat_loss(forward, distance, x, y=None, train=True, epsilon=8.0, xi=1e-6, num_iter=1, p_logit=None):
    if p_logit is None:
        p_logit = forward(x, train=train, update_batch_stats=False).data  # unchain
    else:
        assert not isinstance(p_logit, Variable)

    xp = cuda.get_array_module(x.data)
    d = xp.random.normal(size=x.shape)
    d = get_normalized_vector(d, xp)
    for ip in range(num_iter):
        x_d = Variable(x.data + xi * d.astype(xp.float32))
        p_d_logit = forward(x_d, train=train, update_batch_stats=False)
        kl_loss = distance(p_logit, p_d_logit)
        # print(kl_loss)
        kl_loss.backward()
        d = x_d.grad
        d = d / xp.sqrt(xp.sum(d ** 2, axis=tuple(range(1, len(d.shape))), keepdims=True))
    x_adv = x + epsilon * d
    p_adv_logit = forward(x_adv, train=train, update_batch_stats=False)
    pos_cost = distance(p_logit, p_adv_logit)
    # print(pos_cost)
    return pos_cost
