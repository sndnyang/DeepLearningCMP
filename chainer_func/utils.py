import chainer
import chainer.functions as F
from chainer import Variable, optimizers, cuda, serializers


def kl_binary(p_logit, q_logit):
    if isinstance(p_logit, chainer.Variable):
        xp = cuda.get_array_module(p_logit.data)
    else:
        xp = cuda.get_array_module(p_logit)
    p_logit = F.concat([p_logit, xp.zeros(p_logit.shape, xp.float32)], 1)
    q_logit = F.concat([q_logit, xp.zeros(q_logit.shape, xp.float32)], 1)
    return kl_categorical(p_logit, q_logit)


def kl_categorical(p_logit, q_logit):
    if isinstance(p_logit, chainer.Variable):
        xp = cuda.get_array_module(p_logit.data)
    else:
        xp = cuda.get_array_module(p_logit)
    p = F.softmax(p_logit)
    _kl = F.sum(p * (F.log_softmax(p_logit) - F.log_softmax(q_logit)), 1)
    return F.sum(_kl) / xp.prod(xp.array(_kl.shape))


def kl_no_ent_term(p_logit, q_logit):
    if isinstance(p_logit, chainer.Variable):
        xp = cuda.get_array_module(p_logit.data)
    else:
        xp = cuda.get_array_module(p_logit)
    p = F.softmax(p_logit)
    _kl = F.sum(- p * F.log_softmax(q_logit), 1)
    return F.sum(_kl) / xp.prod(xp.array(_kl.shape))


def cross_entropy(logit, y):
    # y should be one-hot encoded probability
    return - F.sum(y * F.log_softmax(logit)) / logit.shape[0]


def kl(p_logit, q_logit):
    if p_logit.shape[1] == 1:
        return kl_binary(p_logit, q_logit)
    else:
        return kl_categorical(p_logit, q_logit)


def distance(p_logit, q_logit, dist_type="KL"):
    if dist_type == "KL":
        return kl(p_logit, q_logit)
    elif dist_type == "KL2":
        return kl_no_ent_term(p_logit, q_logit)
    else:
        raise NotImplementedError


def entropy(p_logit):
    p = F.softmax(p_logit)
    return - F.sum(p * F.log_softmax(p_logit)) / p_logit.shape[0]


def get_normalized_vector(d, xp):
    d /= (1e-12 + xp.max(xp.abs(d), tuple(range(1, len(d.shape))), keepdims=True))
    d /= xp.sqrt(1e-6 + xp.sum(d ** 2, tuple(range(1, len(d.shape))), keepdims=True))
    return d


def call_bn(bn, x, test=False, update_batch_stats=True):
    if test:
        return F.fixed_batch_normalization(x, bn.gamma, bn.beta, bn.avg_mean, bn.avg_var)
    elif not update_batch_stats:
        return F.batch_normalization(x, bn.gamma, bn.beta)
    else:
        return bn(x)
