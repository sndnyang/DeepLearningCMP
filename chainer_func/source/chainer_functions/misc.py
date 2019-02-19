import chainer.functions as F


def call_bn(bn, x, test=False, update_batch_stats=True):
    if test:
        return F.fixed_batch_normalization(x, bn.gamma, bn.beta, bn.avg_mean, bn.avg_var)
    elif not update_batch_stats:
        return F.batch_normalization(x, bn.gamma, bn.beta)
    else:
        return bn(x)
