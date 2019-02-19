import math
import chainer
import chainer.functions as F
import chainer.links as L


def call_bn(bn, x, test=False, update_batch_stats=True):
    if test:
        return F.fixed_batch_normalization(x, bn.gamma, bn.beta, bn.avg_mean, bn.avg_var)
    elif not update_batch_stats:
        return F.batch_normalization(x, bn.gamma, bn.beta)
    else:
        return bn(x)


class MLP(chainer.Chain):
    def __init__(self, n_outputs=10, dropout_rate=0.5, top_bn=False):
        self.dropout_rate = dropout_rate
        self.top_bn = top_bn
        initializer = chainer.initializers.HeNormal(math.sqrt(5))
        super(MLP, self).__init__(
            l_cl=L.Linear(784, 1200, initialW=initializer),
            l_c2=L.Linear(1200, 1200, initialW=initializer),
            l_c3=L.Linear(128, n_outputs, initialW=initializer),
            bn1=L.BatchNormalization(1200),
            bn2=L.BatchNormalization(1200),
        )
        if top_bn:
            self.add_link('bn_cl', L.BatchNormalization(n_outputs))

    def __call__(self, x, train=True, update_batch_stats=True):
        h = x
        h = self.l_c1(h)
        h = F.relu(call_bn(self.bn1, h, test=not train, update_batch_stats=update_batch_stats))
        h = self.l_c2(h)
        h = F.relu(call_bn(self.bn2, h, test=not train, update_batch_stats=update_batch_stats))
        h = self.l_c3(h)
        logit = self.l_cl(h)
        if self.top_bn:
            logit = call_bn(self.bn_cl, logit, test=not train, update_batch_stats=update_batch_stats)
        return logit
