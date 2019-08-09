import chainer
import chainer.functions as cfunc
from chainer import Variable, optimizers, cuda, serializers


def loss_test(forward, x, t):
    logit = forward(x, train=False)
    L, acc = cfunc.softmax_cross_entropy(logit, t).data, cfunc.accuracy(logit, t).data
    return L, acc


def evaluate_classifier(classifier, test_set, args):
    with chainer.using_config("train", False):
        acc_test_sum = 0
        test_loss = 0
        test_x, test_t = test_set.get()
        n_test = test_x.shape[0]
        for i in range(0, n_test, 256):
            x = test_x[i:i + 256]
            t = test_t[i:i + 256]
            if args.gpu_id > -1:
                x, t = cuda.to_gpu(x), cuda.to_gpu(t)
            loss, acc = loss_test(classifier, Variable(x), Variable(t))
            acc_test_sum += acc * x.shape[0]
            test_loss += loss * x.shape[0]
    return acc_test_sum, n_test - acc_test_sum, test_loss / n_test
